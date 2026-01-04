"""
Paper Trading Simulator.
Tracks virtual positions and calculates simulated P&L without real execution.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json

from utils.logger import log
from src.utils.persistence_manager import PersistenceManager
from src.risk_manager import RiskManager
from config.settings import settings


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"
    SIGNAL_REVERSED = "signal_reversed"


@dataclass
class PaperPosition:
    """Represents a simulated paper trading position."""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    
    # Current state
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    # P&L and fees
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_fee: float = 0.0   # Fee paid on entry
    exit_fee: float = 0.0    # Fee paid on exit
    total_fees: float = 0.0  # Total fees for this trade
    
    # Break-even stop tracking
    original_stop_loss: float = 0.0  # Store original SL
    break_even_activated: bool = False
    highest_price: float = 0.0  # Track highest price since entry (for longs)
    lowest_price: float = 0.0   # Track lowest price since entry (for shorts)
    
    def __post_init__(self):
        """Initialize tracking fields."""
        if self.original_stop_loss == 0.0:
            self.original_stop_loss = self.stop_loss
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperPosition':
        """Deserialize from dictionary."""
        data['entry_time'] = datetime.fromisoformat(data['entry_time']) if data.get('entry_time') else None
        data['exit_time'] = datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None
        if 'status' in data:
            data['status'] = PositionStatus(data['status'])
        return cls(**data)
    
    def update_price(self, new_price: float) -> None:
        """Update current price, track high/low, and recalculate unrealized P&L."""
        self.current_price = new_price
        
        # Track highest/lowest prices
        if new_price > self.highest_price:
            self.highest_price = new_price
        if new_price < self.lowest_price or self.lowest_price == 0:
            self.lowest_price = new_price
        
        if self.status == PositionStatus.OPEN:
            if self.side == 'long':
                self.unrealized_pnl = (new_price - self.entry_price) * self.size
            else:  # short
                self.unrealized_pnl = (self.entry_price - new_price) * self.size
    
    def check_break_even_trigger(self) -> bool:
        """
        Check if position has reached 1R profit to trigger break-even stop.
        1R = the original risk amount (entry to original stop)
        """
        if self.break_even_activated:
            return False  # Already activated
        
        original_risk = abs(self.entry_price - self.original_stop_loss)
        if original_risk == 0: return False
        
        if self.side == 'long':
            # For longs, check if price moved up by 1R
            target_price = self.entry_price + original_risk
            return self.current_price >= target_price
        else:
            # For shorts, check if price moved down by 1R
            target_price = self.entry_price - original_risk
            return self.current_price <= target_price
    
    def move_to_break_even(self, buffer_pct: float = 0.001) -> None:
        """
        Move stop-loss to break-even (entry price + small buffer for fees).
        Buffer ensures we at least cover fees.
        """
        buffer = self.entry_price * buffer_pct
        
        if self.side == 'long':
            self.stop_loss = self.entry_price + buffer
        else:
            self.stop_loss = self.entry_price - buffer
        
        self.break_even_activated = True
    
    def close(self, exit_price: float, reason: str, 
              status: PositionStatus = PositionStatus.CLOSED,
              exit_fee: float = 0.0) -> float:
        """Close the position and return realized P&L (after fees)."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = status
        self.current_price = exit_price
        self.exit_fee = exit_fee
        self.total_fees = self.entry_fee + self.exit_fee
        
        # Calculate gross P&L
        if self.side == 'long':
            gross_pnl = (exit_price - self.entry_price) * self.size
        else:
            gross_pnl = (self.entry_price - exit_price) * self.size
        
        # Net P&L after deducting fees
        self.realized_pnl = gross_pnl - self.total_fees
        
        self.unrealized_pnl = 0.0
        return self.realized_pnl
    
    @property
    def pnl_pct(self) -> float:
        """Return P&L as a percentage of entry value."""
        entry_value = self.entry_price * self.size
        if entry_value == 0:
            return 0.0
        current_pnl = self.unrealized_pnl if self.status == PositionStatus.OPEN else self.realized_pnl
        return (current_pnl / entry_value) * 100
    
    def check_stop_loss(self) -> bool:
        """Check if stop-loss has been hit."""
        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def check_take_profit(self) -> bool:
        """Check if take-profit has been hit."""
        if self.side == 'long':
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit


@dataclass 
class TradingStats:
    """Aggregate trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0  # Total trading fees
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingStats':
        # Remove computed properties from dict if present
        clean_data = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**clean_data)


class PaperTradingSimulator:
    """
    Paper Trading Simulator with P&L tracking.
    
    Features:
    - Track virtual positions (entries, sizes, direction)
    - Calculate unrealized P&L based on current market prices  
    - Record realized P&L when positions close
    - Track trading fees (Delta Exchange: 0.04% maker, 0.06% taker)
    - Support leverage (amplifies gains and losses)
    - Auto square-off when conditions become unfavorable
    - Display running totals and statistics
    - PERSISTENCE: Saves/Loads state from JSON to survive restarts
    """
    
    # Delta Exchange trading fees (effective March 2024)
    MAKER_FEE = 0.0004  # 0.04%
    TAKER_FEE = 0.0006  # 0.06%
    
    def __init__(self, initial_balance: float = 10000.0, 
                 leverage: int = 1,
                 use_maker_fee: bool = False,
                 persistence_file: str = "data/paper_trade_state.json"):
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 fee_rate: float = 0.0006,  # 0.06% taker fee
                 persistence_dir: str = "data/paper"):
        """Initialize simulator."""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.fee_rate = fee_rate
        self.persistence = PersistenceManager(persistence_dir, "paper_trading_state")
        self.risk_manager = RiskManager()
        
        # State
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_positions: List[PaperPosition] = []
        self._position_counter = 0
        self.stats = TradingStats()
        
        # Ensure data directory exists
        import os
        data_dir = os.path.dirname(persistence_dir) # Changed from persistence_file
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            log.info(f"[PAPER] Created data directory: {data_dir}")

        self.persistence = PersistenceManager(persistence_file)
        
        fee_type = "maker (0.04%)" if use_maker_fee else "taker (0.06%)"
        log.info(f"[PAPER] Paper Trading Simulator initialized")
        
        # Try to load existing state
        if self._load_state():
            log.info(f"[PAPER] ♻️ Restored previous session state")
        else:
            log.info(f"[PAPER] ✨ Starting fresh session")
            self._save_state()  # Ensure initial state is saved

        log.info(f"[PAPER]   Balance: ${self.current_balance:.2f}")
        log.info(f"[PAPER]   Leverage: {leverage}x")
        log.info(f"[PAPER]   Fee Rate: {fee_type}")
        if leverage > 1:
            log.info(f"[PAPER]   ⚠️  With {leverage}x leverage, gains AND losses are amplified!")
    
    def _calculate_fee(self, notional_value: float) -> float:
        """Calculate trading fee for a given notional value."""
        return notional_value * self.fee_rate
    
    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"paper_{self._position_counter}"
    
    def _save_state(self) -> None:
        """Save current state to persistent storage."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'position_counter': self._position_counter,
                'stats': self.stats.to_dict(),
                'positions': {
                    sym: pos.to_dict() 
                    for sym, pos in self.positions.items()
                },
                'closed_positions': [
                    pos.to_dict() 
                    for pos in self.closed_positions[-50:]  # Keep last 50 closed positions
                ]
            }
            self.persistence.save_state(state)
        except Exception as e:
            log.error(f"[PAPER] Failed to save state: {e}")

    def _load_state(self) -> bool:
        """Load state from persistent storage."""
        state = self.persistence.load_state()
        if not state:
            return False
            
        try:
            # Restore balance (optionally restore initial_balance to track absolute P&L since day 1)
            self.initial_balance = state.get('initial_balance', self.initial_balance)
            self.current_balance = state.get('current_balance', self.current_balance)
            self._position_counter = state.get('position_counter', 0)
            
            # Restore stats
            if 'stats' in state:
                self.stats = TradingStats.from_dict(state['stats'])
            
            # Restore open positions
            positions_data = state.get('positions', {})
            self.positions = {}
            for sym, pos_data in positions_data.items():
                self.positions[sym] = PaperPosition.from_dict(pos_data)
            
            # Restore closed positions
            closed_data = state.get('closed_positions', [])
            self.closed_positions = [PaperPosition.from_dict(p) for p in closed_data]
            
            return True
        except Exception as e:
            log.error(f"[PAPER] Failed to load state: {e}")
            return False

    def open_position(self, 
                      symbol: str,
                      side: str,
                      size: float,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: float,
                      strategy: str = "unknown") -> Optional[PaperPosition]:
        """
        Open a new paper position.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSD')
            side: 'long' or 'short'
            size: Position size in base currency
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            strategy: Strategy name that generated the signal
            
        Returns:
            PaperPosition if successful, None otherwise
        """
        # Check if already have position in this symbol
        if symbol in self.positions:
            log.warning(f"[PAPER] Already have position in {symbol}, skipping")
            return None
        
        # Calculate entry fee
        entry_value = entry_price * size
        entry_fee = self._calculate_fee(entry_value)
        
        position = PaperPosition(
            id=self._generate_position_id(),
            symbol=symbol,
            side=side.lower(),
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            current_price=entry_price,
            entry_fee=entry_fee
        )
        
        self.positions[symbol] = position
        
        # Deduct entry fee from balance
        self.current_balance -= entry_fee
        self.stats.total_fees_paid += entry_fee
        
        log.info(f"[PAPER] 📈 OPENED {side.upper()} {symbol}")
        log.info(f"         Size: {size:.6f} @ ${entry_price:.2f} = ${entry_value:.2f}")
        log.info(f"         Entry Fee: ${entry_fee:.4f} ({self.fee_rate*100:.2f}%)")
        log.info(f"         SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
        log.info(f"         Strategy: {strategy}")
        
        # Save state immediately
        self._save_state()
        
        return position
    
    def close_position(self, 
                       symbol: str, 
                       exit_price: float,
                       reason: str = "Manual close",
                       status: PositionStatus = PositionStatus.CLOSED) -> Tuple[bool, float]:
        """
        Close an open position.
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            reason: Reason for closing
            status: Position status after close
            
        Returns:
            Tuple of (success, realized_pnl)
        """
        if symbol not in self.positions:
            log.warning(f"[PAPER] No position found for {symbol}")
            return False, 0.0
        
        position = self.positions[symbol]
        
        # Calculate exit fee
        exit_value = exit_price * position.size
        exit_fee = self._calculate_fee(exit_value)
        
        # Close position with exit fee
        realized_pnl = position.close(exit_price, reason, status, exit_fee)
        
        # Update stats
        self.stats.total_trades += 1
        self.stats.total_realized_pnl += realized_pnl
        self.stats.total_fees_paid += exit_fee
        
        if realized_pnl >= 0:
            self.stats.winning_trades += 1
            if realized_pnl > self.stats.largest_win:
                self.stats.largest_win = realized_pnl
        else:
            self.stats.losing_trades += 1
            if realized_pnl < self.stats.largest_loss:
                self.stats.largest_loss = realized_pnl
        
        # Update balance (add back position value, then add/subtract P&L)
        self.current_balance += realized_pnl
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        pnl_sign = "+" if realized_pnl >= 0 else ""
        emoji = "✅" if realized_pnl >= 0 else "❌"
        log.info(f"[PAPER] {emoji} CLOSED {position.side.upper()} {symbol}")
        log.info(f"         Entry: ${position.entry_price:.2f} → Exit: ${exit_price:.2f}")
        log.info(f"         Gross P&L: {pnl_sign}${realized_pnl + position.total_fees:.2f}")
        log.info(f"         Fees Paid: ${position.total_fees:.4f} (entry: ${position.entry_fee:.4f} + exit: ${exit_fee:.4f})")
        log.info(f"         Net P&L: {pnl_sign}${realized_pnl:.2f} ({position.pnl_pct:+.2f}%)")
        log.info(f"         Reason: {reason}")
        log.info(f"         Balance: ${self.current_balance:.2f}")
        
        # Save state immediately
        self._save_state()
        
        return True, realized_pnl
    
    def update_prices(self, prices: Dict[str, float], atrs: Optional[Dict[str, float]] = None) -> None:
        """
        Update current prices for all positions.
        Also checks and activates break-even/trailing stops when triggered.
        
        Args:
            prices: Dict of symbol -> current_price
            atrs: Dict of symbol -> current ATR (optional, for trailing stops)
        """
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.update_price(current_price)
                total_unrealized += position.unrealized_pnl
                
                # Use RiskManager for unified logic
                r_multiple = self.risk_manager.get_r_multiple(
                    entry_price=position.entry_price,
                    current_price=current_price,
                    stop_loss=position.original_stop_loss,
                    side=position.side
                )

                # 1. Check break-even trigger (move SL to entry after 1.0R profit)
                if not position.break_even_activated and r_multiple >= 1.0:
                    old_sl = position.stop_loss
                    new_sl = self.risk_manager.calculate_break_even_stop(
                        entry_price=position.entry_price,
                        side=position.side
                    )
                    
                    if position.side == 'long' and new_sl > position.stop_loss:
                        position.stop_loss = new_sl
                        position.break_even_activated = True
                        log.info(f"[PAPER] 🛡️ BREAK-EVEN ACTIVATED for {symbol}: SL ${old_sl:.2f} → ${new_sl:.2f}")
                    elif position.side == 'short' and new_sl < position.stop_loss:
                        position.stop_loss = new_sl
                        position.break_even_activated = True
                        log.info(f"[PAPER] 🛡️ BREAK-EVEN ACTIVATED for {symbol}: SL ${old_sl:.2f} → ${new_sl:.2f}")

                # 2. Check trailing stop trigger (after 1.5R)
                atr = atrs.get(symbol) if atrs else None
                if r_multiple >= 1.5 and atr is not None:
                    new_trail_sl = self.risk_manager.calculate_trailing_stop(
                        entry_price=position.entry_price,
                        current_price=current_price,
                        atr=atr,
                        side=position.side,
                        multiplier=getattr(settings.enhanced_risk, "atr_stop_multiplier", 2.0)
                    )
                    
                    if position.side == 'long' and new_trail_sl > position.stop_loss:
                        position.stop_loss = new_trail_sl
                    elif position.side == 'short' and new_trail_sl < position.stop_loss:
                        position.stop_loss = new_trail_sl
        
        self.stats.total_unrealized_pnl = total_unrealized
    
    def check_exit_conditions(self, prices: Dict[str, float], 
                              signals: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
        """
        Check if any positions should be closed.
        
        Args:
            prices: Current prices for each symbol
            signals: Optional current signal for each symbol ('buy', 'sell', 'hold')
            
        Returns:
            List of (symbol, reason) for positions to close
        """
        positions_to_close = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            position.update_price(current_price)
            
            # Check stop-loss
            if position.check_stop_loss():
                positions_to_close.append((symbol, "Stop-loss hit"))
                continue
            
            # Check take-profit
            if position.check_take_profit():
                positions_to_close.append((symbol, "Take-profit hit"))
                continue
            
            # Check signal reversal (auto square-off)
            if signals and symbol in signals:
                current_signal = signals[symbol].lower()
                
                # Long position but signal turned bearish
                if position.side == 'long' and current_signal in ['sell', 'strong_sell']:
                    positions_to_close.append((symbol, "Signal reversed to SELL"))
                    continue
                
                # Short position but signal turned bullish
                if position.side == 'short' and current_signal in ['buy', 'strong_buy']:
                    positions_to_close.append((symbol, "Signal reversed to BUY"))
                    continue
        
        return positions_to_close
    
    def process_exits(self, prices: Dict[str, float], 
                      signals: Optional[Dict[str, str]] = None) -> List[float]:
        """
        Process all exit conditions and close positions.
        
        Args:
            prices: Current prices
            signals: Current signals per symbol
            
        Returns:
            List of realized P&L values for closed positions
        """
        positions_to_close = self.check_exit_conditions(prices, signals)
        realized_pnls = []
        
        for symbol, reason in positions_to_close:
            if symbol in prices:
                success, pnl = self.close_position(
                    symbol, 
                    prices[symbol], 
                    reason,
                    PositionStatus.STOPPED_OUT if "Stop" in reason else 
                    PositionStatus.TAKE_PROFIT if "profit" in reason else
                    PositionStatus.SIGNAL_REVERSED
                )
                if success:
                    realized_pnls.append(pnl)
        
        return realized_pnls
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for a symbol if exists."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for symbol."""
        return symbol in self.positions
    
    def get_all_positions(self) -> List[PaperPosition]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def print_positions(self) -> None:
        """Print all open positions with P&L."""
        if not self.positions:
            log.info("[PAPER] No open positions")
            return
        
        log.info("[PAPER] ═══════════════════════════════════════════")
        log.info("[PAPER] OPEN POSITIONS:")
        log.info("[PAPER] ───────────────────────────────────────────")
        
        total_unrealized = 0.0
        total_value = 0.0
        
        for symbol, pos in self.positions.items():
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
            position_value = pos.current_price * pos.size
            total_value += position_value
            total_unrealized += pos.unrealized_pnl
            
            log.info(f"[PAPER]   {emoji} {pos.side.upper():5} {symbol}")
            log.info(f"[PAPER]      Entry: ${pos.entry_price:.2f} → Current: ${pos.current_price:.2f}")
            log.info(f"[PAPER]      Size: {pos.size:.6f} | Value: ${position_value:.2f}")
            log.info(f"[PAPER]      Unrealized P&L: {pnl_sign}${pos.unrealized_pnl:.2f} ({pos.pnl_pct:+.2f}%)")
            log.info(f"[PAPER]      SL: ${pos.stop_loss:.2f} | TP: ${pos.take_profit:.2f}")
        
        log.info("[PAPER] ───────────────────────────────────────────")
        pnl_sign = "+" if total_unrealized >= 0 else ""
        log.info(f"[PAPER]   TOTAL VALUE: ${total_value:.2f}")
        log.info(f"[PAPER]   TOTAL UNREALIZED: {pnl_sign}${total_unrealized:.2f}")
        log.info("[PAPER] ═══════════════════════════════════════════")
    
    def print_summary(self) -> None:
        """Print trading summary and statistics."""
        log.info("[PAPER] ══════════════════════════════════════════════════════")
        log.info("[PAPER] PAPER TRADING SUMMARY")
        log.info("[PAPER] ──────────────────────────────────────────────────────")
        log.info(f"[PAPER]   Initial Balance:    ${self.initial_balance:.2f}")
        log.info(f"[PAPER]   Current Balance:    ${self.current_balance:.2f}")
        
        balance_change = self.current_balance - self.initial_balance
        change_pct = (balance_change / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        change_sign = "+" if balance_change >= 0 else ""
        log.info(f"[PAPER]   Balance Change:     {change_sign}${balance_change:.2f} ({change_pct:+.2f}%)")
        
        log.info("[PAPER] ──────────────────────────────────────────────────────")
        log.info(f"[PAPER]   Total Trades:       {self.stats.total_trades}")
        log.info(f"[PAPER]   Winning Trades:     {self.stats.winning_trades}")
        log.info(f"[PAPER]   Losing Trades:      {self.stats.losing_trades}")
        log.info(f"[PAPER]   Win Rate:           {self.stats.win_rate:.1f}%")
        
        log.info("[PAPER] ──────────────────────────────────────────────────────")
        realized_sign = "+" if self.stats.total_realized_pnl >= 0 else ""
        unrealized_sign = "+" if self.stats.total_unrealized_pnl >= 0 else ""
        total_sign = "+" if self.stats.total_pnl >= 0 else ""
        
        log.info(f"[PAPER]   Realized P&L:       {realized_sign}${self.stats.total_realized_pnl:.2f}")
        log.info(f"[PAPER]   Unrealized P&L:     {unrealized_sign}${self.stats.total_unrealized_pnl:.2f}")
        log.info(f"[PAPER]   TOTAL P&L:          {total_sign}${self.stats.total_pnl:.2f}")
        log.info(f"[PAPER]   Total Fees Paid:    ${self.stats.total_fees_paid:.4f}")
        
        if self.stats.largest_win > 0:
            log.info(f"[PAPER]   Largest Win:        +${self.stats.largest_win:.2f}")
        if self.stats.largest_loss < 0:
            log.info(f"[PAPER]   Largest Loss:       ${self.stats.largest_loss:.2f}")
        
        log.info("[PAPER] ──────────────────────────────────────────────────────")
        log.info(f"[PAPER]   Open Positions:     {len(self.positions)}")
        log.info(f"[PAPER]   Closed Positions:   {len(self.closed_positions)}")
        log.info("[PAPER] ══════════════════════════════════════════════════════")
    
    def get_stats_dict(self) -> Dict:
        """Get statistics as a dictionary."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'balance_change': self.current_balance - self.initial_balance,
            'balance_change_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.stats.total_trades,
            'winning_trades': self.stats.winning_trades,
            'losing_trades': self.stats.losing_trades,
            'win_rate': self.stats.win_rate,
            'realized_pnl': self.stats.total_realized_pnl,
            'unrealized_pnl': self.stats.total_unrealized_pnl,
            'total_pnl': self.stats.total_pnl,
            'largest_win': self.stats.largest_win,
            'largest_loss': self.stats.largest_loss,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions)
        }
    
    def get_positions_as_delta_objects(self) -> List[Any]:
        """Convert paper positions to Delta Exchange Position objects."""
        # Use local import to avoid circular dependency
        from src.delta_client import Position
        
        delta_positions = []
        for p in self.positions.values():
            if p.status != PositionStatus.OPEN:
                continue
            # Calculate sign based on side
            signed_size = p.size if p.side == 'long' else -p.size
            
            delta_positions.append(Position(
                product_id=0,  # Dummy ID
                product_symbol=p.symbol,
                size=abs(p.size), # Delta Position object stores positive size, direction might be inferred or stored elsewhere?
                # Wait, Position object has no 'side' field. 'size' is usually signed in API responses? 
                # Let's check Position definition in delta_client.py
                # It has 'size' (float). API usually returns signed size for positions.
                # Let's verify Position definition again.
                # Step 446: 
                # @dataclass class Position:
                #     product_id: int
                #     product_symbol: str
                #     size: float 
                #     entry_price: float
                #     mark_price: float
                #     unrealized_pnl: float
                #     realized_pnl: float
                
                # If size is signed in Delta API, then I should use signed_size. 
                # Usually negative size = Short.
                # Let's assume signed size.
                entry_price=p.entry_price,
                mark_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl,
                realized_pnl=p.realized_pnl
            ))
            # Wait, Position dataclass logic about size.
            # Delta API documentation says: "size: The size of the position. Positive for buy, negative for sell."
            # So I should use signed_size.
            delta_positions[-1].size = signed_size
            
        return delta_positions
