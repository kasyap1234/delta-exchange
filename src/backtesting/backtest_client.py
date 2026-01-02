"""
Backtest Delta Client Module.
Mock DeltaExchangeClient for realistic strategy backtesting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import uuid

from src.backtesting.backtest_data_provider import BacktestDataProvider
from src.delta_client import Candle, Position, Order
from utils.logger import log


class SimulatedOrderStatus(str, Enum):
    """Order status in simulation."""
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass
class SimulatedPosition:
    """Simulated position for backtesting."""
    product_symbol: str
    size: float  # Positive = long, Negative = short
    entry_price: float
    margin: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def to_position(self) -> Position:
        """Convert to Position object for strategy compatibility."""
        return Position(
            product_symbol=self.product_symbol,
            size=self.size,
            entry_price=self.entry_price,
            margin=self.margin,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl
        )
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        if self.size > 0:  # Long
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # Short
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.size)


@dataclass
class SimulatedOrder:
    """Simulated order for tracking."""
    id: str
    product_symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_price: float = 0.0
    filled_at: str = ""


@dataclass 
class TradeRecord:
    """Record of a completed trade for analysis."""
    id: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy: str
    commission: float = 0.0


class BacktestDeltaClient:
    """
    Mock Delta Exchange client for backtesting.
    
    Simulates all API calls using historical data:
    - get_candles: Returns historical candles up to current bar
    - get_ticker: Returns current bar's price
    - place_order: Simulates order execution with slippage
    - get_positions: Returns simulated positions
    """
    
    def __init__(
        self, 
        data_provider: BacktestDataProvider,
        initial_capital: float = 10000.0,
        leverage: int = 5,
        commission_pct: float = 0.0006,  # 0.06% taker fee
        slippage_pct: float = 0.0001     # 0.01% slippage
    ):
        """
        Initialize mock client.
        
        Args:
            data_provider: BacktestDataProvider with historical data
            initial_capital: Starting capital in USD
            leverage: Leverage multiplier
            commission_pct: Commission per trade
            slippage_pct: Slippage per trade
        """
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        # Track positions and orders
        self.positions: Dict[str, SimulatedPosition] = {}
        self.orders: Dict[str, SimulatedOrder] = {}
        self.closed_trades: List[TradeRecord] = []
        
        # Track pending bracket orders (stop-loss/take-profit)
        self.bracket_orders: Dict[str, Dict[str, float]] = {}  # symbol -> {stop_loss, take_profit}
        
        # Equity tracking
        self.equity_curve: List[float] = [initial_capital]
        
        log.info(f"BacktestDeltaClient initialized: ${initial_capital}, {leverage}x leverage")
    
    # =========================================================================
    # API COMPATIBILITY METHODS (match DeltaExchangeClient interface)
    # =========================================================================
    
    def get_candles(
        self, 
        symbol: str, 
        resolution: str = '15m',
        limit: int = 100
    ) -> List[Candle]:
        """
        Get historical candles up to current bar.
        
        Matches DeltaExchangeClient.get_candles() interface.
        """
        if resolution in ['4h', '1h', '1d']:
            return self.data_provider.get_higher_tf_candles(symbol, resolution)
        return self.data_provider.get_candles_up_to(symbol, limit)
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker (uses current bar's close as mark price).
        
        Matches DeltaExchangeClient.get_ticker() interface.
        """
        price = self.data_provider.get_current_price(symbol)
        ohlcv = self.data_provider.get_current_ohlcv(symbol)
        
        return {
            'symbol': symbol,
            'mark_price': price,
            'spot_price': price,
            'last_price': price,
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'volume': ohlcv['volume']
        }
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        # Update unrealized P&L first
        for symbol, pos in self.positions.items():
            current_price = self.data_provider.get_current_price(symbol)
            pos.update_unrealized_pnl(current_price)
        
        return [pos.to_position() for pos in self.positions.values() if pos.size != 0]
    
    def place_order(
        self,
        product_symbol: str,
        side: str,
        size: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate order placement.
        
        Returns order response matching DeltaExchangeClient format.
        """
        order_id = str(uuid.uuid4())[:8]
        current_price = self.data_provider.get_current_price(product_symbol)
        current_time = self.data_provider.get_current_datetime(
            self.data_provider.get_symbols()[0]
        )
        
        if current_price <= 0:
            log.error(f"Invalid price for {product_symbol}")
            return {'success': False, 'error': 'Invalid price'}
        
        # Apply slippage
        if side == 'buy':
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)
        
        # Calculate position value and commission
        position_value = size * fill_price
        commission = position_value * self.commission_pct
        
        # Check if we have enough capital
        margin_required = position_value / self.leverage
        if margin_required + commission > self.capital:
            log.warning(f"Insufficient capital for order: need ${margin_required:.2f}, have ${self.capital:.2f}")
            return {'success': False, 'error': 'Insufficient capital'}
        
        # Execute the order
        self._execute_order(product_symbol, side, size, fill_price, commission, current_time)
        
        # Set bracket orders if provided
        if stop_loss or take_profit:
            self.bracket_orders[product_symbol] = {
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        
        order = SimulatedOrder(
            id=order_id,
            product_symbol=product_symbol,
            side=side,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_loss,
            status='filled',
            filled_price=fill_price,
            filled_at=current_time
        )
        self.orders[order_id] = order
        
        log.debug(f"Order filled: {side} {size} {product_symbol} @ {fill_price:.2f}")
        
        return {
            'success': True,
            'order_id': order_id,
            'filled_price': fill_price,
            'size': size,
            'side': side
        }
    
    def close_position(self, product_symbol: str) -> Dict[str, Any]:
        """Close an entire position."""
        if product_symbol not in self.positions:
            return {'success': False, 'error': 'No position'}
        
        pos = self.positions[product_symbol]
        side = 'sell' if pos.size > 0 else 'buy'
        size = abs(pos.size)
        
        return self.place_order(product_symbol, side, size, 'market')
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _execute_order(
        self, 
        symbol: str, 
        side: str, 
        size: float, 
        fill_price: float,
        commission: float,
        fill_time: str
    ) -> None:
        """Execute order and update position."""
        # Deduct commission
        self.capital -= commission
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # Check if this is closing/reducing or adding
            is_long = pos.size > 0
            is_closing = (is_long and side == 'sell') or (not is_long and side == 'buy')
            
            if is_closing:
                # Calculate P&L
                if is_long:
                    pnl = (fill_price - pos.entry_price) * min(size, abs(pos.size))
                else:
                    pnl = (pos.entry_price - fill_price) * min(size, abs(pos.size))
                
                self.capital += pnl
                
                # Record the trade
                trade = TradeRecord(
                    id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    direction='long' if is_long else 'short',
                    entry_time=str(pos.entry_price),  # We should track entry time
                    entry_price=pos.entry_price,
                    exit_time=fill_time,
                    exit_price=fill_price,
                    size=min(size, abs(pos.size)),
                    pnl=pnl - commission,
                    pnl_pct=((fill_price - pos.entry_price) / pos.entry_price * 100) if is_long 
                            else ((pos.entry_price - fill_price) / pos.entry_price * 100),
                    exit_reason='manual_close',
                    strategy='backtest',
                    commission=commission
                )
                self.closed_trades.append(trade)
                
                # Update or remove position
                if size >= abs(pos.size):
                    del self.positions[symbol]
                    if symbol in self.bracket_orders:
                        del self.bracket_orders[symbol]
                else:
                    pos.size = pos.size - size if is_long else pos.size + size
            else:
                # Adding to position (average up/down)
                total_size = abs(pos.size) + size
                new_entry = (pos.entry_price * abs(pos.size) + fill_price * size) / total_size
                pos.entry_price = new_entry
                pos.size = pos.size + size if side == 'buy' else pos.size - size
        else:
            # New position
            margin = (size * fill_price) / self.leverage
            self.positions[symbol] = SimulatedPosition(
                product_symbol=symbol,
                size=size if side == 'buy' else -size,
                entry_price=fill_price,
                margin=margin
            )
    
    def check_bracket_orders(self) -> List[TradeRecord]:
        """
        Check and execute stop-loss/take-profit orders.
        
        Called each bar to check if SL/TP levels are hit.
        """
        closed = []
        
        for symbol in list(self.positions.keys()):
            if symbol not in self.bracket_orders:
                continue
            
            pos = self.positions[symbol]
            brackets = self.bracket_orders[symbol]
            current_bar = self.data_provider.get_current_ohlcv(symbol)
            current_time = self.data_provider.get_current_datetime(
                self.data_provider.get_symbols()[0]
            )
            
            is_long = pos.size > 0
            stop_loss = brackets.get('stop_loss')
            take_profit = brackets.get('take_profit')
            
            exit_price = None
            exit_reason = None
            
            if is_long:
                # Check stop-loss (use bar low)
                if stop_loss and current_bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                # Check take-profit (use bar high)
                elif take_profit and current_bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
            else:
                # Short position
                if stop_loss and current_bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif take_profit and current_bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
            
            if exit_price and exit_reason:
                # Execute exit
                side = 'sell' if is_long else 'buy'
                size = abs(pos.size)
                
                # Calculate P&L
                if is_long:
                    pnl = (exit_price - pos.entry_price) * size
                else:
                    pnl = (pos.entry_price - exit_price) * size
                
                commission = (size * exit_price) * self.commission_pct
                self.capital += pnl - commission
                
                trade = TradeRecord(
                    id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    direction='long' if is_long else 'short',
                    entry_time="",
                    entry_price=pos.entry_price,
                    exit_time=current_time,
                    exit_price=exit_price,
                    size=size,
                    pnl=pnl - commission,
                    pnl_pct=((exit_price - pos.entry_price) / pos.entry_price * 100) if is_long
                            else ((pos.entry_price - exit_price) / pos.entry_price * 100),
                    exit_reason=exit_reason,
                    strategy='backtest',
                    commission=commission
                )
                self.closed_trades.append(trade)
                closed.append(trade)
                
                del self.positions[symbol]
                del self.bracket_orders[symbol]
                
                log.debug(f"Bracket order triggered: {exit_reason} on {symbol}, P&L: ${pnl:.2f}")
        
        return closed
    
    def update_equity(self) -> float:
        """Update and return current equity."""
        equity = self.capital
        
        for symbol, pos in self.positions.items():
            current_price = self.data_provider.get_current_price(symbol)
            pos.update_unrealized_pnl(current_price)
            equity += pos.unrealized_pnl
        
        self.equity_curve.append(equity)
        return equity
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        equity = self.update_equity()
        return {
            'available_balance': self.capital,
            'equity': equity,
            'unrealized_pnl': equity - self.capital
        }
    
    def get_product(self, symbol: str) -> Dict[str, Any]:
        """Get product info (mock)."""
        return {
            'symbol': symbol,
            'contract_type': 'perpetual',
            'tick_size': 0.01,
            'contract_size': 1
        }
