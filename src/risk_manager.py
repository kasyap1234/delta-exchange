"""
Risk Management Module.
Handles position sizing, stop-loss, take-profit, and exposure limits.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

from config.settings import settings
from utils.logger import log


class ExitReason(str, Enum):
    """Reason for closing a position."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL = "signal_reversal"
    MANUAL = "manual"
    MAX_POSITIONS = "max_positions_reached"


@dataclass
class PositionSizing:
    """Result of position sizing calculation."""
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float
    potential_profit: float
    risk_reward_ratio: float


@dataclass
class TradeRisk:
    """Risk assessment for a potential trade."""
    can_trade: bool
    reason: str
    available_capital: float
    max_position_size: float
    current_exposure: float


class RiskManager:
    """
    Manages trading risk through position sizing, stop-losses, and exposure limits.
    
    Key risk controls:
    - Maximum capital per trade (default: 10%)
    - Stop-loss percentage (default: 2%)
    - Take-profit percentage (default: 4%, 2:1 reward/risk)
    - Maximum open positions (default: 3)
    """
    
    def __init__(self):
        """Initialize risk manager with configuration settings."""
        self.config = settings.trading
        self.open_positions: Dict[str, PositionSizing] = {}
    
    def assess_trade_risk(self, available_balance: float, 
                          current_positions: int) -> TradeRisk:
        """
        Assess whether a new trade can be opened based on risk limits.
        
        Args:
            available_balance: Current available capital
            current_positions: Number of currently open positions
            
        Returns:
            TradeRisk assessment
        """
        # Check max positions limit
        if current_positions >= self.config.max_open_positions:
            return TradeRisk(
                can_trade=False,
                reason=f"Maximum positions reached ({self.config.max_open_positions})",
                available_capital=available_balance,
                max_position_size=0,
                current_exposure=current_positions / self.config.max_open_positions
            )
        
        # Calculate maximum position size
        max_capital = available_balance * self.config.max_capital_per_trade
        
        if max_capital < 10:  # Minimum viable trade size
            return TradeRisk(
                can_trade=False,
                reason=f"Insufficient capital (${max_capital:.2f} < $10 minimum)",
                available_capital=available_balance,
                max_position_size=max_capital,
                current_exposure=current_positions / self.config.max_open_positions
            )
        
        return TradeRisk(
            can_trade=True,
            reason="Trade allowed within risk limits",
            available_capital=available_balance,
            max_position_size=max_capital,
            current_exposure=current_positions / self.config.max_open_positions
        )
    
    def calculate_position_size(self, symbol: str, side: str, 
                                 entry_price: float, 
                                 available_balance: float) -> PositionSizing:
        """
        Calculate optimal position size with stop-loss and take-profit levels.
        
        Uses the 10% capital rule and calculates risk levels:
        - Stop-loss at 2% below entry (for long) or above (for short)
        - Take-profit at 4% above entry (for long) or below (for short)
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            entry_price: Expected entry price
            available_balance: Current available capital
            
        Returns:
            PositionSizing with size, stop-loss, and take-profit
        """
        # Calculate maximum capital to risk
        max_capital = available_balance * self.config.max_capital_per_trade
        
        # Calculate position size in contracts/units
        position_size = max_capital / entry_price
        
        # Calculate stop-loss and take-profit prices
        if side.lower() == 'buy':
            stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.config.take_profit_pct)
        else:  # sell/short
            stop_loss_price = entry_price * (1 + self.config.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.config.take_profit_pct)
        
        # Calculate risk/reward
        risk_amount = abs(entry_price - stop_loss_price) * position_size
        potential_profit = abs(take_profit_price - entry_price) * position_size
        risk_reward_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
        
        sizing = PositionSizing(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            potential_profit=potential_profit,
            risk_reward_ratio=risk_reward_ratio
        )
        
        log.info(f"Position sizing for {symbol}: Size={position_size:.4f}, "
                 f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, "
                 f"R:R={risk_reward_ratio:.2f}")
        
        return sizing
    
    def should_close_position(self, entry_price: float, current_price: float, 
                               side: str) -> tuple[bool, Optional[ExitReason]]:
        """
        Check if position should be closed based on stop-loss or take-profit.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            side: 'buy' or 'sell'
            
        Returns:
            Tuple of (should_close, exit_reason)
        """
        if side.lower() == 'buy':
            # Long position
            stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.config.take_profit_pct)
            
            if current_price <= stop_loss_price:
                log.warning(f"Stop-loss triggered: entry={entry_price}, current={current_price}")
                return True, ExitReason.STOP_LOSS
            elif current_price >= take_profit_price:
                log.info(f"Take-profit triggered: entry={entry_price}, current={current_price}")
                return True, ExitReason.TAKE_PROFIT
        else:
            # Short position
            stop_loss_price = entry_price * (1 + self.config.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.config.take_profit_pct)
            
            if current_price >= stop_loss_price:
                log.warning(f"Stop-loss triggered: entry={entry_price}, current={current_price}")
                return True, ExitReason.STOP_LOSS
            elif current_price <= take_profit_price:
                log.info(f"Take-profit triggered: entry={entry_price}, current={current_price}")
                return True, ExitReason.TAKE_PROFIT
        
        return False, None
    
    def calculate_pnl(self, entry_price: float, exit_price: float, 
                      size: float, side: str) -> float:
        """
        Calculate profit/loss for a closed position.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            side: 'buy' or 'sell'
            
        Returns:
            Profit/loss in quote currency
        """
        if side.lower() == 'buy':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        return pnl
    
    def get_risk_summary(self, balance: float, 
                         positions_count: int) -> Dict:
        """
        Get summary of current risk exposure.
        
        Returns:
            Dictionary with risk metrics
        """
        exposure_pct = (positions_count / self.config.max_open_positions) * 100
        max_single_trade = balance * self.config.max_capital_per_trade
        
        return {
            'total_balance': balance,
            'max_capital_per_trade': max_single_trade,
            'stop_loss_pct': self.config.stop_loss_pct * 100,
            'take_profit_pct': self.config.take_profit_pct * 100,
            'current_positions': positions_count,
            'max_positions': self.config.max_open_positions,
            'exposure_pct': exposure_pct,
            'can_open_new_position': positions_count < self.config.max_open_positions
        }
    
    def validate_position_size(self, size: float, price: float, 
                                min_size: float = 0.001) -> float:
        """
        Validate and adjust position size to meet exchange requirements.
        
        Args:
            size: Calculated position size
            price: Current price
            min_size: Minimum allowed position size
            
        Returns:
            Validated position size
        """
        if size < min_size:
            log.warning(f"Position size {size} below minimum {min_size}, adjusting")
            return min_size
        
        # Round to reasonable precision
        if price > 10000:  # BTC-like
            return round(size, 5)
        elif price > 100:  # ETH-like
            return round(size, 4)
        else:
            return round(size, 3)
