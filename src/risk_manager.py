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

    def assess_trade_risk(
        self, available_balance: float, current_positions: int
    ) -> TradeRisk:
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
                current_exposure=current_positions / self.config.max_open_positions,
            )

        # Calculate maximum position size
        max_capital = available_balance * self.config.max_capital_per_trade

        if max_capital < 10:  # Minimum viable trade size
            return TradeRisk(
                can_trade=False,
                reason=f"Insufficient capital (${max_capital:.2f} < $10 minimum)",
                available_capital=available_balance,
                max_position_size=max_capital,
                current_exposure=current_positions / self.config.max_open_positions,
            )

        return TradeRisk(
            can_trade=True,
            reason="Trade allowed within risk limits",
            available_capital=available_balance,
            max_position_size=max_capital,
            current_exposure=current_positions / self.config.max_open_positions,
        )

    def calculate_risk_based_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Position Size = (Account × Risk%) / |Entry - StopLoss|
        Industry standard: 1-2% risk per trade.
        """
        risk_pct = risk_pct or settings.enhanced_risk.max_risk_per_trade
        risk_amount = account_balance * risk_pct
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk <= 0:
            return 0

        return risk_amount / price_risk

    def get_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        Formula: f* = (p * b - q) / b
        Where: p = win rate, q = 1-p, b = win/loss ratio
        """
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio

        # Use fractional Kelly (e.g. 0.5) for safety, and cap at max risk
        safe_kelly = kelly * settings.enhanced_risk.kelly_fraction
        return max(0, min(safe_kelly, settings.enhanced_risk.max_risk_per_trade))

    def adjust_for_volatility(
        self, base_size: float, current_atr: float, avg_atr: float
    ) -> float:
        """
        Reduce position size if current volatility is significantly higher than average.
        """
        if not settings.enhanced_risk.reduce_size_high_volatility or avg_atr <= 0:
            return base_size

        # If ATR > 1.5x average, reduce size proportionally
        ratio = current_atr / avg_atr
        if ratio > settings.enhanced_risk.atr_size_multiplier:
            reduction_factor = settings.enhanced_risk.atr_size_multiplier / ratio
            return base_size * reduction_factor

        return base_size

    def calculate_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        available_balance: float,
        atr: Optional[float] = None,
        avg_atr: Optional[float] = None,
        performance_data: Optional[dict] = None,
    ) -> PositionSizing:
        """
        Calculate optimal position size using risk-based rules.

        Implements:
        1. 2% Risk Rule: Position Size = (Account * Risk%) / |Entry - StopLoss|
        2. Kelly Criterion: Fraction of capital to allocate based on actual win rate
        3. ATR-based Stops: Uses ATR for volatility-adjusted stop distance

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            entry_price: Expected entry price
            available_balance: Current available capital
            atr: Current ATR for dynamic stops
            avg_atr: Average ATR for volatility adjustment
            performance_data: Dict with 'total_trades', 'winning_trades', 'total_pnl' for Kelly

        Returns:
            PositionSizing with size, stop-loss, and take-profit
        """
        # 1. Determine Stop-Loss Distance based on ATR or % fallback
        if atr:
            sl_multiplier = settings.enhanced_risk.atr_stop_multiplier
            risk_dist = atr * sl_multiplier
        else:
            risk_dist = entry_price * self.config.stop_loss_pct

        # 2. Calculate explicit SL/TP prices
        if side.lower() == "buy" or side.lower() == "long":
            stop_loss_price = entry_price - risk_dist
            take_profit_price = entry_price + (risk_dist * 2)  # 2:1 R:R default
        else:
            stop_loss_price = entry_price + risk_dist
            take_profit_price = entry_price - (risk_dist * 2)

        # 3. Calculate Base Position Size based on RISK
        nominal_size = self.calculate_risk_based_size(
            entry_price, stop_loss_price, available_balance
        )

        # 4. Apply Kelly Criterion Limit (if enabled) with ACTUAL performance data
        if settings.enhanced_risk.use_kelly_sizing:
            # Use actual performance data if available, otherwise conservative defaults
            if performance_data and performance_data.get("total_trades", 0) >= 10:
                win_rate = performance_data.get(
                    "winning_trades", 0
                ) / performance_data.get("total_trades", 1)
                total_pnl = performance_data.get("total_pnl", 0)
                total_trades = performance_data.get("total_trades", 1)

                # Calculate win/loss ratio from actual performance
                if win_rate > 0:
                    avg_win = (
                        total_pnl / (total_trades * win_rate) if win_rate > 0 else 0
                    )
                    avg_loss = (
                        abs(total_pnl / (total_trades * (1 - win_rate)))
                        if win_rate < 1
                        else 1
                    )
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                else:
                    win_loss_ratio = 1.0

                log.info(
                    f"Kelly using actual data: WinRate={win_rate:.2%}, WinLoss={win_loss_ratio:.2f}"
                )
            else:
                # Conservative defaults for low sample size
                win_rate = 0.45
                win_loss_ratio = 2.0
                log.info("Kelly using defaults (insufficient trade data)")

            kelly_fraction = self.get_kelly_fraction(win_rate, win_loss_ratio)

            # Kelly-capped size (Kelly fraction refers to fraction of account)
            kelly_max_allocation = available_balance * kelly_fraction
            kelly_size_units = kelly_max_allocation / entry_price

            nominal_size = min(nominal_size, kelly_size_units)

        # 5. Apply Volatility Adjustment
        if atr and avg_atr:
            nominal_size = self.adjust_for_volatility(nominal_size, atr, avg_atr)

        # 6. Global Caps Check
        # Never exceed max_position_pct of total account
        total_account_cap = available_balance * settings.enhanced_risk.max_position_pct
        cap_units = total_account_cap / entry_price

        # Also respect the older max_capital_per_trade for backward compatibility or extra safety
        legacy_cap = available_balance * self.config.max_capital_per_trade
        legacy_cap_units = legacy_cap / entry_price

        final_size = min(nominal_size, cap_units, legacy_cap_units)

        # Calculate final metrics
        risk_amount = abs(entry_price - stop_loss_price) * final_size
        potential_profit = abs(take_profit_price - entry_price) * final_size
        risk_reward_ratio = potential_profit / risk_amount if risk_amount > 0 else 0

        sizing = PositionSizing(
            symbol=symbol,
            side=side,
            size=final_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            potential_profit=potential_profit,
            risk_reward_ratio=risk_reward_ratio,
        )

        log.info(
            f"Position sizing for {symbol}: Size={final_size:.4f}, "
            f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, "
            f"R:R={risk_reward_ratio:.2f}"
        )

        return sizing

    def should_close_position(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> tuple[bool, Optional[ExitReason]]:
        """
        Check if position should be closed based on stop-loss or take-profit.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            side: 'buy' or 'sell'
            stop_loss: Explicit stop loss price if available
            take_profit: Explicit take profit price if available

        Returns:
            Tuple of (should_close, exit_reason)
        """
        if side.lower() == "buy":
            # Long position
            sl_price = stop_loss or (entry_price * (1 - self.config.stop_loss_pct))
            tp_price = take_profit or (entry_price * (1 + self.config.take_profit_pct))

            if current_price <= sl_price:
                log.warning(
                    f"Stop-loss triggered: entry={entry_price}, current={current_price}, sl={sl_price}"
                )
                return True, ExitReason.STOP_LOSS
            elif current_price >= tp_price:
                log.info(
                    f"Take-profit triggered: entry={entry_price}, current={current_price}, tp={tp_price}"
                )
                return True, ExitReason.TAKE_PROFIT
        else:
            # Short position
            sl_price = stop_loss or (entry_price * (1 + self.config.stop_loss_pct))
            tp_price = take_profit or (entry_price * (1 - self.config.take_profit_pct))

            if current_price >= sl_price:
                log.warning(
                    f"Stop-loss triggered: entry={entry_price}, current={current_price}, sl={sl_price}"
                )
                return True, ExitReason.STOP_LOSS
            elif current_price <= tp_price:
                log.info(
                    f"Take-profit triggered: entry={entry_price}, current={current_price}, tp={tp_price}"
                )
                return True, ExitReason.TAKE_PROFIT

        return False, None

    def calculate_pnl(
        self, entry_price: float, exit_price: float, size: float, side: str
    ) -> float:
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
        if side.lower() == "buy":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        return pnl

    def get_risk_summary(self, balance: float, positions_count: int) -> Dict:
        """
        Get summary of current risk exposure.

        Returns:
            Dictionary with risk metrics
        """
        exposure_pct = (positions_count / self.config.max_open_positions) * 100
        max_single_trade = balance * self.config.max_capital_per_trade

        return {
            "total_balance": balance,
            "max_capital_per_trade": max_single_trade,
            "stop_loss_pct": self.config.stop_loss_pct * 100,
            "take_profit_pct": self.config.take_profit_pct * 100,
            "current_positions": positions_count,
            "max_positions": self.config.max_open_positions,
            "exposure_pct": exposure_pct,
            "can_open_new_position": positions_count < self.config.max_open_positions,
        }

    def validate_position_size(
        self, size: float, price: float, min_size: float = 0.001
    ) -> float:
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
