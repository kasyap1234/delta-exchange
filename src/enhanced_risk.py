"""
Enhanced Risk Management Module.

Advanced risk management features beyond basic stop-loss/take-profit:
- Dynamic stop-loss adjustment based on ATR and volatility
- Break-even stops after reaching profit threshold
- Trailing stops with multiple strategies
- Time-based position management
- Profit protection (lock in gains)
- Position scaling (scale in/out)
- Drawdown-based risk reduction
- Volatility-adjusted position sizing
- Maximum adverse excursion (MAE) tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from utils.logger import log


class StopType(str, Enum):
    """Type of stop-loss."""

    FIXED = "fixed"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    BREAK_EVEN = "break_even"
    TIME_BASED = "time_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class TrailingMethod(str, Enum):
    """Method for calculating trailing stop."""

    FIXED_DISTANCE = "fixed_distance"
    ATR_MULTIPLE = "atr_multiple"
    PERCENTAGE = "percentage"
    SWING_LOW_HIGH = "swing_low_high"
    CHANDELIER = "chandelier"
    PARABOLIC = "parabolic"


class PositionStatus(str, Enum):
    """Current status of a position."""

    OPEN = "open"
    IN_PROFIT = "in_profit"
    AT_BREAK_EVEN = "at_break_even"
    IN_DRAWDOWN = "in_drawdown"
    NEAR_STOP = "near_stop"
    NEAR_TARGET = "near_target"
    STAGNANT = "stagnant"


@dataclass
class RiskMetrics:
    """Risk metrics for a position."""

    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    side: str

    # Calculated metrics
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    risk_amount: float = 0.0
    reward_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    current_r_multiple: float = 0.0  # How many R we're at
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    def update(self, current_price: float) -> None:
        """Update metrics with new price."""
        self.current_price = current_price

        if self.side == "long":
            self.unrealized_pnl = (
                current_price - self.entry_price
            ) * self.position_size
            self.unrealized_pnl_pct = (
                (current_price - self.entry_price) / self.entry_price
            ) * 100
            self.risk_amount = (self.entry_price - self.stop_loss) * self.position_size
            self.reward_amount = (
                self.take_profit - self.entry_price
            ) * self.position_size
        else:
            self.unrealized_pnl = (
                self.entry_price - current_price
            ) * self.position_size
            self.unrealized_pnl_pct = (
                (self.entry_price - current_price) / self.entry_price
            ) * 100
            self.risk_amount = (self.stop_loss - self.entry_price) * self.position_size
            self.reward_amount = (
                self.entry_price - self.take_profit
            ) * self.position_size

        if self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount
            self.current_r_multiple = self.unrealized_pnl / self.risk_amount

        # Track MAE and MFE
        if self.unrealized_pnl < 0:
            self.mae = min(self.mae, self.unrealized_pnl)
        else:
            self.mfe = max(self.mfe, self.unrealized_pnl)


@dataclass
class DynamicStop:
    """Dynamic stop-loss configuration."""

    initial_stop: float
    current_stop: float
    stop_type: StopType
    trailing_method: Optional[TrailingMethod] = None

    # Configuration
    atr_multiplier: float = 2.0
    trailing_distance: float = 0.0
    break_even_trigger_r: float = 1.0  # Move to BE after 1R profit
    break_even_buffer: float = 0.001  # 0.1% above entry for BE

    # State
    is_at_break_even: bool = False
    highest_price: float = 0.0  # For trailing (long)
    lowest_price: float = float("inf")  # For trailing (short)
    last_updated: datetime = field(default_factory=datetime.now)

    # Profit protection
    profit_lock_levels: List[Tuple[float, float]] = field(default_factory=list)
    # Format: [(r_multiple, lock_pct), ...] e.g., [(2.0, 0.5)] = at 2R, lock 50% of profit


@dataclass
class PositionRiskState:
    """Complete risk state for a position."""

    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    position_size: float

    # Stop/Target
    dynamic_stop: DynamicStop
    take_profit: float

    # Current state
    metrics: RiskMetrics = None
    status: PositionStatus = PositionStatus.OPEN

    # Scaling
    original_size: float = 0.0
    scaled_out_size: float = 0.0
    scale_out_prices: List[float] = field(default_factory=list)

    # Time management
    max_hold_time_hours: Optional[float] = None
    stagnant_threshold_hours: float = 12.0  # Close if no progress in 12h

    def __post_init__(self):
        if self.original_size == 0:
            self.original_size = self.position_size
        if self.metrics is None:
            self.metrics = RiskMetrics(
                entry_price=self.entry_price,
                current_price=self.entry_price,
                stop_loss=self.dynamic_stop.current_stop,
                take_profit=self.take_profit,
                position_size=self.position_size,
                side=self.side,
            )


class EnhancedRiskManager:
    """
    Enhanced risk management with dynamic stops and profit protection.

    Features:
    1. Dynamic Stop-Loss
       - ATR-based initial stops
       - Trailing stops with multiple methods
       - Break-even stops after profit threshold

    2. Profit Protection
       - Lock in profits at R-multiple levels
       - Scale out at profit targets
       - Never give back more than X% of max profit

    3. Time-Based Management
       - Maximum hold time
       - Stagnant position detection
       - Time-decay risk adjustment

    4. Drawdown Management
       - Reduce position size during drawdown
       - Tighter stops during losing streaks
       - Circuit breaker for extreme losses
    """

    # Default configuration
    DEFAULT_ATR_STOP_MULT = 2.0
    DEFAULT_ATR_TRAIL_MULT = 1.5
    BREAK_EVEN_TRIGGER_R = 1.0
    BREAK_EVEN_BUFFER_PCT = 0.001  # 0.1%

    # Profit protection defaults
    PROFIT_LOCK_LEVELS = [
        (1.5, 0.25),  # At 1.5R, lock 25% of profit
        (2.0, 0.50),  # At 2R, lock 50% of profit
        (3.0, 0.75),  # At 3R, lock 75% of profit
    ]

    # Drawdown thresholds
    LIGHT_DRAWDOWN = 0.03  # 3% - slight reduction
    MODERATE_DRAWDOWN = 0.05  # 5% - significant reduction
    SEVERE_DRAWDOWN = 0.10  # 10% - halt new trades

    # Time limits
    DEFAULT_MAX_HOLD_HOURS = 168  # 1 week
    STAGNANT_THRESHOLD_HOURS = 12

    def __init__(self):
        """Initialize enhanced risk manager."""
        self._positions: Dict[str, PositionRiskState] = {}
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._drawdown: float = 0.0
        self._consecutive_losses: int = 0
        self._trade_history: List[Dict] = []

        log.info("EnhancedRiskManager initialized")

    def create_position_risk_state(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        position_size: float,
        atr: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> PositionRiskState:
        """
        Create a new position risk state with dynamic stops.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            position_size: Position size
            atr: Current ATR (for dynamic stops)
            stop_loss: Override stop-loss price
            take_profit: Override take-profit price

        Returns:
            PositionRiskState with configured stops
        """
        # Calculate stops
        if atr:
            sl_distance = atr * self.DEFAULT_ATR_STOP_MULT
            tp_distance = atr * self.DEFAULT_ATR_STOP_MULT * 2  # 2:1 R:R
            trailing_distance = atr * self.DEFAULT_ATR_TRAIL_MULT
        else:
            # Fallback to percentage-based
            sl_pct = getattr(settings.trading, "stop_loss_pct", 0.04)
            tp_pct = getattr(settings.trading, "take_profit_pct", 0.09)
            sl_distance = entry_price * sl_pct
            tp_distance = entry_price * tp_pct
            trailing_distance = entry_price * (sl_pct * 0.75)

        if side == "long":
            initial_sl = stop_loss if stop_loss else entry_price - sl_distance
            initial_tp = take_profit if take_profit else entry_price + tp_distance
        else:
            initial_sl = stop_loss if stop_loss else entry_price + sl_distance
            initial_tp = take_profit if take_profit else entry_price - tp_distance

        # Create dynamic stop
        dynamic_stop = DynamicStop(
            initial_stop=initial_sl,
            current_stop=initial_sl,
            stop_type=StopType.ATR_BASED if atr else StopType.FIXED,
            trailing_method=TrailingMethod.ATR_MULTIPLE
            if atr
            else TrailingMethod.PERCENTAGE,
            atr_multiplier=self.DEFAULT_ATR_STOP_MULT,
            trailing_distance=trailing_distance,
            break_even_trigger_r=self.BREAK_EVEN_TRIGGER_R,
            break_even_buffer=self.BREAK_EVEN_BUFFER_PCT,
            highest_price=entry_price,
            lowest_price=entry_price,
            profit_lock_levels=list(self.PROFIT_LOCK_LEVELS),
        )

        # Create position state
        state = PositionRiskState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            position_size=position_size,
            dynamic_stop=dynamic_stop,
            take_profit=initial_tp,
            max_hold_time_hours=self.DEFAULT_MAX_HOLD_HOURS,
            stagnant_threshold_hours=self.STAGNANT_THRESHOLD_HOURS,
        )

        self._positions[symbol] = state

        log.info(
            f"[RISK] Created position state for {symbol}: "
            f"Entry={entry_price:.2f}, SL={initial_sl:.2f}, TP={initial_tp:.2f}"
        )

        return state

    def update_position(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None,
    ) -> Tuple[PositionRiskState, Optional[str]]:
        """
        Update position risk state with new price.

        Returns:
            Tuple of (updated_state, action) where action is None, 'close', or 'scale_out'
        """
        if symbol not in self._positions:
            log.warning(f"[RISK] No position state for {symbol}")
            return None, None

        state = self._positions[symbol]

        # Update metrics
        state.metrics.update(current_price)
        state.metrics.stop_loss = state.dynamic_stop.current_stop

        # Update status
        state.status = self._determine_status(state, current_price)

        # Check for stop/target hit
        action = self._check_exit_conditions(state, current_price)
        if action:
            return state, action

        # Update dynamic stop
        self._update_dynamic_stop(state, current_price, atr)

        # Check for time-based exit
        action = self._check_time_conditions(state)
        if action:
            return state, action

        return state, None

    def _determine_status(
        self, state: PositionRiskState, current_price: float
    ) -> PositionStatus:
        """Determine current position status."""
        entry = state.entry_price
        stop = state.dynamic_stop.current_stop
        target = state.take_profit
        side = state.side

        if side == "long":
            pnl_pct = (current_price - entry) / entry
            distance_to_stop = (current_price - stop) / current_price
            distance_to_target = (target - current_price) / current_price
        else:
            pnl_pct = (entry - current_price) / entry
            distance_to_stop = (stop - current_price) / current_price
            distance_to_target = (current_price - target) / current_price

        # Check various conditions
        if state.dynamic_stop.is_at_break_even:
            if abs(pnl_pct) < 0.001:
                return PositionStatus.AT_BREAK_EVEN

        if pnl_pct > 0.005:  # 0.5% in profit
            return PositionStatus.IN_PROFIT
        elif pnl_pct < -0.005:
            return PositionStatus.IN_DRAWDOWN

        if distance_to_stop < 0.01:  # Within 1% of stop
            return PositionStatus.NEAR_STOP

        if distance_to_target < 0.01:  # Within 1% of target
            return PositionStatus.NEAR_TARGET

        # Check for stagnant position
        hours_held = (datetime.now() - state.entry_time).total_seconds() / 3600
        if hours_held > state.stagnant_threshold_hours:
            if abs(pnl_pct) < 0.01:  # Less than 1% move
                return PositionStatus.STAGNANT

        return PositionStatus.OPEN

    def _check_exit_conditions(
        self, state: PositionRiskState, current_price: float
    ) -> Optional[str]:
        """Check if position should be closed."""
        stop = state.dynamic_stop.current_stop
        target = state.take_profit

        if state.side == "long":
            if current_price <= stop:
                log.warning(
                    f"[RISK] {state.symbol}: Stop-loss hit at {current_price:.2f}"
                )
                return "stop_loss"
            if current_price >= target:
                log.info(
                    f"[RISK] {state.symbol}: Take-profit hit at {current_price:.2f}"
                )
                return "take_profit"
        else:
            if current_price >= stop:
                log.warning(
                    f"[RISK] {state.symbol}: Stop-loss hit at {current_price:.2f}"
                )
                return "stop_loss"
            if current_price <= target:
                log.info(
                    f"[RISK] {state.symbol}: Take-profit hit at {current_price:.2f}"
                )
                return "take_profit"

        return None

    def _update_dynamic_stop(
        self,
        state: PositionRiskState,
        current_price: float,
        atr: Optional[float] = None,
    ) -> None:
        """Update dynamic stop-loss based on price movement."""
        ds = state.dynamic_stop
        entry = state.entry_price
        side = state.side

        # Update highest/lowest price tracking
        if side == "long":
            ds.highest_price = max(ds.highest_price, current_price)
        else:
            ds.lowest_price = min(ds.lowest_price, current_price)

        # Calculate current R-multiple
        if side == "long":
            r_multiple = (current_price - entry) / (entry - ds.initial_stop)
        else:
            r_multiple = (entry - current_price) / (ds.initial_stop - entry)

        # 1. Check for break-even trigger
        if not ds.is_at_break_even and r_multiple >= ds.break_even_trigger_r:
            self._move_to_break_even(state)

        # 2. Update trailing stop if in profit
        if r_multiple > 0:
            self._update_trailing_stop(state, current_price, atr)

        # 3. Apply profit protection locks
        self._apply_profit_protection(state, current_price, r_multiple)

        ds.last_updated = datetime.now()

    def _move_to_break_even(self, state: PositionRiskState) -> None:
        """Move stop-loss to break-even with buffer."""
        ds = state.dynamic_stop
        entry = state.entry_price
        buffer = entry * ds.break_even_buffer

        if state.side == "long":
            new_stop = entry + buffer
            if new_stop > ds.current_stop:
                ds.current_stop = new_stop
                ds.is_at_break_even = True
                ds.stop_type = StopType.BREAK_EVEN
                log.info(
                    f"[RISK] {state.symbol}: Moved to break-even at {new_stop:.2f}"
                )
        else:
            new_stop = entry - buffer
            if new_stop < ds.current_stop:
                ds.current_stop = new_stop
                ds.is_at_break_even = True
                ds.stop_type = StopType.BREAK_EVEN
                log.info(
                    f"[RISK] {state.symbol}: Moved to break-even at {new_stop:.2f}"
                )

    def _update_trailing_stop(
        self,
        state: PositionRiskState,
        current_price: float,
        atr: Optional[float] = None,
    ) -> None:
        """Update trailing stop based on configured method."""
        ds = state.dynamic_stop

        if ds.trailing_method == TrailingMethod.ATR_MULTIPLE and atr:
            trail_distance = atr * ds.atr_multiplier
        elif ds.trailing_method == TrailingMethod.FIXED_DISTANCE:
            trail_distance = ds.trailing_distance
        elif ds.trailing_method == TrailingMethod.PERCENTAGE:
            trail_distance = current_price * 0.02  # 2% trailing
        else:
            trail_distance = ds.trailing_distance

        if state.side == "long":
            new_stop = ds.highest_price - trail_distance
            if new_stop > ds.current_stop:
                old_stop = ds.current_stop
                ds.current_stop = new_stop
                ds.stop_type = StopType.TRAILING
                log.info(
                    f"[RISK] {state.symbol}: Trailing stop updated "
                    f"{old_stop:.2f} -> {new_stop:.2f}"
                )
        else:
            new_stop = ds.lowest_price + trail_distance
            if new_stop < ds.current_stop:
                old_stop = ds.current_stop
                ds.current_stop = new_stop
                ds.stop_type = StopType.TRAILING
                log.info(
                    f"[RISK] {state.symbol}: Trailing stop updated "
                    f"{old_stop:.2f} -> {new_stop:.2f}"
                )

    def _apply_profit_protection(
        self,
        state: PositionRiskState,
        current_price: float,
        r_multiple: float,
    ) -> None:
        """Apply profit protection by locking in gains."""
        ds = state.dynamic_stop
        entry = state.entry_price

        if not ds.profit_lock_levels:
            return

        for lock_r, lock_pct in ds.profit_lock_levels:
            if r_multiple >= lock_r:
                # Calculate profit to lock
                if state.side == "long":
                    total_profit = current_price - entry
                    locked_profit = total_profit * lock_pct
                    min_exit = entry + locked_profit

                    if min_exit > ds.current_stop:
                        ds.current_stop = min_exit
                        log.info(
                            f"[RISK] {state.symbol}: Profit lock at {lock_r}R - "
                            f"Locking {lock_pct:.0%} profit, new stop {min_exit:.2f}"
                        )
                else:
                    total_profit = entry - current_price
                    locked_profit = total_profit * lock_pct
                    min_exit = entry - locked_profit

                    if min_exit < ds.current_stop:
                        ds.current_stop = min_exit
                        log.info(
                            f"[RISK] {state.symbol}: Profit lock at {lock_r}R - "
                            f"Locking {lock_pct:.0%} profit, new stop {min_exit:.2f}"
                        )

    def _check_time_conditions(self, state: PositionRiskState) -> Optional[str]:
        """Check time-based exit conditions."""
        hours_held = (datetime.now() - state.entry_time).total_seconds() / 3600

        # Check max hold time
        if state.max_hold_time_hours and hours_held >= state.max_hold_time_hours:
            log.warning(
                f"[RISK] {state.symbol}: Max hold time ({state.max_hold_time_hours}h) exceeded"
            )
            return "time_exit"

        # Check stagnant position
        if state.status == PositionStatus.STAGNANT:
            log.warning(
                f"[RISK] {state.symbol}: Position stagnant for {hours_held:.1f}h"
            )
            return "stagnant_exit"

        return None

    def get_adjusted_position_size(
        self,
        base_size: float,
        current_drawdown: float,
        current_volatility_percentile: float,
        confidence: float = 0.5,
    ) -> float:
        """
        Adjust position size based on current conditions.

        Args:
            base_size: Calculated base position size
            current_drawdown: Current account drawdown (0-1)
            current_volatility_percentile: Current volatility (0-100)
            confidence: Signal confidence (0-1)

        Returns:
            Adjusted position size
        """
        adjustment = 1.0

        # 1. Drawdown adjustment
        if current_drawdown >= self.SEVERE_DRAWDOWN:
            adjustment *= 0.0  # No new trades
            log.warning("[RISK] Severe drawdown - blocking new trades")
        elif current_drawdown >= self.MODERATE_DRAWDOWN:
            adjustment *= 0.5
            log.info("[RISK] Moderate drawdown - reducing size 50%")
        elif current_drawdown >= self.LIGHT_DRAWDOWN:
            adjustment *= 0.75
            log.info("[RISK] Light drawdown - reducing size 25%")

        # 2. Volatility adjustment
        if current_volatility_percentile > 80:
            adjustment *= 0.6  # High volatility - reduce significantly
        elif current_volatility_percentile > 60:
            adjustment *= 0.8
        elif current_volatility_percentile < 20:
            adjustment *= 1.1  # Low volatility - can increase slightly

        # 3. Confidence adjustment
        if confidence >= 0.8:
            adjustment *= 1.1  # High confidence
        elif confidence < 0.5:
            adjustment *= 0.8  # Low confidence

        # 4. Consecutive loss adjustment
        if self._consecutive_losses >= 3:
            adjustment *= 0.7
            log.info(
                f"[RISK] {self._consecutive_losses} consecutive losses - reducing size 30%"
            )
        elif self._consecutive_losses >= 5:
            adjustment *= 0.5
            log.warning(
                f"[RISK] {self._consecutive_losses} consecutive losses - reducing size 50%"
            )

        adjusted_size = base_size * adjustment

        log.info(
            f"[RISK] Position size adjusted: {base_size:.4f} -> {adjusted_size:.4f} "
            f"(adjustment: {adjustment:.2f})"
        )

        return adjusted_size

    def calculate_dynamic_stop(
        self,
        entry_price: float,
        side: str,
        atr: float,
        volatility_regime: str = "normal",
        trend_strength: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit based on conditions.

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr: Current ATR
            volatility_regime: 'low', 'normal', 'high'
            trend_strength: ADX-based strength (0-1)

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Base multipliers
        sl_mult = self.DEFAULT_ATR_STOP_MULT
        tp_mult = self.DEFAULT_ATR_STOP_MULT * 2  # 2:1 R:R default

        # Adjust for volatility regime
        if volatility_regime == "high":
            sl_mult *= 1.3  # Wider stops in high volatility
            tp_mult *= 1.2
        elif volatility_regime == "low":
            sl_mult *= 0.8  # Tighter stops in low volatility
            tp_mult *= 0.9

        # Adjust for trend strength
        if trend_strength > 0.7:
            # Strong trend - can use tighter stops, wider targets
            sl_mult *= 0.9
            tp_mult *= 1.3
        elif trend_strength < 0.3:
            # Weak trend - need wider stops, conservative targets
            sl_mult *= 1.2
            tp_mult *= 0.8

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        if side == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit

    def should_scale_out(
        self,
        state: PositionRiskState,
        current_price: float,
    ) -> Tuple[bool, float]:
        """
        Check if position should scale out.

        Returns:
            Tuple of (should_scale, portion_to_close)
        """
        if state.scaled_out_size >= state.original_size * 0.5:
            # Already scaled out 50%
            return False, 0.0

        entry = state.entry_price

        # Calculate R-multiple
        if state.side == "long":
            r_mult = (current_price - entry) / (entry - state.dynamic_stop.initial_stop)
        else:
            r_mult = (entry - current_price) / (state.dynamic_stop.initial_stop - entry)

        # Scale out rules
        if r_mult >= 2.0 and state.scaled_out_size == 0:
            # First scale out at 2R - take 25%
            return True, 0.25
        elif r_mult >= 3.0 and state.scaled_out_size < state.original_size * 0.25:
            # Second scale out at 3R - take another 25%
            return True, 0.25

        return False, 0.0

    def record_trade_close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[float]:
        """
        Record a trade closure and update risk state.

        Returns:
            PnL of the trade
        """
        if symbol not in self._positions:
            return None

        state = self._positions[symbol]

        # Calculate PnL
        if state.side == "long":
            pnl = (exit_price - state.entry_price) * state.position_size
        else:
            pnl = (state.entry_price - exit_price) * state.position_size

        # Update counters
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Record trade
        self._trade_history.append(
            {
                "symbol": symbol,
                "side": state.side,
                "entry_price": state.entry_price,
                "exit_price": exit_price,
                "position_size": state.position_size,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "entry_time": state.entry_time.isoformat(),
                "exit_time": datetime.now().isoformat(),
                "mae": state.metrics.mae,
                "mfe": state.metrics.mfe,
            }
        )

        # Keep last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

        # Remove position
        del self._positions[symbol]

        log.info(
            f"[RISK] Trade closed: {symbol} {state.side} PnL=${pnl:.2f} ({exit_reason})"
        )
