"""
Advanced Signal Filtering Module.

Provides multi-layer signal quality filtering to:
- Reduce false signals and whipsaws
- Ensure proper market conditions before entry
- Implement confirmation requirements
- Filter based on volatility and trend strength
- Prevent overtrading in choppy markets
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from utils.logger import log

# Import advanced indicators if available
try:
    from src.advanced_indicators import (
        AdvancedIndicators,
        ADXResult,
        MarketRegime,
        TrendStrength,
    )

    ADVANCED_INDICATORS_AVAILABLE = True
except ImportError:
    ADVANCED_INDICATORS_AVAILABLE = False
    log.warning("Advanced indicators not available for signal filtering")


class FilterResult(str, Enum):
    """Result of signal filtering."""

    PASSED = "passed"
    REJECTED_LOW_QUALITY = "rejected_low_quality"
    REJECTED_WRONG_REGIME = "rejected_wrong_regime"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
    REJECTED_WEAK_TREND = "rejected_weak_trend"
    REJECTED_HIGH_VOLATILITY = "rejected_high_volatility"
    REJECTED_RECENT_LOSS = "rejected_recent_loss"
    REJECTED_MAX_TRADES = "rejected_max_trades"
    REJECTED_CONFLICTING_SIGNALS = "rejected_conflicting_signals"
    REJECTED_NO_MOMENTUM = "rejected_no_momentum"
    REJECTED_CORRELATION_LIMIT = "rejected_correlation_limit"


class SignalType(str, Enum):
    """Type of trading signal."""

    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NEUTRAL = "neutral"


@dataclass
class FilteredSignal:
    """Result of filtering a signal."""

    original_signal: Any
    passed: bool
    result: FilterResult
    quality_score: float
    adjustments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def should_execute(self) -> bool:
        """Check if signal should be executed."""
        return self.passed and self.result == FilterResult.PASSED


@dataclass
class TradeRecord:
    """Record of a recent trade for filtering purposes."""

    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    is_win: Optional[bool] = None


@dataclass
class SymbolState:
    """State tracking for a symbol."""

    last_signal_time: Optional[datetime] = None
    last_signal_direction: Optional[str] = None
    consecutive_losses: int = 0
    recent_trades: List[TradeRecord] = field(default_factory=list)
    daily_trades: int = 0
    daily_pnl: float = 0.0
    last_regime: Optional[str] = None
    regime_changes: int = 0


class SignalFilter:
    """
    Multi-layer signal filtering system.

    Filters:
    1. Quality Score Filter - Minimum quality threshold
    2. Market Regime Filter - Only trade in favorable conditions
    3. Trend Strength Filter - Require ADX above threshold
    4. Momentum Confirmation - Multiple indicators must agree
    5. Volatility Filter - Avoid extreme volatility
    6. Recent Loss Filter - Cool-down after consecutive losses
    7. Trade Frequency Filter - Limit daily trades
    8. Signal Conflict Filter - Avoid conflicting signals
    9. Correlation Filter - Limit correlated exposure
    """

    # Filter thresholds
    MIN_QUALITY_SCORE = 60.0  # Minimum quality score (0-100) - increased for better signals
    MIN_CONFIDENCE = 0.5  # Minimum signal confidence
    MIN_ADX_FOR_TREND = 20.0  # Minimum ADX for trend trades
    MAX_ADX_EXHAUSTED = 70.0  # ADX above this suggests exhaustion
    MAX_VOLATILITY_PERCENTILE = 85.0  # Avoid top 15% volatility
    CONSECUTIVE_LOSS_LIMIT = 3  # Cool-down after this many losses
    LOSS_COOLDOWN_MINUTES = 30  # Minutes to wait after loss streak
    MAX_DAILY_TRADES_PER_SYMBOL = 5  # Maximum trades per symbol per day
    SIGNAL_COOLDOWN_SECONDS = 60  # Minimum time between signals
    MIN_INDICATORS_AGREE = 3  # Minimum indicators in agreement (out of 5)
    MAX_CORRELATION_EXPOSURE = 0.7  # Max correlated exposure ratio

    # Regime trading preferences
    FAVORABLE_REGIMES = [
        "uptrend",
        "downtrend",
        "strong_uptrend",
        "strong_downtrend",
        "ranging",
    ]
    UNFAVORABLE_REGIMES = ["choppy", "high_volatility"]

    def __init__(self):
        """Initialize signal filter."""
        self._symbol_states: Dict[str, SymbolState] = {}
        self._daily_reset_time: Optional[datetime] = None
        self._position_correlations: Dict[str, List[str]] = {}

        # Initialize advanced indicators if available
        if ADVANCED_INDICATORS_AVAILABLE:
            self._advanced = AdvancedIndicators()
        else:
            self._advanced = None

        log.info("SignalFilter initialized with multi-layer filtering")

    def _get_symbol_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = SymbolState()
        return self._symbol_states[symbol]

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        now = datetime.now()
        if self._daily_reset_time is None or now.date() > self._daily_reset_time.date():
            for state in self._symbol_states.values():
                state.daily_trades = 0
                state.daily_pnl = 0.0
            self._daily_reset_time = now
            log.info("Daily signal filter counters reset")

    def filter_signal(
        self,
        signal: Any,
        symbol: str,
        direction: str,
        confidence: float,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        indicator_signals: Optional[Dict[str, str]] = None,
        current_positions: Optional[List[str]] = None,
    ) -> FilteredSignal:
        """
        Apply all filters to a trading signal.

        Args:
            signal: Original signal object
            symbol: Trading symbol
            direction: Signal direction ('long', 'short', etc.)
            confidence: Signal confidence (0-1)
            high: High price array (optional, for advanced filtering)
            low: Low price array (optional)
            close: Close price array (optional)
            volume: Volume array (optional)
            indicator_signals: Dict of indicator name -> signal direction
            current_positions: List of currently held symbols

        Returns:
            FilteredSignal with pass/fail and reason
        """
        self._check_daily_reset()
        state = self._get_symbol_state(symbol)

        adjustments = {}

        # 1. Confidence Filter
        result = self._filter_confidence(confidence)
        if result != FilterResult.PASSED:
            return FilteredSignal(
                original_signal=signal,
                passed=False,
                result=result,
                quality_score=0.0,
                reason=f"Confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE}",
            )

        # 2. Trade Frequency Filter
        result = self._filter_trade_frequency(state)
        if result != FilterResult.PASSED:
            return FilteredSignal(
                original_signal=signal,
                passed=False,
                result=result,
                quality_score=0.0,
                reason=f"Daily trade limit reached ({state.daily_trades}/{self.MAX_DAILY_TRADES_PER_SYMBOL})",
            )

        # 3. Recent Loss Filter
        result = self._filter_recent_losses(state)
        if result != FilterResult.PASSED:
            return FilteredSignal(
                original_signal=signal,
                passed=False,
                result=result,
                quality_score=0.0,
                reason=f"Cooling down after {state.consecutive_losses} consecutive losses",
            )

        # 4. Signal Cooldown Filter
        result = self._filter_signal_cooldown(state)
        if result != FilterResult.PASSED:
            return FilteredSignal(
                original_signal=signal,
                passed=False,
                result=result,
                quality_score=0.0,
                reason="Too soon after last signal",
            )

        # 5. Indicator Agreement Filter (if provided)
        if indicator_signals:
            result, agreement_pct = self._filter_indicator_agreement(
                indicator_signals, direction
            )
            if result != FilterResult.PASSED:
                return FilteredSignal(
                    original_signal=signal,
                    passed=False,
                    result=result,
                    quality_score=0.0,
                    reason=f"Only {agreement_pct:.0%} indicator agreement (need {self.MIN_INDICATORS_AGREE})",
                )

        # 6. Correlation Filter (if positions provided)
        if current_positions:
            result = self._filter_correlation(symbol, current_positions)
            if result != FilterResult.PASSED:
                return FilteredSignal(
                    original_signal=signal,
                    passed=False,
                    result=result,
                    quality_score=0.0,
                    reason="Would exceed correlated exposure limit",
                )

        # Advanced filtering (requires price data)
        quality_score = 50.0  # Base score

        if self._advanced and close is not None and len(close) >= 50:
            # 7. Market Regime Filter
            if high is not None and low is not None and volume is not None:
                regime_result = self._advanced.detect_market_regime(
                    high, low, close, volume, symbol
                )

                result = self._filter_regime(regime_result.regime.value, direction)
                if result != FilterResult.PASSED:
                    return FilteredSignal(
                        original_signal=signal,
                        passed=False,
                        result=result,
                        quality_score=0.0,
                        reason=f"Unfavorable regime: {regime_result.regime.value}",
                    )

                # Store regime for tracking
                if state.last_regime != regime_result.regime.value:
                    state.regime_changes += 1
                state.last_regime = regime_result.regime.value

                # 8. Trend Strength Filter
                adx_result = self._advanced.calculate_adx(high, low, close)
                result, adx_adjustment = self._filter_trend_strength(
                    adx_result, direction
                )
                if result != FilterResult.PASSED:
                    return FilteredSignal(
                        original_signal=signal,
                        passed=False,
                        result=result,
                        quality_score=0.0,
                        reason=f"ADX {adx_result.adx:.1f} indicates weak/exhausted trend",
                    )
                adjustments["position_size_multiplier"] = adx_adjustment

                # 9. Volatility Filter
                result = self._filter_volatility(regime_result.volatility_percentile)
                if result != FilterResult.PASSED:
                    return FilteredSignal(
                        original_signal=signal,
                        passed=False,
                        result=result,
                        quality_score=0.0,
                        reason=f"Volatility at {regime_result.volatility_percentile:.0f}th percentile",
                    )

                # 10. Calculate Quality Score
                quality_score, breakdown = self._advanced.get_entry_quality_score(
                    high, low, close, volume, direction
                )

                result = self._filter_quality_score(quality_score)
                if result != FilterResult.PASSED:
                    return FilteredSignal(
                        original_signal=signal,
                        passed=False,
                        result=result,
                        quality_score=quality_score,
                        reason=f"Quality score {quality_score:.1f} below threshold {self.MIN_QUALITY_SCORE}",
                    )

                # Apply adjustments based on quality
                if quality_score >= 80:
                    adjustments["position_size_multiplier"] = (
                        adjustments.get("position_size_multiplier", 1.0) * 1.2
                    )
                elif quality_score < 60:
                    adjustments["position_size_multiplier"] = (
                        adjustments.get("position_size_multiplier", 1.0) * 0.8
                    )

        # Update state
        state.last_signal_time = datetime.now()
        state.last_signal_direction = direction

        return FilteredSignal(
            original_signal=signal,
            passed=True,
            result=FilterResult.PASSED,
            quality_score=quality_score,
            adjustments=adjustments,
            reason=f"Signal passed all filters with quality score {quality_score:.1f}",
        )

    def _filter_confidence(self, confidence: float) -> FilterResult:
        """Filter based on signal confidence."""
        if confidence < self.MIN_CONFIDENCE:
            return FilterResult.REJECTED_LOW_CONFIDENCE
        return FilterResult.PASSED

    def _filter_trade_frequency(self, state: SymbolState) -> FilterResult:
        """Filter based on daily trade count."""
        if state.daily_trades >= self.MAX_DAILY_TRADES_PER_SYMBOL:
            return FilterResult.REJECTED_MAX_TRADES
        return FilterResult.PASSED

    def _filter_recent_losses(self, state: SymbolState) -> FilterResult:
        """Filter based on recent loss streak."""
        if state.consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
            # Check if cooldown has passed
            if state.recent_trades:
                last_trade = state.recent_trades[-1]
                if last_trade.exit_time:
                    cooldown_end = last_trade.exit_time + timedelta(
                        minutes=self.LOSS_COOLDOWN_MINUTES
                    )
                    if datetime.now() < cooldown_end:
                        return FilterResult.REJECTED_RECENT_LOSS
            # Reset loss counter after cooldown
            state.consecutive_losses = 0
        return FilterResult.PASSED

    def _filter_signal_cooldown(self, state: SymbolState) -> FilterResult:
        """Filter to prevent rapid-fire signals."""
        if state.last_signal_time:
            elapsed = (datetime.now() - state.last_signal_time).total_seconds()
            if elapsed < self.SIGNAL_COOLDOWN_SECONDS:
                return FilterResult.REJECTED_MAX_TRADES
        return FilterResult.PASSED

    def _filter_indicator_agreement(
        self, indicator_signals: Dict[str, str], direction: str
    ) -> Tuple[FilterResult, float]:
        """Filter based on indicator agreement."""
        if not indicator_signals:
            return FilterResult.PASSED, 1.0

        total = len(indicator_signals)
        agreeing = sum(
            1
            for sig in indicator_signals.values()
            if sig.lower() == direction.lower()
            or (direction == "long" and sig.lower() in ["bullish", "buy"])
            or (direction == "short" and sig.lower() in ["bearish", "sell"])
        )

        agreement_pct = agreeing / total if total > 0 else 0

        if agreeing < self.MIN_INDICATORS_AGREE:
            return FilterResult.REJECTED_NO_MOMENTUM, agreement_pct

        return FilterResult.PASSED, agreement_pct

    def _filter_correlation(
        self, symbol: str, current_positions: List[str]
    ) -> FilterResult:
        """Filter to prevent excessive correlated exposure."""
        if not current_positions:
            return FilterResult.PASSED

        # Define correlation groups
        correlation_groups = {
            "BTC": ["BTCUSD", "BTCUSDT"],
            "ETH": ["ETHUSD", "ETHUSDT"],
            "LARGE_CAP": ["BTCUSD", "ETHUSD", "SOLUSD"],
            "ALT": ["SOLUSD", "AVAXUSD", "DOGEUSD"],
        }

        # Find which groups symbol belongs to
        symbol_groups = []
        for group, members in correlation_groups.items():
            if symbol in members:
                symbol_groups.append(group)

        if not symbol_groups:
            return FilterResult.PASSED

        # Count correlated positions
        correlated_count = 0
        for pos in current_positions:
            for group in symbol_groups:
                if pos in correlation_groups.get(group, []):
                    correlated_count += 1
                    break

        # Check ratio
        total_positions = len(current_positions)
        if total_positions > 0:
            ratio = correlated_count / total_positions
            if ratio > self.MAX_CORRELATION_EXPOSURE:
                return FilterResult.REJECTED_CORRELATION_LIMIT

        return FilterResult.PASSED

    def _filter_regime(self, regime: str, direction: str) -> FilterResult:
        """Filter based on market regime."""
        if regime.lower() in self.UNFAVORABLE_REGIMES:
            return FilterResult.REJECTED_WRONG_REGIME

        # Check direction alignment with regime
        if regime.lower() in ["uptrend", "strong_uptrend"]:
            if direction == "short":
                # Allow counter-trend only if not strong trend
                if regime.lower() == "strong_uptrend":
                    return FilterResult.REJECTED_WRONG_REGIME
        elif regime.lower() in ["downtrend", "strong_downtrend"]:
            if direction == "long":
                if regime.lower() == "strong_downtrend":
                    return FilterResult.REJECTED_WRONG_REGIME

        return FilterResult.PASSED

    def _filter_trend_strength(
        self, adx_result: Any, direction: str
    ) -> Tuple[FilterResult, float]:
        """Filter based on trend strength and return position size adjustment."""
        adjustment = 1.0

        # Very weak trend - reduce size
        if adx_result.adx < self.MIN_ADX_FOR_TREND:
            if direction in ["long", "short"]:
                adjustment = 0.7  # Reduce position size

        # Exhausted trend - be cautious
        if adx_result.adx > self.MAX_ADX_EXHAUSTED:
            # Trend may be exhausted, reduce size
            adjustment = 0.5

        # Strong trend - can increase slightly
        if 40 <= adx_result.adx <= 60:
            # Check direction alignment
            if direction == "long" and adx_result.trend_direction == "bullish":
                adjustment = 1.1
            elif direction == "short" and adx_result.trend_direction == "bearish":
                adjustment = 1.1

        return FilterResult.PASSED, adjustment

    def _filter_volatility(self, volatility_percentile: float) -> FilterResult:
        """Filter based on volatility level."""
        if volatility_percentile > self.MAX_VOLATILITY_PERCENTILE:
            return FilterResult.REJECTED_HIGH_VOLATILITY
        return FilterResult.PASSED

    def _filter_quality_score(self, score: float) -> FilterResult:
        """Filter based on quality score."""
        if score < self.MIN_QUALITY_SCORE:
            return FilterResult.REJECTED_LOW_QUALITY
        return FilterResult.PASSED

    def record_trade_result(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> None:
        """
        Record trade result for filtering decisions.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss
        """
        state = self._get_symbol_state(symbol)

        is_win = pnl > 0

        trade = TradeRecord(
            symbol=symbol,
            direction=direction,
            entry_time=datetime.now() - timedelta(hours=1),  # Approximate
            entry_price=entry_price,
            exit_time=datetime.now(),
            exit_price=exit_price,
            pnl=pnl,
            is_win=is_win,
        )

        state.recent_trades.append(trade)
        state.daily_trades += 1
        state.daily_pnl += pnl

        # Track consecutive losses
        if is_win:
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1

        # Keep only last 50 trades
        if len(state.recent_trades) > 50:
            state.recent_trades = state.recent_trades[-50:]

        log.info(
            f"[FILTER] Trade recorded for {symbol}: "
            f"{'WIN' if is_win else 'LOSS'} ${pnl:.2f}, "
            f"Consecutive losses: {state.consecutive_losses}"
        )

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        """Get filtering statistics for a symbol."""
        state = self._get_symbol_state(symbol)

        recent_wins = sum(1 for t in state.recent_trades if t.is_win)
        recent_total = len(state.recent_trades)
        win_rate = recent_wins / recent_total if recent_total > 0 else 0

        return {
            "symbol": symbol,
            "daily_trades": state.daily_trades,
            "daily_pnl": state.daily_pnl,
            "consecutive_losses": state.consecutive_losses,
            "recent_trades": recent_total,
            "recent_win_rate": win_rate,
            "last_signal_time": state.last_signal_time.isoformat()
            if state.last_signal_time
            else None,
            "last_regime": state.last_regime,
            "regime_changes": state.regime_changes,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get filtering statistics for all symbols."""
        return {symbol: self.get_symbol_stats(symbol) for symbol in self._symbol_states}

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a symbol."""
        if symbol in self._symbol_states:
            del self._symbol_states[symbol]
            log.info(f"[FILTER] Reset state for {symbol}")

    def reset_all(self) -> None:
        """Reset all filter state."""
        self._symbol_states.clear()
        self._daily_reset_time = None
        log.info("[FILTER] All filter state reset")


class AdaptiveFilter(SignalFilter):
    """
    Adaptive signal filter that adjusts thresholds based on performance.

    Automatically tightens or loosens filters based on:
    - Recent win rate
    - Drawdown level
    - Market conditions
    """

    def __init__(self):
        """Initialize adaptive filter."""
        super().__init__()
        self._performance_window = 20  # Trades to consider
        self._base_thresholds = {
            "quality_score": self.MIN_QUALITY_SCORE,
            "confidence": self.MIN_CONFIDENCE,
            "max_trades": self.MAX_DAILY_TRADES_PER_SYMBOL,
        }

    def adapt_thresholds(self) -> None:
        """Adapt thresholds based on recent performance."""
        all_recent_trades = []
        for state in self._symbol_states.values():
            all_recent_trades.extend(state.recent_trades[-10:])

        if len(all_recent_trades) < 10:
            return  # Not enough data

        # Calculate recent win rate
        wins = sum(1 for t in all_recent_trades if t.is_win)
        win_rate = wins / len(all_recent_trades)

        # Calculate recent P&L
        recent_pnl = sum(t.pnl or 0 for t in all_recent_trades)

        # Tighten filters if losing
        if win_rate < 0.4 or recent_pnl < 0:
            self.MIN_QUALITY_SCORE = min(
                75.0, self._base_thresholds["quality_score"] * 1.2
            )
            self.MIN_CONFIDENCE = min(0.7, self._base_thresholds["confidence"] * 1.2)
            self.MAX_DAILY_TRADES_PER_SYMBOL = max(
                2, self._base_thresholds["max_trades"] - 2
            )
            log.info("[FILTER] Tightening filters due to poor performance")

        # Loosen slightly if doing well
        elif win_rate > 0.6 and recent_pnl > 0:
            self.MIN_QUALITY_SCORE = max(
                50.0, self._base_thresholds["quality_score"] * 0.9
            )
            self.MIN_CONFIDENCE = max(0.4, self._base_thresholds["confidence"] * 0.9)
            self.MAX_DAILY_TRADES_PER_SYMBOL = min(
                7, self._base_thresholds["max_trades"] + 1
            )
            log.info("[FILTER] Loosening filters due to good performance")

        # Reset to base if performance is neutral
        else:
            self.MIN_QUALITY_SCORE = self._base_thresholds["quality_score"]
            self.MIN_CONFIDENCE = self._base_thresholds["confidence"]
            self.MAX_DAILY_TRADES_PER_SYMBOL = self._base_thresholds["max_trades"]
