"""
Enhanced Trading Strategy Module v2.

Integrates all advanced features for improved trading performance:
- Advanced technical indicators (ADX, VWAP, Volume Profile)
- Market regime detection (trending/ranging/choppy)
- Multi-layer signal filtering
- Enhanced risk management with dynamic stops
- Confidence-weighted position sizing
- Momentum confirmation requirements

This module provides higher quality trades with:
- Fewer false signals through stricter filtering
- Better entry timing with market regime awareness
- Improved risk management with dynamic stops
- Adaptive position sizing based on conditions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from src.delta_client import DeltaExchangeClient, OrderSide, Position
from src.risk_manager import PositionSizing, RiskManager
from src.technical_analysis import (
    MultiTimeframeAnalyzer,
    Signal,
    TechnicalAnalysisResult,
    TechnicalAnalyzer,
)
from utils.logger import log

# Import enhanced modules
try:
    from src.advanced_indicators import (
        AdvancedIndicators,
        ADXResult,
        MarketRegime,
        MarketRegimeResult,
        MomentumResult,
        TrendStrength,
        VWAPResult,
    )

    ADVANCED_INDICATORS_AVAILABLE = True
except ImportError:
    ADVANCED_INDICATORS_AVAILABLE = False
    log.warning("Advanced indicators not available")

try:
    from src.signal_filter import AdaptiveFilter, FilteredSignal, FilterResult

    SIGNAL_FILTER_AVAILABLE = True
except ImportError:
    SIGNAL_FILTER_AVAILABLE = False
    log.warning("Signal filter not available")

try:
    from src.enhanced_risk import (
        DynamicStop,
        EnhancedRiskManager,
        PositionRiskState,
        RiskMetrics,
    )

    ENHANCED_RISK_AVAILABLE = True
except ImportError:
    ENHANCED_RISK_AVAILABLE = False
    log.warning("Enhanced risk manager not available")


class TradeAction(str, Enum):
    """Possible trading actions."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SCALE_OUT = "scale_out"
    HOLD = "hold"


class TradeQuality(str, Enum):
    """Quality classification of trade signal."""

    EXCELLENT = "excellent"  # Score 80+
    GOOD = "good"  # Score 65-80
    ACCEPTABLE = "acceptable"  # Score 55-65
    POOR = "poor"  # Score < 55


@dataclass
class EnhancedTradeDecision:
    """Complete trade decision with all necessary information."""

    symbol: str
    action: TradeAction
    signal: Signal
    confidence: float
    quality_score: float
    quality_grade: TradeQuality
    entry_price: float

    # Position management
    position_size: Optional[float] = None
    adjusted_position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Market context
    market_regime: Optional[str] = None
    trend_strength: Optional[str] = None
    volatility_percentile: Optional[float] = None

    # Analysis details
    reason: str = ""
    filter_result: Optional[str] = None
    adjustments: Dict[str, Any] = field(default_factory=dict)

    # Technical analysis
    ta_result: Optional[TechnicalAnalysisResult] = None
    adx_result: Optional[Any] = None
    momentum_result: Optional[Any] = None

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_actionable(self) -> bool:
        """Check if this decision requires action."""
        return self.action != TradeAction.HOLD

    @property
    def has_high_quality(self) -> bool:
        """Check if this is a high-quality trade."""
        return self.quality_grade in [TradeQuality.EXCELLENT, TradeQuality.GOOD]


class EnhancedTradingStrategy:
    """
    Enhanced trading strategy with comprehensive market analysis.

    Features:
    1. Multi-layer signal filtering
    2. Market regime detection
    3. Advanced indicator confirmation
    4. Dynamic risk management
    5. Confidence-weighted sizing

    Strategy Logic:
    1. Detect market regime (trending/ranging/volatile)
    2. Run technical analysis with strict thresholds
    3. Validate signal with advanced indicators
    4. Apply signal filter for quality control
    5. Calculate dynamic stops based on ATR
    6. Adjust position size based on conditions
    7. Output high-quality trade decision
    """

    # Quality score thresholds
    EXCELLENT_QUALITY_THRESHOLD = 80.0
    GOOD_QUALITY_THRESHOLD = 65.0
    ACCEPTABLE_QUALITY_THRESHOLD = 55.0
    MIN_QUALITY_THRESHOLD = 55.0

    # Confidence requirements
    MIN_CONFIDENCE = 0.5
    HIGH_CONFIDENCE = 0.75

    def __init__(
        self,
        client: DeltaExchangeClient,
        executor=None,
        dry_run: bool = False,
        strict_mode: bool = True,
    ):
        """
        Initialize enhanced trading strategy.

        Args:
            client: Delta Exchange API client
            executor: TradeExecutor instance for performance tracking
            dry_run: If True, don't execute real trades
            strict_mode: If True, use stricter signal requirements
        """
        self.client = client
        self.executor = executor
        self.dry_run = dry_run
        self.strict_mode = strict_mode

        # Core analyzers
        self.analyzer = TechnicalAnalyzer(strict_mode=strict_mode)
        self.risk_manager = RiskManager()
        self.config = settings.trading

        # Enhanced components
        if ADVANCED_INDICATORS_AVAILABLE:
            self.advanced = AdvancedIndicators()
        else:
            self.advanced = None

        if SIGNAL_FILTER_AVAILABLE:
            self.signal_filter = AdaptiveFilter()
        else:
            self.signal_filter = None

        if ENHANCED_RISK_AVAILABLE:
            self.enhanced_risk = EnhancedRiskManager()
        else:
            self.enhanced_risk = None

        # State tracking
        self._regime_cache: Dict[str, Tuple[MarketRegimeResult, datetime]] = {}
        self._regime_cache_ttl = 300  # 5 minutes

        log.info(
            f"EnhancedTradingStrategy initialized "
            f"(strict_mode={strict_mode}, dry_run={dry_run})"
        )
        log.info(
            f"  Advanced Indicators: {ADVANCED_INDICATORS_AVAILABLE}, "
            f"Signal Filter: {SIGNAL_FILTER_AVAILABLE}, "
            f"Enhanced Risk: {ENHANCED_RISK_AVAILABLE}"
        )

    def analyze_market(
        self, symbol: str, include_advanced: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Comprehensive market analysis for a symbol.

        Args:
            symbol: Trading pair symbol
            include_advanced: Include advanced indicator analysis

        Returns:
            Dictionary with analysis results or None if failed
        """
        try:
            # Fetch candles
            candles = self.client.get_candles(
                symbol=symbol, resolution=self.config.candle_interval
            )

            if len(candles) < 50:
                log.warning(f"Insufficient candles for {symbol}: {len(candles)}")
                return None

            # Extract price arrays
            close = np.array([c.close for c in candles])
            high = np.array([c.high for c in candles])
            low = np.array([c.low for c in candles])

            # Get volume if available
            volume = np.array([getattr(c, "volume", 0) for c in candles])
            has_volume = np.sum(volume) > 0

            # Basic technical analysis
            ta_result = self.analyzer.analyze(close, high, low, symbol)

            # Get current price
            try:
                ticker = self.client.get_ticker(symbol)
                current_price = float(
                    ticker.get("mark_price", 0) or ticker.get("close", 0)
                )
            except Exception as e:
                log.error(f"Failed to get ticker for {symbol}: {e}")
                current_price = float(close[-1])

            result = {
                "symbol": symbol,
                "current_price": current_price,
                "ta_result": ta_result,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume if has_volume else None,
            }

            # Advanced analysis
            if include_advanced and self.advanced:
                # Calculate ATR
                atr = self.analyzer.calculate_atr(high, low, close)
                result["atr"] = atr

                # ADX for trend strength
                adx_result = self.advanced.calculate_adx(high, low, close)
                result["adx_result"] = adx_result

                # Market regime detection
                if has_volume:
                    regime_result = self._get_market_regime(
                        symbol, high, low, close, volume
                    )
                else:
                    # Fallback regime detection without volume
                    regime_result = self._get_market_regime(
                        symbol, high, low, close, np.ones_like(close)
                    )
                result["regime_result"] = regime_result

                # Momentum analysis
                if has_volume:
                    momentum = self.advanced.calculate_momentum_suite(
                        high, low, close, volume
                    )
                    result["momentum"] = momentum

                # VWAP if volume available
                if has_volume:
                    vwap = self.advanced.calculate_vwap(high, low, close, volume)
                    result["vwap"] = vwap

            return result

        except Exception as e:
            log.error(f"Market analysis failed for {symbol}: {e}")
            import traceback

            log.error(traceback.format_exc())
            return None

    def _get_market_regime(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Optional[MarketRegimeResult]:
        """Get market regime with caching."""
        if not self.advanced:
            return None

        # Check cache
        cache_key = symbol
        if cache_key in self._regime_cache:
            cached_result, cached_time = self._regime_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._regime_cache_ttl:
                return cached_result

        # Calculate fresh regime
        regime_result = self.advanced.detect_market_regime(
            high, low, close, volume, symbol
        )

        # Cache result
        self._regime_cache[cache_key] = (regime_result, datetime.now())

        return regime_result

    def make_decision(
        self,
        symbol: str,
        available_balance: float,
        current_positions: List[Position],
    ) -> EnhancedTradeDecision:
        """
        Make a comprehensive trading decision for a symbol.

        Args:
            symbol: Trading pair symbol
            available_balance: Available capital
            current_positions: List of current open positions

        Returns:
            EnhancedTradeDecision with action and details
        """
        # Run comprehensive market analysis
        analysis = self.analyze_market(symbol)

        if analysis is None:
            return self._create_hold_decision(
                symbol, 0.0, "Market analysis failed", Signal.HOLD, 0.0
            )

        ta_result = analysis["ta_result"]
        current_price = analysis["current_price"]

        # Check for existing position
        existing_position = self._find_position(symbol, current_positions)

        if existing_position:
            return self._decide_with_position(
                symbol, analysis, existing_position, current_price
            )
        else:
            return self._decide_without_position(
                symbol, analysis, current_price, available_balance, current_positions
            )

    def _find_position(
        self, symbol: str, positions: List[Position]
    ) -> Optional[Position]:
        """Find existing position for symbol."""
        for pos in positions:
            if pos.product_symbol == symbol and abs(pos.size) > 0:
                return pos
        return None

    def _decide_with_position(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        position: Position,
        current_price: float,
    ) -> EnhancedTradeDecision:
        """Make decision when we have an existing position."""
        ta_result = analysis["ta_result"]
        is_long = position.size > 0

        # Check if signal reversed
        if is_long and self.analyzer.should_close_long(ta_result):
            return EnhancedTradeDecision(
                symbol=symbol,
                action=TradeAction.CLOSE_LONG,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                quality_score=50.0,
                quality_grade=TradeQuality.ACCEPTABLE,
                entry_price=current_price,
                reason="Signal reversed to bearish",
                ta_result=ta_result,
                market_regime=analysis.get("regime_result", {}).regime.value
                if analysis.get("regime_result")
                else None,
            )
        elif not is_long and self.analyzer.should_close_short(ta_result):
            return EnhancedTradeDecision(
                symbol=symbol,
                action=TradeAction.CLOSE_SHORT,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                quality_score=50.0,
                quality_grade=TradeQuality.ACCEPTABLE,
                entry_price=current_price,
                reason="Signal reversed to bullish",
                ta_result=ta_result,
                market_regime=analysis.get("regime_result", {}).regime.value
                if analysis.get("regime_result")
                else None,
            )

        # Check for scale out opportunity
        if self.enhanced_risk and symbol in self.enhanced_risk._positions:
            risk_state = self.enhanced_risk._positions[symbol]
            should_scale, portion = self.enhanced_risk.should_scale_out(
                risk_state, current_price
            )
            if should_scale:
                return EnhancedTradeDecision(
                    symbol=symbol,
                    action=TradeAction.SCALE_OUT,
                    signal=ta_result.combined_signal,
                    confidence=ta_result.confidence,
                    quality_score=60.0,
                    quality_grade=TradeQuality.GOOD,
                    entry_price=current_price,
                    position_size=position.size * portion,
                    reason=f"Scaling out {portion:.0%} at profit target",
                    ta_result=ta_result,
                )

        # Hold position
        return self._create_hold_decision(
            symbol,
            current_price,
            "Maintaining position",
            ta_result.combined_signal,
            ta_result.confidence,
        )

    def _decide_without_position(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        current_price: float,
        available_balance: float,
        current_positions: List[Position],
    ) -> EnhancedTradeDecision:
        """Make decision when we have no existing position."""
        ta_result = analysis["ta_result"]
        regime_result = analysis.get("regime_result")
        adx_result = analysis.get("adx_result")
        momentum = analysis.get("momentum")
        atr = analysis.get("atr", 0)

        # 1. Check risk limits
        risk = self.risk_manager.assess_trade_risk(
            available_balance, len(current_positions)
        )
        if not risk.can_trade:
            return self._create_hold_decision(
                symbol,
                current_price,
                risk.reason,
                ta_result.combined_signal,
                ta_result.confidence,
            )

        # 2. Determine signal direction
        if self.analyzer.should_enter_long(ta_result):
            direction = "long"
            action = TradeAction.OPEN_LONG
        elif self.analyzer.should_enter_short(ta_result):
            direction = "short"
            action = TradeAction.OPEN_SHORT
        else:
            return self._create_hold_decision(
                symbol,
                current_price,
                "No strong signal",
                ta_result.combined_signal,
                ta_result.confidence,
            )

        # 3. Check market regime filter
        if regime_result and not regime_result.should_trade:
            return self._create_hold_decision(
                symbol,
                current_price,
                f"Unfavorable regime: {regime_result.regime.value}",
                ta_result.combined_signal,
                ta_result.confidence,
                market_regime=regime_result.regime.value,
            )

        # 4. Check trend strength (ADX filter)
        if adx_result and adx_result.should_avoid_trading:
            return self._create_hold_decision(
                symbol,
                current_price,
                f"Weak trend (ADX={adx_result.adx:.1f})",
                ta_result.combined_signal,
                ta_result.confidence,
                trend_strength=adx_result.trend_strength.value,
            )

        # 5. Check momentum confirmation
        if momentum:
            momentum_aligned = self._check_momentum_alignment(momentum, direction)
            if not momentum_aligned:
                return self._create_hold_decision(
                    symbol,
                    current_price,
                    f"Momentum not aligned ({momentum.momentum_direction})",
                    ta_result.combined_signal,
                    ta_result.confidence,
                )

        # 6. Apply signal filter
        quality_score = self._calculate_quality_score(analysis, direction)

        if self.signal_filter:
            high = analysis.get("high")
            low = analysis.get("low")
            close = analysis.get("close")
            volume = analysis.get("volume")

            # Get indicator signals for filter
            indicator_signals = {
                ind.name: ind.signal.name for ind in ta_result.indicators
            }

            # Filter the signal
            filtered = self.signal_filter.filter_signal(
                signal=ta_result,
                symbol=symbol,
                direction=direction,
                confidence=ta_result.confidence,
                high=high,
                low=low,
                close=close,
                volume=volume,
                indicator_signals=indicator_signals,
                current_positions=[p.product_symbol for p in current_positions],
            )

            if not filtered.should_execute:
                return self._create_hold_decision(
                    symbol,
                    current_price,
                    f"Signal filtered: {filtered.result.value}",
                    ta_result.combined_signal,
                    ta_result.confidence,
                    filter_result=filtered.result.value,
                )

            # Use filter's quality score if available
            if filtered.quality_score > 0:
                quality_score = filtered.quality_score

            # Apply adjustments from filter
            adjustments = filtered.adjustments

        else:
            adjustments = {}

        # 7. Check minimum quality threshold
        if quality_score < self.MIN_QUALITY_THRESHOLD:
            return self._create_hold_decision(
                symbol,
                current_price,
                f"Quality score {quality_score:.1f} below threshold",
                ta_result.combined_signal,
                ta_result.confidence,
                quality_score=quality_score,
            )

        # 8. Calculate position sizing
        sizing = self._calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            available_balance=available_balance,
            atr=atr,
            quality_score=quality_score,
            regime_result=regime_result,
            adjustments=adjustments,
        )

        # 9. Determine quality grade
        quality_grade = self._get_quality_grade(quality_score)

        # 10. Build final decision
        decision = EnhancedTradeDecision(
            symbol=symbol,
            action=action,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            quality_score=quality_score,
            quality_grade=quality_grade,
            entry_price=current_price,
            position_size=sizing.size,
            adjusted_position_size=sizing.size,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            market_regime=regime_result.regime.value if regime_result else None,
            trend_strength=adx_result.trend_strength.value if adx_result else None,
            volatility_percentile=regime_result.volatility_percentile
            if regime_result
            else None,
            reason=f"High-quality {direction} signal "
            f"(quality={quality_score:.0f}, conf={ta_result.confidence:.0%})",
            adjustments=adjustments,
            ta_result=ta_result,
            adx_result=adx_result,
            momentum_result=momentum,
        )

        log.info(f"[STRATEGY] {symbol}: {action.value.upper()} signal generated")
        log.info(
            f"  Quality: {quality_grade.value} ({quality_score:.0f}/100), "
            f"Confidence: {ta_result.confidence:.0%}"
        )
        log.info(
            f"  Size: {sizing.size:.6f}, SL: {sizing.stop_loss_price:.2f}, "
            f"TP: {sizing.take_profit_price:.2f}"
        )
        if regime_result:
            log.info(
                f"  Regime: {regime_result.regime.value}, "
                f"Vol%: {regime_result.volatility_percentile:.0f}"
            )

        return decision

    def _check_momentum_alignment(
        self, momentum: MomentumResult, direction: str
    ) -> bool:
        """Check if momentum aligns with trade direction."""
        if direction == "long":
            # For long, want bullish momentum or oversold conditions
            if momentum.is_oversold:
                return True
            if momentum.momentum_direction == "bullish":
                return True
            # Allow neutral momentum with positive ROC
            if momentum.momentum_direction == "neutral" and momentum.roc > 0:
                return True
            return False
        else:
            # For short, want bearish momentum or overbought conditions
            if momentum.is_overbought:
                return True
            if momentum.momentum_direction == "bearish":
                return True
            if momentum.momentum_direction == "neutral" and momentum.roc < 0:
                return True
            return False

    def _calculate_quality_score(
        self, analysis: Dict[str, Any], direction: str
    ) -> float:
        """Calculate comprehensive quality score for the signal."""
        score = 50.0  # Base score

        ta_result = analysis.get("ta_result")
        regime_result = analysis.get("regime_result")
        adx_result = analysis.get("adx_result")
        momentum = analysis.get("momentum")

        # 1. Confidence contribution (max 20 points)
        if ta_result:
            score += ta_result.confidence * 20

        # 2. Trend strength contribution (max 15 points)
        if adx_result:
            if adx_result.trend_strength.value == "strong":
                score += 15
            elif adx_result.trend_strength.value == "moderate":
                score += 10
            elif adx_result.trend_strength.value == "weak":
                score += 5

            # Bonus for aligned trend direction
            if direction == "long" and adx_result.trend_direction == "bullish":
                score += 5
            elif direction == "short" and adx_result.trend_direction == "bearish":
                score += 5

        # 3. Regime favorability (max 10 points)
        if regime_result:
            if regime_result.is_favorable:
                score += 10
            elif regime_result.should_trade:
                score += 5

        # 4. Momentum alignment (max 10 points)
        if momentum:
            if self._check_momentum_alignment(momentum, direction):
                score += 10
                # Bonus for extreme readings
                if direction == "long" and momentum.is_oversold:
                    score += 5
                elif direction == "short" and momentum.is_overbought:
                    score += 5

        # 5. Indicator agreement (up to -15 penalty)
        if ta_result:
            bullish = sum(1 for ind in ta_result.indicators if ind.signal.value == 1)
            bearish = sum(1 for ind in ta_result.indicators if ind.signal.value == -1)
            total = len(ta_result.indicators)

            # Penalize conflicting signals
            if bullish > 0 and bearish > 0:
                conflict_ratio = min(bullish, bearish) / total
                score -= conflict_ratio * 15

        return min(100.0, max(0.0, score))

    def _calculate_position_size(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        available_balance: float,
        atr: float,
        quality_score: float,
        regime_result: Optional[MarketRegimeResult],
        adjustments: Dict[str, Any],
    ) -> PositionSizing:
        """Calculate position size with all adjustments."""
        # Get performance data for Kelly sizing
        performance_data = None
        if self.executor:
            performance_data = self.executor.get_performance_data()

        # Calculate base position size
        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            side=direction,
            entry_price=entry_price,
            available_balance=available_balance,
            atr=atr if atr > 0 else None,
            performance_data=performance_data,
        )

        # Apply quality-based adjustment
        if quality_score >= 80:
            size_multiplier = 1.2
        elif quality_score >= 65:
            size_multiplier = 1.0
        elif quality_score >= 55:
            size_multiplier = 0.8
        else:
            size_multiplier = 0.6

        # Apply regime-based adjustment
        if regime_result:
            if regime_result.volatility_percentile > 80:
                size_multiplier *= 0.6
            elif regime_result.volatility_percentile > 60:
                size_multiplier *= 0.8

        # Apply filter adjustments
        if "position_size_multiplier" in adjustments:
            size_multiplier *= adjustments["position_size_multiplier"]

        # Apply enhanced risk adjustments
        if self.enhanced_risk:
            current_drawdown = 0.0  # Would come from account tracking
            volatility_pct = (
                regime_result.volatility_percentile if regime_result else 50.0
            )
            confidence = sizing.risk_reward_ratio / 3  # Normalize to 0-1

            adjusted_size = self.enhanced_risk.get_adjusted_position_size(
                base_size=sizing.size * size_multiplier,
                current_drawdown=current_drawdown,
                current_volatility_percentile=volatility_pct,
                confidence=confidence,
            )
        else:
            adjusted_size = sizing.size * size_multiplier

        # Create new sizing with adjusted size
        return PositionSizing(
            symbol=sizing.symbol,
            side=sizing.side,
            size=adjusted_size,
            entry_price=sizing.entry_price,
            stop_loss_price=sizing.stop_loss_price,
            take_profit_price=sizing.take_profit_price,
            risk_amount=sizing.risk_amount * (adjusted_size / sizing.size)
            if sizing.size > 0
            else 0,
            potential_profit=sizing.potential_profit * (adjusted_size / sizing.size)
            if sizing.size > 0
            else 0,
            risk_reward_ratio=sizing.risk_reward_ratio,
        )

    def _get_quality_grade(self, score: float) -> TradeQuality:
        """Convert quality score to grade."""
        if score >= self.EXCELLENT_QUALITY_THRESHOLD:
            return TradeQuality.EXCELLENT
        elif score >= self.GOOD_QUALITY_THRESHOLD:
            return TradeQuality.GOOD
        elif score >= self.ACCEPTABLE_QUALITY_THRESHOLD:
            return TradeQuality.ACCEPTABLE
        else:
            return TradeQuality.POOR

    def _create_hold_decision(
        self,
        symbol: str,
        current_price: float,
        reason: str,
        signal: Signal,
        confidence: float,
        market_regime: Optional[str] = None,
        trend_strength: Optional[str] = None,
        filter_result: Optional[str] = None,
        quality_score: float = 0.0,
    ) -> EnhancedTradeDecision:
        """Create a HOLD decision with context."""
        return EnhancedTradeDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            signal=signal,
            confidence=confidence,
            quality_score=quality_score,
            quality_grade=TradeQuality.POOR,
            entry_price=current_price,
            market_regime=market_regime,
            trend_strength=trend_strength,
            reason=reason,
            filter_result=filter_result,
        )

    def analyze_all_pairs(
        self,
        available_balance: float,
        current_positions: List[Position],
    ) -> List[EnhancedTradeDecision]:
        """Analyze all configured trading pairs."""
        decisions = []

        for symbol in self.config.trading_pairs:
            decision = self.make_decision(symbol, available_balance, current_positions)
            decisions.append(decision)

        return decisions

    def get_actionable_decisions(
        self,
        decisions: List[EnhancedTradeDecision],
        min_quality: TradeQuality = TradeQuality.ACCEPTABLE,
    ) -> List[EnhancedTradeDecision]:
        """Filter decisions to only high-quality actionable ones."""
        quality_order = [
            TradeQuality.EXCELLENT,
            TradeQuality.GOOD,
            TradeQuality.ACCEPTABLE,
            TradeQuality.POOR,
        ]
        min_quality_idx = quality_order.index(min_quality)

        actionable = []
        for d in decisions:
            if not d.is_actionable:
                continue

            quality_idx = quality_order.index(d.quality_grade)
            if quality_idx <= min_quality_idx:
                actionable.append(d)

        # Sort by quality score descending
        actionable.sort(key=lambda x: x.quality_score, reverse=True)

        return actionable

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and statistics."""
        status = {
            "strict_mode": self.strict_mode,
            "dry_run": self.dry_run,
            "advanced_indicators_enabled": ADVANCED_INDICATORS_AVAILABLE,
            "signal_filter_enabled": SIGNAL_FILTER_AVAILABLE,
            "enhanced_risk_enabled": ENHANCED_RISK_AVAILABLE,
            "quality_thresholds": {
                "excellent": self.EXCELLENT_QUALITY_THRESHOLD,
                "good": self.GOOD_QUALITY_THRESHOLD,
                "acceptable": self.ACCEPTABLE_QUALITY_THRESHOLD,
                "minimum": self.MIN_QUALITY_THRESHOLD,
            },
            "trading_pairs": list(self.config.trading_pairs),
        }

        # Add signal filter stats
        if self.signal_filter:
            status["signal_filter_stats"] = self.signal_filter.get_all_stats()

        # Add enhanced risk stats
        if self.enhanced_risk:
            status["active_risk_positions"] = len(self.enhanced_risk._positions)
            status["consecutive_losses"] = self.enhanced_risk._consecutive_losses
            status["daily_pnl"] = self.enhanced_risk._daily_pnl

        # Add regime cache info
        status["cached_regimes"] = {
            symbol: regime.regime.value
            for symbol, (regime, _) in self._regime_cache.items()
        }

        return status

    def update_filter_with_trade_result(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> None:
        """
        Update signal filter with trade result for adaptive filtering.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss
        """
        if self.signal_filter:
            self.signal_filter.record_trade_result(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
            )
            # Adapt thresholds based on performance
            self.signal_filter.adapt_thresholds()

        if self.enhanced_risk:
            # Find exit reason based on stop/target
            if pnl > 0:
                exit_reason = "take_profit"
            else:
                exit_reason = "stop_loss"

            self.enhanced_risk.record_trade_close(
                symbol=symbol,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )

    def reset_filters(self) -> None:
        """Reset all filter states."""
        if self.signal_filter:
            self.signal_filter.reset_all()
        self._regime_cache.clear()
        log.info("[STRATEGY] All filters and caches reset")

    def get_market_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get market summary for all trading pairs.

        Returns:
            Dictionary with regime and trend info for each symbol
        """
        summary = {}

        for symbol in self.config.trading_pairs:
            analysis = self.analyze_market(symbol, include_advanced=True)

            if analysis:
                regime_result = analysis.get("regime_result")
                adx_result = analysis.get("adx_result")
                ta_result = analysis.get("ta_result")

                summary[symbol] = {
                    "current_price": analysis.get("current_price", 0),
                    "regime": regime_result.regime.value
                    if regime_result
                    else "unknown",
                    "should_trade": regime_result.should_trade
                    if regime_result
                    else False,
                    "trend_strength": adx_result.trend_strength.value
                    if adx_result
                    else "unknown",
                    "trend_direction": adx_result.trend_direction
                    if adx_result
                    else "neutral",
                    "adx": adx_result.adx if adx_result else 0,
                    "signal": ta_result.combined_signal.value if ta_result else "hold",
                    "confidence": ta_result.confidence if ta_result else 0,
                    "volatility_percentile": regime_result.volatility_percentile
                    if regime_result
                    else 50,
                }
            else:
                summary[symbol] = {"error": "Analysis failed"}

        return summary
