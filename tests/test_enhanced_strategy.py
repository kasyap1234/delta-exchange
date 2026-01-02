"""
Test Suite for Enhanced Strategy Components.

Tests for:
- Advanced Indicators (ADX, VWAP, Volume Profile, Market Regime)
- Signal Filter
- Enhanced Risk Management
- Enhanced Trading Strategy v2
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_trending_data(length: int = 100, direction: str = "up", volatility: float = 0.02):
    """Generate synthetic price data with trend."""
    np.random.seed(42)

    if direction == "up":
        trend = np.linspace(100, 120, length)
    elif direction == "down":
        trend = np.linspace(120, 100, length)
    else:
        trend = np.ones(length) * 100

    noise = np.random.normal(0, volatility * 100, length)
    close = trend + noise
    high = close * (1 + np.random.uniform(0.001, 0.01, length))
    low = close * (1 - np.random.uniform(0.001, 0.01, length))
    volume = np.random.uniform(1000, 10000, length)

    return high, low, close, volume


def generate_choppy_data(length: int = 100):
    """Generate synthetic choppy/ranging price data."""
    np.random.seed(43)

    base = 100
    close = base + np.sin(np.linspace(0, 10 * np.pi, length)) * 5
    close += np.random.normal(0, 1, length)
    high = close * 1.005
    low = close * 0.995
    volume = np.random.uniform(1000, 5000, length)

    return high, low, close, volume


# =============================================================================
# Advanced Indicators Tests
# =============================================================================

class TestAdvancedIndicators:
    """Tests for the AdvancedIndicators class."""

    @pytest.fixture
    def indicators(self):
        """Create AdvancedIndicators instance."""
        try:
            from src.advanced_indicators import AdvancedIndicators
            return AdvancedIndicators()
        except ImportError:
            pytest.skip("Advanced indicators module not available")

    def test_adx_trending_market(self, indicators):
        """Test ADX correctly identifies trending market."""
        high, low, close, _ = generate_trending_data(100, "up")

        result = indicators.calculate_adx(high, low, close)

        # Trending market should have higher ADX
        assert result.adx >= 20, f"ADX should be >= 20 for trending market, got {result.adx}"
        assert result.trend_direction == "bullish", f"Expected bullish trend, got {result.trend_direction}"
        assert result.is_trending is True

    def test_adx_choppy_market(self, indicators):
        """Test ADX correctly identifies choppy market."""
        high, low, close, _ = generate_choppy_data(100)

        result = indicators.calculate_adx(high, low, close)

        # Choppy market should have lower ADX
        assert result.adx < 30, f"ADX should be < 30 for choppy market, got {result.adx}"

    def test_vwap_calculation(self, indicators):
        """Test VWAP calculation with bands."""
        high, low, close, volume = generate_trending_data(50, "up")

        result = indicators.calculate_vwap(high, low, close, volume)

        assert result.vwap > 0, "VWAP should be positive"
        assert result.upper_band_1std > result.vwap, "Upper band should be above VWAP"
        assert result.lower_band_1std < result.vwap, "Lower band should be below VWAP"
        assert result.upper_band_2std > result.upper_band_1std, "2std band should be wider"

    def test_vwap_price_position(self, indicators):
        """Test VWAP correctly identifies price position."""
        high, low, close, volume = generate_trending_data(50, "up")

        result = indicators.calculate_vwap(high, low, close, volume)
        current_price = close[-1]

        if current_price > result.vwap:
            assert result.price_vs_vwap == "above"
        elif current_price < result.vwap:
            assert result.price_vs_vwap == "below"

    def test_market_regime_detection_trending(self, indicators):
        """Test market regime correctly identifies uptrend."""
        high, low, close, volume = generate_trending_data(100, "up")

        result = indicators.detect_market_regime(high, low, close, volume, "TEST")

        assert result.regime.value in ["uptrend", "strong_uptrend"], \
            f"Expected uptrend regime, got {result.regime.value}"
        assert result.should_trade is True

    def test_market_regime_detection_choppy(self, indicators):
        """Test market regime correctly identifies choppy conditions."""
        high, low, close, volume = generate_choppy_data(100)

        result = indicators.detect_market_regime(high, low, close, volume, "TEST")

        # Choppy market should recommend avoiding
        assert result.regime.value in ["ranging", "choppy", "low_volatility"], \
            f"Expected ranging/choppy regime, got {result.regime.value}"

    def test_choppiness_index(self, indicators):
        """Test choppiness index calculation."""
        high, low, close, _ = generate_choppy_data(50)

        ci = indicators.calculate_choppiness_index(high, low, close)

        # Choppy data should have higher CI
        assert 0 <= ci <= 100, f"CI should be between 0-100, got {ci}"

    def test_efficiency_ratio(self, indicators):
        """Test Kaufman Efficiency Ratio calculation."""
        # Trending data should have higher ER
        _, _, close_trend, _ = generate_trending_data(50, "up")
        er_trend = indicators.calculate_efficiency_ratio(close_trend)

        # Choppy data should have lower ER
        _, _, close_choppy, _ = generate_choppy_data(50)
        er_choppy = indicators.calculate_efficiency_ratio(close_choppy)

        assert 0 <= er_trend <= 1, "ER should be between 0-1"
        assert 0 <= er_choppy <= 1, "ER should be between 0-1"
        # Trending should have higher efficiency than choppy
        assert er_trend > er_choppy * 0.8, \
            f"Trending ER ({er_trend}) should be > choppy ER ({er_choppy})"

    def test_momentum_suite(self, indicators):
        """Test comprehensive momentum calculation."""
        high, low, close, volume = generate_trending_data(50, "up")

        result = indicators.calculate_momentum_suite(high, low, close, volume)

        # Uptrend should show positive momentum
        assert result.roc > 0, f"ROC should be positive in uptrend, got {result.roc}"
        assert result.momentum > 0, f"Momentum should be positive in uptrend"
        assert 0 <= result.rsi <= 100, "RSI should be between 0-100"
        assert 0 <= result.mfi <= 100, "MFI should be between 0-100"

    def test_entry_quality_score(self, indicators):
        """Test entry quality score calculation."""
        high, low, close, volume = generate_trending_data(100, "up")

        score, breakdown = indicators.get_entry_quality_score(
            high, low, close, volume, "long"
        )

        assert 0 <= score <= 100, f"Score should be between 0-100, got {score}"
        assert "trend" in breakdown
        assert "momentum" in breakdown
        assert "regime" in breakdown


# =============================================================================
# Signal Filter Tests
# =============================================================================

class TestSignalFilter:
    """Tests for the SignalFilter class."""

    @pytest.fixture
    def signal_filter(self):
        """Create SignalFilter instance."""
        try:
            from src.signal_filter import SignalFilter
            return SignalFilter()
        except ImportError:
            pytest.skip("Signal filter module not available")

    def test_confidence_filter_pass(self, signal_filter):
        """Test that high confidence signals pass."""
        from src.signal_filter import FilterResult

        result = signal_filter._filter_confidence(0.7)
        assert result == FilterResult.PASSED

    def test_confidence_filter_fail(self, signal_filter):
        """Test that low confidence signals fail."""
        from src.signal_filter import FilterResult

        result = signal_filter._filter_confidence(0.3)
        assert result == FilterResult.REJECTED_LOW_CONFIDENCE

    def test_trade_frequency_filter(self, signal_filter):
        """Test daily trade limit filtering."""
        from src.signal_filter import FilterResult

        state = signal_filter._get_symbol_state("BTCUSD")

        # First few trades should pass
        state.daily_trades = 2
        result = signal_filter._filter_trade_frequency(state)
        assert result == FilterResult.PASSED

        # Exceeding limit should fail
        state.daily_trades = 10
        result = signal_filter._filter_trade_frequency(state)
        assert result == FilterResult.REJECTED_MAX_TRADES

    def test_recent_loss_filter(self, signal_filter):
        """Test consecutive loss filtering."""
        from src.signal_filter import FilterResult, TradeRecord

        state = signal_filter._get_symbol_state("BTCUSD")

        # Normal state should pass
        state.consecutive_losses = 1
        result = signal_filter._filter_recent_losses(state)
        assert result == FilterResult.PASSED

        # Too many losses should fail
        state.consecutive_losses = 5
        state.recent_trades = [
            TradeRecord(
                symbol="BTCUSD",
                direction="long",
                entry_time=datetime.now() - timedelta(hours=1),
                entry_price=50000,
                exit_time=datetime.now(),
                exit_price=49000,
                pnl=-100,
                is_win=False
            )
        ]
        result = signal_filter._filter_recent_losses(state)
        assert result == FilterResult.REJECTED_RECENT_LOSS

    def test_indicator_agreement_filter(self, signal_filter):
        """Test indicator agreement filtering."""
        from src.signal_filter import FilterResult

        # Strong agreement
        indicators_agree = {
            "RSI": "bullish",
            "MACD": "bullish",
            "BB": "bullish",
            "EMA": "neutral"
        }
        result, pct = signal_filter._filter_indicator_agreement(indicators_agree, "long")
        assert result == FilterResult.PASSED

        # Weak agreement
        indicators_disagree = {
            "RSI": "bullish",
            "MACD": "bearish",
            "BB": "neutral",
            "EMA": "neutral"
        }
        result, pct = signal_filter._filter_indicator_agreement(indicators_disagree, "long")
        assert result == FilterResult.REJECTED_NO_MOMENTUM

    def test_correlation_filter(self, signal_filter):
        """Test correlation exposure filtering."""
        from src.signal_filter import FilterResult

        # No positions - should pass
        result = signal_filter._filter_correlation("BTCUSD", [])
        assert result == FilterResult.PASSED

        # Already holding correlated assets
        result = signal_filter._filter_correlation(
            "ETHUSD",
            ["BTCUSD", "SOLUSD", "AVAXUSD"]
        )
        # May or may not pass depending on correlation config
        assert result in [FilterResult.PASSED, FilterResult.REJECTED_CORRELATION_LIMIT]

    def test_record_trade_result(self, signal_filter):
        """Test trade result recording."""
        signal_filter.record_trade_result(
            symbol="BTCUSD",
            direction="long",
            entry_price=50000,
            exit_price=51000,
            pnl=100
        )

        stats = signal_filter.get_symbol_stats("BTCUSD")
        assert stats["daily_trades"] == 1
        assert stats["daily_pnl"] == 100
        assert stats["consecutive_losses"] == 0

    def test_adaptive_filter_threshold_adjustment(self):
        """Test that adaptive filter adjusts thresholds."""
        try:
            from src.signal_filter import AdaptiveFilter

            adaptive = AdaptiveFilter()

            # Record some losses
            for i in range(10):
                adaptive.record_trade_result("BTCUSD", "long", 50000, 49000, -100)

            # Adapt thresholds
            adaptive.adapt_thresholds()

            # Thresholds should be tightened
            assert adaptive.MIN_QUALITY_SCORE >= 55.0

        except ImportError:
            pytest.skip("Adaptive filter not available")


# =============================================================================
# Enhanced Risk Management Tests
# =============================================================================

class TestEnhancedRiskManager:
    """Tests for the EnhancedRiskManager class."""

    @pytest.fixture
    def risk_manager(self):
        """Create EnhancedRiskManager instance."""
        try:
            from src.enhanced_risk import EnhancedRiskManager
            return EnhancedRiskManager()
        except ImportError:
            pytest.skip("Enhanced risk module not available")

    def test_create_position_risk_state(self, risk_manager):
        """Test creating position risk state."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        assert state.symbol == "BTCUSD"
        assert state.side == "long"
        assert state.entry_price == 50000
        assert state.dynamic_stop.initial_stop < 50000  # Stop below for long
        assert state.take_profit > 50000  # Target above for long

    def test_position_risk_state_short(self, risk_manager):
        """Test creating short position risk state."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="short",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        assert state.dynamic_stop.initial_stop > 50000  # Stop above for short
        assert state.take_profit < 50000  # Target below for short

    def test_update_position_in_profit(self, risk_manager):
        """Test position update when in profit."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        # Price moved up significantly
        updated_state, action = risk_manager.update_position("BTCUSD", 52000, atr=500)

        assert updated_state is not None
        assert updated_state.metrics.unrealized_pnl > 0
        # Should have moved to break-even or trailing
        assert updated_state.dynamic_stop.current_stop >= state.dynamic_stop.initial_stop

    def test_break_even_trigger(self, risk_manager):
        """Test break-even stop trigger."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        initial_stop = state.dynamic_stop.current_stop

        # Move price to 1R profit (should trigger break-even)
        risk_distance = 50000 - initial_stop
        target_price = 50000 + risk_distance  # 1R profit

        updated_state, _ = risk_manager.update_position("BTCUSD", target_price, atr=500)

        # Stop should have moved to at least break-even
        if updated_state.dynamic_stop.is_at_break_even:
            assert updated_state.dynamic_stop.current_stop >= 50000

    def test_trailing_stop_update(self, risk_manager):
        """Test trailing stop updates with price."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        # Simulate price moving up in stages
        prices = [51000, 52000, 53000, 54000]
        previous_stop = state.dynamic_stop.current_stop

        for price in prices:
            updated_state, _ = risk_manager.update_position("BTCUSD", price, atr=500)
            # Stop should never decrease for long
            assert updated_state.dynamic_stop.current_stop >= previous_stop
            previous_stop = updated_state.dynamic_stop.current_stop

    def test_stop_loss_trigger(self, risk_manager):
        """Test stop-loss trigger detection."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        stop_price = state.dynamic_stop.current_stop

        # Price drops below stop
        updated_state, action = risk_manager.update_position("BTCUSD", stop_price - 100)

        assert action == "stop_loss"

    def test_take_profit_trigger(self, risk_manager):
        """Test take-profit trigger detection."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        target_price = state.take_profit

        # Price reaches target
        updated_state, action = risk_manager.update_position("BTCUSD", target_price + 100)

        assert action == "take_profit"

    def test_adjusted_position_size_drawdown(self, risk_manager):
        """Test position size adjustment during drawdown."""
        base_size = 0.01

        # No drawdown
        adjusted = risk_manager.get_adjusted_position_size(
            base_size=base_size,
            current_drawdown=0.0,
            current_volatility_percentile=50.0,
            confidence=0.5
        )
        assert adjusted >= base_size * 0.9

        # Moderate drawdown
        adjusted = risk_manager.get_adjusted_position_size(
            base_size=base_size,
            current_drawdown=0.06,  # 6%
            current_volatility_percentile=50.0,
            confidence=0.5
        )
        assert adjusted < base_size

        # Severe drawdown
        adjusted = risk_manager.get_adjusted_position_size(
            base_size=base_size,
            current_drawdown=0.12,  # 12%
            current_volatility_percentile=50.0,
            confidence=0.5
        )
        assert adjusted == 0  # Should block trading

    def test_adjusted_position_size_volatility(self, risk_manager):
        """Test position size adjustment for high volatility."""
        base_size = 0.01

        # Normal volatility
        normal = risk_manager.get_adjusted_position_size(
            base_size=base_size,
            current_drawdown=0.0,
            current_volatility_percentile=50.0,
            confidence=0.5
        )

        # High volatility
        high_vol = risk_manager.get_adjusted_position_size(
            base_size=base_size,
            current_drawdown=0.0,
            current_volatility_percentile=85.0,
            confidence=0.5
        )

        assert high_vol < normal

    def test_dynamic_stop_calculation(self, risk_manager):
        """Test dynamic stop calculation with different conditions."""
        entry_price = 50000
        atr = 500

        # Normal conditions
        sl_normal, tp_normal = risk_manager.calculate_dynamic_stop(
            entry_price=entry_price,
            side="long",
            atr=atr,
            volatility_regime="normal",
            trend_strength=0.5
        )

        # High volatility
        sl_high_vol, tp_high_vol = risk_manager.calculate_dynamic_stop(
            entry_price=entry_price,
            side="long",
            atr=atr,
            volatility_regime="high",
            trend_strength=0.5
        )

        # High volatility should have wider stops
        assert abs(entry_price - sl_high_vol) > abs(entry_price - sl_normal)

    def test_scale_out_logic(self, risk_manager):
        """Test scale-out logic at profit targets."""
        state = risk_manager.create_position_risk_state(
            symbol="BTCUSD",
            side="long",
            entry_price=50000,
            position_size=0.01,
            atr=500
        )

        # At entry - shouldn't scale
        should_scale, portion = risk_manager.should_scale_out(state, 50000)
        assert should_scale is False

        # At 2R profit - should scale
        risk_distance = 50000 - state.dynamic_stop.initial_stop
        price_at_2r = 50000 + (2 * risk_distance)
        should_scale, portion = risk_manager.should_scale_out(state, price_at_2r)
        assert should_scale is True
        assert portion > 0


# =============================================================================
# Enhanced Strategy Tests
# =============================================================================

class TestEnhancedTradingStrategy:
    """Tests for the EnhancedTradingStrategy class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Delta Exchange client."""
        client = MagicMock()

        # Mock candle data
        high, low, close, volume = generate_trending_data(100, "up")

        mock_candles = []
        for i in range(len(close)):
            candle = MagicMock()
            candle.high = high[i]
            candle.low = low[i]
            candle.close = close[i]
            candle.volume = volume[i]
            mock_candles.append(candle)

        client.get_candles.return_value = mock_candles
        client.get_ticker.return_value = {"mark_price": close[-1]}

        return client

    @pytest.fixture
    def strategy(self, mock_client):
        """Create EnhancedTradingStrategy instance."""
        try:
            from src.strategy_v2 import EnhancedTradingStrategy
            return EnhancedTradingStrategy(
                client=mock_client,
                dry_run=True,
                strict_mode=True
            )
        except ImportError:
            pytest.skip("Enhanced strategy module not available")

    def test_market_analysis(self, strategy):
        """Test comprehensive market analysis."""
        analysis = strategy.analyze_market("BTCUSD")

        assert analysis is not None
        assert "current_price" in analysis
        assert "ta_result" in analysis
        assert "close" in analysis

    def test_market_analysis_with_advanced(self, strategy):
        """Test market analysis includes advanced indicators."""
        analysis = strategy.analyze_market("BTCUSD", include_advanced=True)

        assert analysis is not None
        if strategy.advanced:
            assert "atr" in analysis
            assert "adx_result" in analysis
            assert "regime_result" in analysis

    def test_make_decision_no_position(self, strategy):
        """Test decision making without existing position."""
        from src.strategy_v2 import TradeAction

        decision = strategy.make_decision(
            symbol="BTCUSD",
            available_balance=10000,
            current_positions=[]
        )

        assert decision is not None
        assert decision.symbol == "BTCUSD"
        assert decision.action in [
            TradeAction.OPEN_LONG,
            TradeAction.OPEN_SHORT,
            TradeAction.HOLD
        ]

    def test_make_decision_with_position(self, strategy):
        """Test decision making with existing position."""
        from src.strategy_v2 import TradeAction

        # Mock position
        position = MagicMock()
        position.product_symbol = "BTCUSD"
        position.size = 0.01
        position.entry_price = 50000

        decision = strategy.make_decision(
            symbol="BTCUSD",
            available_balance=10000,
            current_positions=[position]
        )

        assert decision is not None
        assert decision.action in [
            TradeAction.CLOSE_LONG,
            TradeAction.CLOSE_SHORT,
            TradeAction.SCALE_OUT,
            TradeAction.HOLD
        ]

    def test_quality_grade_assignment(self, strategy):
        """Test quality grade assignment based on score."""
        from src.strategy_v2 import TradeQuality

        assert strategy._get_quality_grade(85) == TradeQuality.EXCELLENT
        assert strategy._get_quality_grade(70) == TradeQuality.GOOD
        assert strategy._get_quality_grade(58) == TradeQuality.ACCEPTABLE
        assert strategy._get_quality_grade(40) == TradeQuality.POOR

    def test_actionable_decisions_filter(self, strategy):
        """Test filtering for actionable decisions."""
        from src.strategy_v2 import (
            EnhancedTradeDecision,
            TradeAction,
            TradeQuality,
            Signal
        )

        decisions = [
            EnhancedTradeDecision(
                symbol="BTCUSD",
                action=TradeAction.OPEN_LONG,
                signal=Signal.BUY,
                confidence=0.8,
                quality_score=75,
                quality_grade=TradeQuality.GOOD,
                entry_price=50000
            ),
            EnhancedTradeDecision(
                symbol="ETHUSD",
                action=TradeAction.HOLD,
                signal=Signal.HOLD,
                confidence=0.3,
                quality_score=40,
                quality_grade=TradeQuality.POOR,
                entry_price=3000
            ),
            EnhancedTradeDecision(
                symbol="SOLUSD",
                action=TradeAction.OPEN_SHORT,
                signal=Signal.SELL,
                confidence=0.6,
                quality_score=60,
                quality_grade=TradeQuality.ACCEPTABLE,
                entry_price=100
            )
        ]

        # Should only return actionable decisions above quality threshold
        actionable = strategy.get_actionable_decisions(
            decisions,
            min_quality=TradeQuality.ACCEPTABLE
        )

        assert len(actionable) == 2
        assert all(d.is_actionable for d in actionable)
        # Should be sorted by quality score
        assert actionable[0].quality_score >= actionable[1].quality_score

    def test_strategy_status(self, strategy):
        """Test getting strategy status."""
        status = strategy.get_strategy_status()

        assert "strict_mode" in status
        assert "dry_run" in status
        assert "trading_pairs" in status
        assert "quality_thresholds" in status

    def test_market_summary(self, strategy):
        """Test market summary generation."""
        summary = strategy.get_market_summary()

        assert isinstance(summary, dict)
        # Should have entries for configured trading pairs
        assert len(summary) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the enhanced trading system."""

    def test_full_signal_flow(self):
        """Test complete signal flow from analysis to decision."""
        try:
            from src.advanced_indicators import AdvancedIndicators
            from src.signal_filter import SignalFilter
            from src.enhanced_risk import EnhancedRiskManager

            # Generate test data
            high, low, close, volume = generate_trending_data(100, "up")

            # 1. Advanced analysis
            indicators = AdvancedIndicators()
            adx = indicators.calculate_adx(high, low, close)
            regime = indicators.detect_market_regime(high, low, close, volume, "TEST")
            quality_score, _ = indicators.get_entry_quality_score(
                high, low, close, volume, "long"
            )

            # 2. Signal filtering
            signal_filter = SignalFilter()
            # (Would normally use actual signal here)

            # 3. Risk management
            risk_manager = EnhancedRiskManager()

            if regime.should_trade and quality_score >= 55:
                state = risk_manager.create_position_risk_state(
                    symbol="BTCUSD",
                    side="long",
                    entry_price=close[-1],
                    position_size=0.01,
                    atr=indicators._calculate_atr(high, low, close)
                )

                assert state is not None
                assert state.dynamic_stop.current_stop < close[-1]

        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    def test_position_lifecycle(self):
        """Test complete position lifecycle from open to close."""
        try:
            from src.enhanced_risk import EnhancedRiskManager

            risk_manager = EnhancedRiskManager()

            # 1. Open position
            state = risk_manager.create_position_risk_state(
                symbol="BTCUSD",
                side="long",
                entry_price=50000,
                position_size=0.01,
                atr=500
            )

            # 2. Price moves favorably
            state, action = risk_manager.update_position("BTCUSD", 52000, atr=500)
            assert action is None  # No exit yet

            # 3. Price continues up
            state, action = risk_manager.update_position("BTCUSD", 54000, atr=500)
            # Stop should have trailed up
            assert state.dynamic_stop.current_stop > state.dynamic_stop.initial_stop

            #
