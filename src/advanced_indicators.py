"""
Advanced Technical Indicators Module.

Provides sophisticated market analysis tools beyond basic indicators:
- ADX (Average Directional Index) for trend strength
- VWAP (Volume Weighted Average Price) for institutional levels
- Volume Profile for support/resistance detection
- Market Regime Detection (trending vs ranging vs volatile)
- Momentum indicators (ROC, Momentum, Williams %R)
- Volatility regime analysis
- Order flow imbalance estimation
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import log

# Try to import TA-Lib, fall back to manual calculations
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    log.warning(
        "TA-Lib not available, using fallback implementations for advanced indicators"
    )


class MarketRegime(str, Enum):
    """Market regime classification."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CHOPPY = "choppy"


class TrendStrength(str, Enum):
    """Trend strength classification."""

    ABSENT = "absent"  # ADX < 20
    WEAK = "weak"  # ADX 20-25
    MODERATE = "moderate"  # ADX 25-40
    STRONG = "strong"  # ADX 40-60
    VERY_STRONG = "very_strong"  # ADX > 60


@dataclass
class ADXResult:
    """Result of ADX calculation."""

    adx: float
    plus_di: float
    minus_di: float
    trend_strength: TrendStrength
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    is_trending: bool
    crossover_signal: Optional[str]  # '+DI_cross_up', '-DI_cross_up', None

    @property
    def should_trade_trend(self) -> bool:
        """Check if market conditions favor trend trading."""
        return self.adx >= 25 and self.trend_strength in [
            TrendStrength.MODERATE,
            TrendStrength.STRONG,
            TrendStrength.VERY_STRONG,
        ]

    @property
    def should_avoid_trading(self) -> bool:
        """Check if conditions suggest avoiding new trades."""
        return self.adx < 20 or self.trend_strength == TrendStrength.ABSENT


@dataclass
class VWAPResult:
    """Result of VWAP calculation."""

    vwap: float
    upper_band_1std: float
    upper_band_2std: float
    lower_band_1std: float
    lower_band_2std: float
    price_vs_vwap: str  # 'above', 'below', 'at'
    deviation_pct: float
    is_extended: bool  # Price > 2 std from VWAP

    def get_entry_bias(self, current_price: float) -> str:
        """
        Get entry bias based on VWAP.

        Institutional traders often use VWAP as fair value:
        - Price below VWAP = potential long opportunity
        - Price above VWAP = potential short opportunity (mean reversion)
        """
        if current_price < self.lower_band_1std:
            return "long"  # Oversold vs VWAP
        elif current_price > self.upper_band_1std:
            return "short"  # Overbought vs VWAP
        return "neutral"


@dataclass
class VolumeProfileLevel:
    """Single level in volume profile."""

    price: float
    volume: float
    is_poc: bool  # Point of Control (highest volume)
    is_hvn: bool  # High Volume Node
    is_lvn: bool  # Low Volume Node


@dataclass
class VolumeProfileResult:
    """Result of Volume Profile analysis."""

    poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    hvn_levels: List[float]  # High Volume Nodes
    lvn_levels: List[float]  # Low Volume Nodes
    current_price_in_value_area: bool
    nearest_support: float
    nearest_resistance: float

    def get_trade_bias(self, current_price: float) -> str:
        """Get trading bias based on volume profile."""
        if current_price < self.value_area_low:
            return "long"  # Below value area, likely to revert
        elif current_price > self.value_area_high:
            return "short"  # Above value area, likely to revert
        return "neutral"  # In value area


@dataclass
class MarketRegimeResult:
    """Result of market regime analysis."""

    regime: MarketRegime
    trend_strength: TrendStrength
    volatility_percentile: float  # 0-100
    choppiness_index: float
    efficiency_ratio: float
    should_trade: bool
    preferred_strategy: str  # 'trend_following', 'mean_reversion', 'avoid'
    confidence: float

    @property
    def is_favorable(self) -> bool:
        """Check if market conditions are favorable for trading."""
        unfavorable_regimes = [MarketRegime.CHOPPY, MarketRegime.HIGH_VOLATILITY]
        return self.regime not in unfavorable_regimes and self.should_trade


@dataclass
class MomentumResult:
    """Combined momentum indicators result."""

    roc: float  # Rate of Change
    momentum: float
    williams_r: float
    stoch_k: float
    stoch_d: float
    rsi: float
    mfi: float  # Money Flow Index (volume-weighted RSI)

    @property
    def is_overbought(self) -> bool:
        """Check if multiple momentum indicators show overbought."""
        overbought_count = sum(
            [self.rsi > 70, self.williams_r > -20, self.stoch_k > 80, self.mfi > 80]
        )
        return overbought_count >= 3

    @property
    def is_oversold(self) -> bool:
        """Check if multiple momentum indicators show oversold."""
        oversold_count = sum(
            [self.rsi < 30, self.williams_r < -80, self.stoch_k < 20, self.mfi < 20]
        )
        return oversold_count >= 3

    @property
    def momentum_direction(self) -> str:
        """Get overall momentum direction."""
        bullish = sum(
            [self.roc > 0, self.momentum > 0, self.williams_r > -50, self.stoch_k > 50]
        )
        if bullish >= 3:
            return "bullish"
        elif bullish <= 1:
            return "bearish"
        return "neutral"


class AdvancedIndicators:
    """
    Advanced technical indicators for enhanced signal generation.

    Features:
    - ADX for trend strength (avoid trading in choppy markets)
    - VWAP for institutional price levels
    - Volume Profile for S/R detection
    - Market Regime detection (trending/ranging/volatile)
    - Multi-indicator momentum confirmation
    """

    # ADX thresholds
    ADX_ABSENT = 20
    ADX_WEAK = 25
    ADX_MODERATE = 40
    ADX_STRONG = 60

    # Volatility percentile thresholds
    LOW_VOLATILITY_PERCENTILE = 20
    HIGH_VOLATILITY_PERCENTILE = 80

    # Choppiness Index thresholds
    CHOPPY_THRESHOLD = 61.8
    TRENDING_THRESHOLD = 38.2

    def __init__(self):
        """Initialize advanced indicators calculator."""
        self._atr_history: Dict[str, List[float]] = {}
        self._regime_history: Dict[str, List[MarketRegime]] = {}

    # =========================================================================
    # ADX (Average Directional Index)
    # =========================================================================

    def calculate_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> ADXResult:
        """
        Calculate ADX (Average Directional Index).

        ADX measures trend strength:
        - ADX < 20: Absent or weak trend (avoid trend trading)
        - ADX 20-25: Weak trend (cautious)
        - ADX 25-40: Moderate trend (good for trend trading)
        - ADX 40-60: Strong trend (optimal for trend trading)
        - ADX > 60: Very strong trend (may be exhausted)

        +DI and -DI indicate trend direction.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ADX period (default 14)

        Returns:
            ADXResult with trend strength and direction
        """
        if len(close) < period + 1:
            return ADXResult(
                adx=0.0,
                plus_di=0.0,
                minus_di=0.0,
                trend_strength=TrendStrength.ABSENT,
                trend_direction="neutral",
                is_trending=False,
                crossover_signal=None,
            )

        if TALIB_AVAILABLE:
            adx = talib.ADX(high, low, close, timeperiod=period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)

            current_adx = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            current_plus_di = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0.0
            current_minus_di = (
                float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0.0
            )

            # Check for crossover
            crossover = None
            if len(plus_di) >= 2 and len(minus_di) >= 2:
                prev_plus_di = float(plus_di[-2]) if not np.isnan(plus_di[-2]) else 0.0
                prev_minus_di = (
                    float(minus_di[-2]) if not np.isnan(minus_di[-2]) else 0.0
                )

                if prev_plus_di < prev_minus_di and current_plus_di > current_minus_di:
                    crossover = "+DI_cross_up"  # Bullish crossover
                elif (
                    prev_plus_di > prev_minus_di and current_plus_di < current_minus_di
                ):
                    crossover = "-DI_cross_up"  # Bearish crossover
        else:
            current_adx, current_plus_di, current_minus_di, crossover = (
                self._fallback_adx(high, low, close, period)
            )

        # Determine trend strength
        if current_adx < self.ADX_ABSENT:
            strength = TrendStrength.ABSENT
        elif current_adx < self.ADX_WEAK:
            strength = TrendStrength.WEAK
        elif current_adx < self.ADX_MODERATE:
            strength = TrendStrength.MODERATE
        elif current_adx < self.ADX_STRONG:
            strength = TrendStrength.STRONG
        else:
            strength = TrendStrength.VERY_STRONG

        # Determine trend direction
        if current_plus_di > current_minus_di + 5:  # +5 buffer for significance
            direction = "bullish"
        elif current_minus_di > current_plus_di + 5:
            direction = "bearish"
        else:
            direction = "neutral"

        is_trending = current_adx >= 25

        return ADXResult(
            adx=current_adx,
            plus_di=current_plus_di,
            minus_di=current_minus_di,
            trend_strength=strength,
            trend_direction=direction,
            is_trending=is_trending,
            crossover_signal=crossover,
        )

    def _fallback_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> Tuple[float, float, float, Optional[str]]:
        """Fallback ADX calculation without TA-Lib."""
        n = len(close)

        # Calculate True Range and Directional Movement
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth using Wilder's method
        atr = np.zeros(n)
        smooth_plus_dm = np.zeros(n)
        smooth_minus_dm = np.zeros(n)

        # Initial values
        atr[period] = np.mean(tr[1 : period + 1])
        smooth_plus_dm[period] = np.mean(plus_dm[1 : period + 1])
        smooth_minus_dm[period] = np.mean(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            smooth_plus_dm[i] = (
                smooth_plus_dm[i - 1] * (period - 1) + plus_dm[i]
            ) / period
            smooth_minus_dm[i] = (
                smooth_minus_dm[i - 1] * (period - 1) + minus_dm[i]
            ) / period

        # Calculate DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]

            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # Calculate ADX
        adx = np.zeros(n)
        adx[2 * period] = np.mean(dx[period : 2 * period + 1])

        for i in range(2 * period + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        # Check for crossover
        crossover = None
        if n >= 2:
            if plus_di[-2] < minus_di[-2] and plus_di[-1] > minus_di[-1]:
                crossover = "+DI_cross_up"
            elif plus_di[-2] > minus_di[-2] and plus_di[-1] < minus_di[-1]:
                crossover = "-DI_cross_up"

        return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1]), crossover

    # =========================================================================
    # VWAP (Volume Weighted Average Price)
    # =========================================================================

    def calculate_vwap(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        session_bars: Optional[int] = None,
    ) -> VWAPResult:
        """
        Calculate VWAP (Volume Weighted Average Price) with standard deviation bands.

        VWAP is used by institutions as a benchmark for fair value:
        - Price below VWAP: Asset is undervalued vs average traded price
        - Price above VWAP: Asset is overvalued vs average traded price

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
            session_bars: Number of bars in trading session (None = use all)

        Returns:
            VWAPResult with bands and deviation
        """
        if len(close) == 0 or len(volume) == 0:
            return VWAPResult(
                vwap=0.0,
                upper_band_1std=0.0,
                upper_band_2std=0.0,
                lower_band_1std=0.0,
                lower_band_2std=0.0,
                price_vs_vwap="at",
                deviation_pct=0.0,
                is_extended=False,
            )

        # Use only session bars if specified
        if session_bars and len(close) > session_bars:
            high = high[-session_bars:]
            low = low[-session_bars:]
            close = close[-session_bars:]
            volume = volume[-session_bars:]

        # Typical price
        typical_price = (high + low + close) / 3

        # VWAP calculation
        cumulative_tp_volume = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)

        # Avoid division by zero
        vwap = np.where(
            cumulative_volume > 0,
            cumulative_tp_volume / cumulative_volume,
            typical_price,
        )

        current_vwap = float(vwap[-1])
        current_price = float(close[-1])

        # Calculate standard deviation bands
        squared_dev = (typical_price - vwap) ** 2
        cumulative_sq_dev = np.cumsum(squared_dev * volume)
        variance = np.where(
            cumulative_volume > 0, cumulative_sq_dev / cumulative_volume, 0
        )
        std_dev = np.sqrt(variance)

        current_std = float(std_dev[-1]) if not np.isnan(std_dev[-1]) else 0.0

        # Calculate bands
        upper_1std = current_vwap + current_std
        upper_2std = current_vwap + (2 * current_std)
        lower_1std = current_vwap - current_std
        lower_2std = current_vwap - (2 * current_std)

        # Determine price position
        if current_price > current_vwap * 1.001:
            price_position = "above"
        elif current_price < current_vwap * 0.999:
            price_position = "below"
        else:
            price_position = "at"

        # Calculate deviation percentage
        deviation_pct = (
            ((current_price - current_vwap) / current_vwap * 100)
            if current_vwap > 0
            else 0.0
        )

        # Check if price is extended (> 2 std)
        is_extended = abs(current_price - current_vwap) > (2 * current_std)

        return VWAPResult(
            vwap=current_vwap,
            upper_band_1std=upper_1std,
            upper_band_2std=upper_2std,
            lower_band_1std=lower_1std,
            lower_band_2std=lower_2std,
            price_vs_vwap=price_position,
            deviation_pct=deviation_pct,
            is_extended=is_extended,
        )

    # =========================================================================
    # Volume Profile
    # =========================================================================

    def calculate_volume_profile(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        num_levels: int = 24,
        value_area_pct: float = 0.70,
    ) -> VolumeProfileResult:
        """
        Calculate Volume Profile with POC and Value Area.

        Volume Profile shows where most trading occurred:
        - POC (Point of Control): Price level with highest volume
        - Value Area: Range containing 70% of volume
        - HVN (High Volume Nodes): Strong support/resistance
        - LVN (Low Volume Nodes): Price often moves quickly through

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
            num_levels: Number of price levels to analyze
            value_area_pct: Percentage of volume for value area (default 70%)

        Returns:
            VolumeProfileResult with POC, Value Area, and key levels
        """
        if len(close) == 0 or len(volume) == 0:
            return VolumeProfileResult(
                poc=0.0,
                value_area_high=0.0,
                value_area_low=0.0,
                hvn_levels=[],
                lvn_levels=[],
                current_price_in_value_area=False,
                nearest_support=0.0,
                nearest_resistance=0.0,
            )

        current_price = float(close[-1])
        price_min = float(np.min(low))
        price_max = float(np.max(high))
        price_range = price_max - price_min

        if price_range <= 0:
            return VolumeProfileResult(
                poc=current_price,
                value_area_high=current_price,
                value_area_low=current_price,
                hvn_levels=[current_price],
                lvn_levels=[],
                current_price_in_value_area=True,
                nearest_support=current_price,
                nearest_resistance=current_price,
            )

        level_height = price_range / num_levels

        # Distribute volume to price levels
        level_volumes = np.zeros(num_levels)
        level_prices = np.array(
            [price_min + (i + 0.5) * level_height for i in range(num_levels)]
        )

        for i in range(len(close)):
            # Find which level this bar's typical price falls into
            typical = (high[i] + low[i] + close[i]) / 3
            level_idx = int((typical - price_min) / level_height)
            level_idx = min(max(level_idx, 0), num_levels - 1)
            level_volumes[level_idx] += volume[i]

        # Find POC (highest volume level)
        poc_idx = int(np.argmax(level_volumes))
        poc = float(level_prices[poc_idx])

        # Calculate Value Area
        total_volume = np.sum(level_volumes)
        target_volume = total_volume * value_area_pct

        # Start from POC and expand outward
        included_volume = level_volumes[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while included_volume < target_volume:
            # Expand to higher volume side
            vol_below = level_volumes[va_low_idx - 1] if va_low_idx > 0 else 0
            vol_above = (
                level_volumes[va_high_idx + 1] if va_high_idx < num_levels - 1 else 0
            )

            if vol_below == 0 and vol_above == 0:
                break

            if vol_above >= vol_below and va_high_idx < num_levels - 1:
                va_high_idx += 1
                included_volume += level_volumes[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                included_volume += level_volumes[va_low_idx]
            else:
                break

        value_area_high = float(level_prices[va_high_idx] + level_height / 2)
        value_area_low = float(level_prices[va_low_idx] - level_height / 2)

        # Find HVN and LVN levels
        avg_volume = np.mean(level_volumes)
        std_volume = np.std(level_volumes)

        hvn_levels = []
        lvn_levels = []

        for i, (price, vol) in enumerate(zip(level_prices, level_volumes)):
            if vol > avg_volume + std_volume:
                hvn_levels.append(float(price))
            elif vol < avg_volume - std_volume * 0.5 and vol > 0:
                lvn_levels.append(float(price))

        # Check if current price is in value area
        in_value_area = value_area_low <= current_price <= value_area_high

        # Find nearest support and resistance
        support_candidates = [p for p in hvn_levels if p < current_price]
        resistance_candidates = [p for p in hvn_levels if p > current_price]

        nearest_support = (
            max(support_candidates) if support_candidates else value_area_low
        )
        nearest_resistance = (
            min(resistance_candidates) if resistance_candidates else value_area_high
        )

        return VolumeProfileResult(
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            hvn_levels=hvn_levels,
            lvn_levels=lvn_levels,
            current_price_in_value_area=in_value_area,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

    # =========================================================================
    # Market Regime Detection
    # =========================================================================

    def calculate_choppiness_index(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """
        Calculate Choppiness Index.

        Choppiness Index measures if market is trending or choppy:
        - CI > 61.8: Market is choppy/consolidating (avoid trend strategies)
        - CI < 38.2: Market is trending (favor trend strategies)
        - 38.2 - 61.8: Transitional

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: Lookback period

        Returns:
            Choppiness Index value (0-100)
        """
        if len(close) < period + 1:
            return 50.0  # Neutral

        # Calculate ATR sum
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr_sum = np.sum(tr[-period:])

        # Calculate highest high - lowest low over period
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        hl_diff = highest_high - lowest_low

        if hl_diff <= 0:
            return 50.0

        # Choppiness Index formula
        ci = 100 * np.log10(atr_sum / hl_diff) / np.log10(period)

        return float(np.clip(ci, 0, 100))

    def calculate_efficiency_ratio(self, close: np.ndarray, period: int = 10) -> float:
        """
        Calculate Kaufman Efficiency Ratio.

        Measures how efficiently price moves:
        - ER close to 1: Trending market (directional movement)
        - ER close to 0: Choppy market (lots of back and forth)

        Args:
            close: Array of close prices
            period: Lookback period

        Returns:
            Efficiency Ratio (0-1)
        """
        if len(close) < period + 1:
            return 0.5  # Neutral

        # Direction = |Close - Close[period]|
        direction = abs(close[-1] - close[-period - 1])

        # Volatility = sum of |Close - Close[1]|
        volatility = np.sum(np.abs(np.diff(close[-period - 1 :])))

        if volatility == 0:
            return 0.0

        er = direction / volatility

        return float(np.clip(er, 0, 1))

    def detect_market_regime(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> MarketRegimeResult:
        """
        Comprehensive market regime detection.

        Combines multiple indicators to determine current market state:
        - Trend strength (ADX)
        - Trend direction (EMA alignment)
        - Volatility regime (ATR percentile)
        - Choppiness (Choppiness Index)
        - Efficiency (Kaufman ER)

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
            symbol: Symbol for logging

        Returns:
            MarketRegimeResult with regime classification and trading recommendation
        """
        if len(close) < 50:
            return MarketRegimeResult(
                regime=MarketRegime.RANGING,
                trend_strength=TrendStrength.ABSENT,
                volatility_percentile=50.0,
                choppiness_index=50.0,
                efficiency_ratio=0.5,
                should_trade=False,
                preferred_strategy="avoid",
                confidence=0.0,
            )

        # Calculate ADX for trend strength
        adx_result = self.calculate_adx(high, low, close)

        # Calculate volatility percentile
        atr = self._calculate_atr(high, low, close)
        vol_percentile = self._get_volatility_percentile(symbol, atr)

        # Calculate choppiness index
        ci = self.calculate_choppiness_index(high, low, close)

        # Calculate efficiency ratio
        er = self.calculate_efficiency_ratio(close)

        # Determine regime
        regime = self._classify_regime(
            adx_result.adx, adx_result.trend_direction, vol_percentile, ci, er
        )

        # Determine if should trade
        should_trade, preferred_strategy = self._get_trading_recommendation(
            regime, adx_result.trend_strength, vol_percentile, ci
        )

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            adx_result.adx, ci, er, vol_percentile
        )

        # Store regime for history
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []
        self._regime_history[symbol].append(regime)
        if len(self._regime_history[symbol]) > 100:
            self._regime_history[symbol] = self._regime_history[symbol][-100:]

        return MarketRegimeResult(
            regime=regime,
            trend_strength=adx_result.trend_strength,
            volatility_percentile=vol_percentile,
            choppiness_index=ci,
            efficiency_ratio=er,
            should_trade=should_trade,
            preferred_strategy=preferred_strategy,
            confidence=confidence,
        )

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Calculate ATR for volatility measurement."""
        if len(close) < period + 1:
            return 0.0

        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[1 : period + 1])

        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return float(atr[-1])

    def _get_volatility_percentile(self, symbol: str, current_atr: float) -> float:
        """Calculate percentile of current ATR vs historical."""
        if symbol not in self._atr_history:
            self._atr_history[symbol] = []

        self._atr_history[symbol].append(current_atr)

        # Keep last 100 ATR values
        if len(self._atr_history[symbol]) > 100:
            self._atr_history[symbol] = self._atr_history[symbol][-100:]

        if len(self._atr_history[symbol]) < 10:
            return 50.0  # Not enough history

        # Calculate percentile
        history = np.array(self._atr_history[symbol])
        percentile = (np.sum(history < current_atr) / len(history)) * 100

        return float(percentile)

    def _classify_regime(
        self,
        adx: float,
        trend_direction: str,
        vol_percentile: float,
        choppiness: float,
        efficiency: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators."""
        # High volatility takes precedence
        if vol_percentile > self.HIGH_VOLATILITY_PERCENTILE:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility
        if vol_percentile < self.LOW_VOLATILITY_PERCENTILE:
            return MarketRegime.LOW_VOLATILITY

        # Choppy market
        if choppiness > self.CHOPPY_THRESHOLD and efficiency < 0.3:
            return MarketRegime.CHOPPY

        # Strong trend classification
        if adx >= 40 and efficiency > 0.5:
            if trend_direction == "bullish":
                return MarketRegime.STRONG_UPTREND
            elif trend_direction == "bearish":
                return MarketRegime.STRONG_DOWNTREND

        # Moderate trend
        if adx >= 25 and choppiness < self.CHOPPY_THRESHOLD:
            if trend_direction == "bullish":
                return MarketRegime.UPTREND
            elif trend_direction == "bearish":
                return MarketRegime.DOWNTREND

        # Default to ranging
        return MarketRegime.RANGING

    def _get_trading_recommendation(
        self,
        regime: MarketRegime,
        trend_strength: TrendStrength,
        vol_percentile: float,
        choppiness: float,
    ) -> Tuple[bool, str]:
        """Get trading recommendation based on regime."""
        # Avoid trading in these conditions
        if regime in [MarketRegime.CHOPPY, MarketRegime.HIGH_VOLATILITY]:
            return False, "avoid"

        if trend_strength == TrendStrength.ABSENT:
            return False, "avoid"

        # Trend following for trending markets
        if regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.UPTREND,
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.DOWNTREND,
        ]:
            return True, "trend_following"

        # Mean reversion for ranging/low volatility
        if regime in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]:
            if choppiness < 55:  # Not too choppy
                return True, "mean_reversion"

        return False, "avoid"

    def _calculate_regime_confidence(
        self, adx: float, choppiness: float, efficiency: float, vol_percentile: float
    ) -> float:
        """Calculate confidence in regime classification."""
        confidence = 0.5  # Base confidence

        # ADX contribution (strong trend = high confidence)
        if adx > 40:
            confidence += 0.2
        elif adx > 25:
            confidence += 0.1
        elif adx < 15:
            confidence -= 0.1

        # Choppiness contribution
        if choppiness > 70 or choppiness < 30:
            confidence += 0.1  # Clear signal
        elif 45 < choppiness < 55:
            confidence -= 0.1  # Unclear

        # Efficiency contribution
        if efficiency > 0.6 or efficiency < 0.2:
            confidence += 0.1  # Clear trending or ranging
        elif 0.35 < efficiency < 0.45:
            confidence -= 0.1  # Unclear

        # Volatility contribution
        if vol_percentile > 80 or vol_percentile < 20:
            confidence += 0.1  # Clear regime

        return float(np.clip(confidence, 0.0, 1.0))

    # =========================================================================
    # Momentum Indicators
    # =========================================================================

    def calculate_momentum_suite(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> MomentumResult:
        """
        Calculate comprehensive momentum indicators.

        Provides multiple momentum measures for confirmation:
        - ROC (Rate of Change)
        - Raw Momentum
        - Williams %R
        - Stochastic K/D
        - RSI
        - MFI (Money Flow Index)

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume

        Returns:
            MomentumResult with all indicators
        """
        n = len(close)
        if n < 20:
            return MomentumResult(
                roc=0.0,
                momentum=0.0,
                williams_r=-50.0,
                stoch_k=50.0,
                stoch_d=50.0,
                rsi=50.0,
                mfi=50.0,
            )

        # Rate of Change (12 period)
        roc_period = min(12, n - 1)
        roc = ((close[-1] - close[-roc_period - 1]) / close[-roc_period - 1]) * 100

        # Raw Momentum
        momentum = close[-1] - close[-roc_period - 1]

        # Williams %R (14 period)
        period = min(14, n)
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        williams_r = (
            -100 * (highest_high - close[-1]) / max(highest_high - lowest_low, 0.0001)
        )

        # Stochastic K/D
        stoch_k = (
            100 * (close[-1] - lowest_low) / max(highest_high - lowest_low, 0.0001)
        )
        # Smooth K to get D (3 period SMA)
        stoch_d = stoch_k  # Simplified

        # RSI (14 period)
        if TALIB_AVAILABLE:
            rsi_arr = talib.RSI(close, timeperiod=14)
            rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
        else:
            rsi = self._calculate_rsi(close, 14)

        # MFI (Money Flow Index) - volume-weighted RSI
        if TALIB_AVAILABLE:
            mfi_arr = talib.MFI(high, low, close, volume, timeperiod=14)
            mfi = float(mfi_arr[-1]) if not np.isnan(mfi_arr[-1]) else 50.0
        else:
            mfi = self._calculate_mfi(high, low, close, volume, 14)

        return MomentumResult(
            roc=float(roc),
            momentum=float(momentum),
            williams_r=float(williams_r),
            stoch_k=float(stoch_k),
            stoch_d=float(stoch_d),
            rsi=float(rsi),
            mfi=float(mfi),
        )

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Fallback RSI calculation."""
        if len(close) < period + 1:
            return 50.0

        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_mfi(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Money Flow Index (volume-weighted RSI)."""
        if len(close) < period + 1:
            return 50.0

        # Typical price
        typical_price = (high + low + close) / 3

        # Raw money flow
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        positive_flow = np.zeros(len(close))
        negative_flow = np.zeros(len(close))

        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow[i] = raw_money_flow[i]
            elif typical_price[i] < typical_price[i - 1]:
                negative_flow[i] = raw_money_flow[i]

        # Sum over period
        positive_sum = np.sum(positive_flow[-period:])
        negative_sum = np.sum(negative_flow[-period:])

        if negative_sum == 0:
            return 100.0 if positive_sum > 0 else 50.0

        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))

        return float(mfi)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_entry_quality_score(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        direction: str,
    ) -> Tuple[float, Dict[str, any]]:
        """
        Calculate comprehensive entry quality score.

        Combines multiple factors to score trade entry quality:
        - Trend alignment (ADX + direction)
        - Momentum confirmation
        - Volume support
        - Volatility suitability
        - Market regime favorability

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
            direction: 'long' or 'short'

        Returns:
            Tuple of (score 0-100, breakdown dict)
        """
        score = 0.0
        breakdown = {}

        # 1. ADX/Trend Analysis (30 points max)
        adx_result = self.calculate_adx(high, low, close)
        trend_score = 0.0

        if adx_result.should_trade_trend:
            if direction == "long" and adx_result.trend_direction == "bullish":
                trend_score = 30.0
            elif direction == "short" and adx_result.trend_direction == "bearish":
                trend_score = 30.0
            elif adx_result.trend_direction == "neutral":
                trend_score = 15.0
        elif not adx_result.should_avoid_trading:
            trend_score = 10.0

        score += trend_score
        breakdown["trend"] = trend_score

        # 2. Momentum (25 points max)
        momentum = self.calculate_momentum_suite(high, low, close, volume)
        momentum_score = 0.0

        if direction == "long":
            if momentum.is_oversold:
                momentum_score = 25.0  # Good long entry
            elif momentum.momentum_direction == "bullish":
                momentum_score = 15.0
        else:
            if momentum.is_overbought:
                momentum_score = 25.0  # Good short entry
            elif momentum.momentum_direction == "bearish":
                momentum_score = 15.0

        score += momentum_score
        breakdown["momentum"] = momentum_score

        # 3. VWAP Analysis (20 points max)
        if len(volume) > 0 and np.sum(volume) > 0:
            vwap_result = self.calculate_vwap(high, low, close, volume)
            vwap_score = 0.0

            vwap_bias = vwap_result.get_entry_bias(close[-1])
            if vwap_bias == direction:
                vwap_score = 20.0
            elif vwap_bias == "neutral":
                vwap_score = 10.0

            score += vwap_score
            breakdown["vwap"] = vwap_score
        else:
            breakdown["vwap"] = 0.0

        # 4. Market Regime (25 points max)
        regime = self.detect_market_regime(high, low, close, volume)
        regime_score = 0.0

        if regime.is_favorable:
            regime_score = 25.0
        elif regime.should_trade:
            regime_score = 15.0
        elif regime.regime == MarketRegime.RANGING and direction in ["long", "short"]:
            regime_score = 10.0  # Mean reversion possible

        score += regime_score
        breakdown["regime"] = regime_score
        breakdown["regime_type"] = regime.regime.value

        return score, breakdown

    def should_enter_trade(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        direction: str,
        min_score: float = 60.0,
    ) -> Tuple[bool, float, str]:
        """
        Determine if conditions are favorable for trade entry.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            volume: Array of volume
            direction: 'long' or 'short'
            min_score: Minimum quality score to enter (default 60)

        Returns:
            Tuple of (should_enter, quality_score, reason)
        """
        score, breakdown = self.get_entry_quality_score(
            high, low, close, volume, direction
        )

        if score >= min_score:
            reason = f"Quality score {score:.1f}/100 - " + ", ".join(
                f"{k}:{v:.0f}"
                for k, v in breakdown.items()
                if isinstance(v, (int, float))
            )
            return True, score, reason
        else:
            weak_areas = [
                k
                for k, v in breakdown.items()
                if isinstance(v, (int, float)) and v < 10
            ]
            reason = (
                f"Quality score {score:.1f}/100 too low. Weak: {', '.join(weak_areas)}"
            )
            return False, score, reason
