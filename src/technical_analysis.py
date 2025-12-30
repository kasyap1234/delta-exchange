"""
Technical Analysis Module.
Provides RSI, MACD, Bollinger Bands, EMA indicators and combined signal generation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # Fallback implementations will be used

from config.settings import settings
from utils.logger import log


class Signal(str, Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class IndicatorSignal(int, Enum):
    """Individual indicator signal value."""
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


@dataclass
class IndicatorResult:
    """Result from a single indicator calculation."""
    name: str
    signal: IndicatorSignal
    value: float
    description: str


@dataclass
class TechnicalAnalysisResult:
    """Combined result from all technical indicators."""
    symbol: str
    timestamp: int
    indicators: List[IndicatorResult]
    combined_signal: Signal
    signal_strength: int  # Number of indicators agreeing
    confidence: float  # Percentage of indicators agreeing
    
    def __str__(self):
        return (f"TA Result for {self.symbol}: {self.combined_signal.value} "
                f"(strength: {self.signal_strength}, confidence: {self.confidence:.0%})")


class TechnicalAnalyzer:
    """
    Technical analysis calculator using multiple indicators.
    Supports RSI, MACD, Bollinger Bands, and EMA crossover.
    """
    
    def __init__(self):
        """Initialize technical analyzer with settings."""
        self.config = settings.trading
        
        if not TALIB_AVAILABLE:
            log.warning("TA-Lib not available, using fallback implementations")
    
    def analyze(self, close: np.ndarray, high: Optional[np.ndarray] = None, 
                low: Optional[np.ndarray] = None, symbol: str = "UNKNOWN") -> TechnicalAnalysisResult:
        """
        Perform complete technical analysis on price data.
        
        Args:
            close: Array of closing prices (oldest first)
            high: Optional array of high prices (for Bollinger Bands)
            low: Optional array of low prices (for Bollinger Bands)
            symbol: Symbol name for logging
            
        Returns:
            TechnicalAnalysisResult with all indicator results
        """
        if len(close) < 50:
            log.warning(f"Insufficient data for TA: {len(close)} candles, need at least 50")
            return TechnicalAnalysisResult(
                symbol=symbol,
                timestamp=0,
                indicators=[],
                combined_signal=Signal.HOLD,
                signal_strength=0,
                confidence=0.0
            )
        
        # If high/low not provided, estimate from close
        if high is None:
            high = close * 1.001  # Approximate
        if low is None:
            low = close * 0.999
        
        indicators = []
        
        # Calculate each indicator
        try:
            rsi_result = self._calculate_rsi(close)
            indicators.append(rsi_result)
        except Exception as e:
            log.error(f"RSI calculation failed: {e}")
        
        try:
            macd_result = self._calculate_macd(close)
            indicators.append(macd_result)
        except Exception as e:
            log.error(f"MACD calculation failed: {e}")
        
        try:
            bb_result = self._calculate_bollinger_bands(close)
            indicators.append(bb_result)
        except Exception as e:
            log.error(f"Bollinger Bands calculation failed: {e}")
        
        try:
            ema_result = self._calculate_ema_crossover(close)
            indicators.append(ema_result)
        except Exception as e:
            log.error(f"EMA calculation failed: {e}")
        
        # Generate combined signal
        combined_signal, signal_strength = self._generate_combined_signal(indicators)
        
        # Calculate confidence as the max of bullish or bearish count / total
        # This way, 2 bullish + 2 bearish = 50% confidence (not 0%)
        if indicators:
            bullish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BULLISH)
            bearish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BEARISH)
            max_directional = max(bullish_count, bearish_count)
            confidence = max_directional / len(indicators)
        else:
            confidence = 0.0
        
        result = TechnicalAnalysisResult(
            symbol=symbol,
            timestamp=int(close[-1]) if len(close) > 0 else 0,
            indicators=indicators,
            combined_signal=combined_signal,
            signal_strength=signal_strength,
            confidence=confidence
        )
        
        log.debug(f"TA for {symbol}: {result}")
        
        return result
    
    def _calculate_rsi(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Relative Strength Index.
        
        RSI < 30 = Oversold (Bullish)
        RSI > 70 = Overbought (Bearish)
        """
        if TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=self.config.rsi_period)
        else:
            rsi = self._fallback_rsi(close, self.config.rsi_period)
        
        current_rsi = rsi[-1]
        
        if np.isnan(current_rsi):
            return IndicatorResult(
                name="RSI",
                signal=IndicatorSignal.NEUTRAL,
                value=50.0,
                description="RSI calculation failed"
            )
        
        if current_rsi < self.config.rsi_oversold:
            signal = IndicatorSignal.BULLISH
            description = f"Oversold at {current_rsi:.1f}"
        elif current_rsi > self.config.rsi_overbought:
            signal = IndicatorSignal.BEARISH
            description = f"Overbought at {current_rsi:.1f}"
        else:
            signal = IndicatorSignal.NEUTRAL
            description = f"Neutral at {current_rsi:.1f}"
        
        return IndicatorResult(
            name="RSI",
            signal=signal,
            value=current_rsi,
            description=description
        )
    
    def _fallback_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Fallback RSI calculation without TA-Lib."""
        delta = np.diff(close)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial SMA
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Smoothed moving average
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD crosses above signal line = Bullish
        MACD crosses below signal line = Bearish
        """
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(
                close,
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
        else:
            macd, signal, hist = self._fallback_macd(close)
        
        current_macd = macd[-1]
        current_signal = signal[-1]
        current_hist = hist[-1]
        prev_hist = hist[-2] if len(hist) > 1 else 0
        
        if np.isnan(current_hist):
            return IndicatorResult(
                name="MACD",
                signal=IndicatorSignal.NEUTRAL,
                value=0.0,
                description="MACD calculation failed"
            )
        
        # Check for crossover
        if current_hist > 0 and prev_hist <= 0:
            signal_type = IndicatorSignal.BULLISH
            description = "Bullish crossover"
        elif current_hist < 0 and prev_hist >= 0:
            signal_type = IndicatorSignal.BEARISH
            description = "Bearish crossover"
        elif current_hist > 0:
            signal_type = IndicatorSignal.BULLISH
            description = f"Above signal line ({current_hist:.4f})"
        elif current_hist < 0:
            signal_type = IndicatorSignal.BEARISH
            description = f"Below signal line ({current_hist:.4f})"
        else:
            signal_type = IndicatorSignal.NEUTRAL
            description = "At signal line"
        
        return IndicatorResult(
            name="MACD",
            signal=signal_type,
            value=current_hist,
            description=description
        )
    
    def _fallback_macd(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback MACD calculation without TA-Lib."""
        fast_ema = self._ema(close, self.config.macd_fast)
        slow_ema = self._ema(close, self.config.macd_slow)
        macd = fast_ema - slow_ema
        signal = self._ema(macd, self.config.macd_signal)
        hist = macd - signal
        return macd, signal, hist
    
    def _calculate_bollinger_bands(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate Bollinger Bands.
        
        Price below lower band = Bullish (oversold)
        Price above upper band = Bearish (overbought)
        """
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std,
                nbdevdn=self.config.bb_std
            )
        else:
            upper, middle, lower = self._fallback_bollinger(close)
        
        current_price = close[-1]
        upper_band = upper[-1]
        lower_band = lower[-1]
        middle_band = middle[-1]
        
        if np.isnan(upper_band) or np.isnan(lower_band):
            return IndicatorResult(
                name="Bollinger Bands",
                signal=IndicatorSignal.NEUTRAL,
                value=0.0,
                description="BB calculation failed"
            )
        
        # Calculate %B (position within bands)
        band_width = upper_band - lower_band
        pct_b = (current_price - lower_band) / band_width if band_width > 0 else 0.5
        
        if current_price < lower_band:
            signal = IndicatorSignal.BULLISH
            description = f"Below lower band (oversold) - %B: {pct_b:.2f}"
        elif current_price > upper_band:
            signal = IndicatorSignal.BEARISH
            description = f"Above upper band (overbought) - %B: {pct_b:.2f}"
        else:
            signal = IndicatorSignal.NEUTRAL
            description = f"Within bands - %B: {pct_b:.2f}"
        
        return IndicatorResult(
            name="Bollinger Bands",
            signal=signal,
            value=pct_b,
            description=description
        )
    
    def _fallback_bollinger(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback Bollinger Bands calculation without TA-Lib."""
        period = self.config.bb_period
        std_dev = self.config.bb_std
        
        middle = np.zeros_like(close)
        upper = np.zeros_like(close)
        lower = np.zeros_like(close)
        
        for i in range(period - 1, len(close)):
            window = close[i - period + 1:i + 1]
            middle[i] = np.mean(window)
            std = np.std(window)
            upper[i] = middle[i] + (std * std_dev)
            lower[i] = middle[i] - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_ema_crossover(self, close: np.ndarray) -> IndicatorResult:
        """
        Calculate EMA crossover signal.
        
        Short EMA crosses above long EMA = Bullish
        Short EMA crosses below long EMA = Bearish
        """
        if TALIB_AVAILABLE:
            short_ema = talib.EMA(close, timeperiod=self.config.ema_short)
            long_ema = talib.EMA(close, timeperiod=self.config.ema_long)
        else:
            short_ema = self._ema(close, self.config.ema_short)
            long_ema = self._ema(close, self.config.ema_long)
        
        current_short = short_ema[-1]
        current_long = long_ema[-1]
        prev_short = short_ema[-2] if len(short_ema) > 1 else current_short
        prev_long = long_ema[-2] if len(long_ema) > 1 else current_long
        
        if np.isnan(current_short) or np.isnan(current_long):
            return IndicatorResult(
                name="EMA Crossover",
                signal=IndicatorSignal.NEUTRAL,
                value=0.0,
                description="EMA calculation failed"
            )
        
        diff = current_short - current_long
        prev_diff = prev_short - prev_long
        
        # Check for crossover
        if diff > 0 and prev_diff <= 0:
            signal = IndicatorSignal.BULLISH
            description = "Golden cross (bullish crossover)"
        elif diff < 0 and prev_diff >= 0:
            signal = IndicatorSignal.BEARISH
            description = "Death cross (bearish crossover)"
        elif diff > 0:
            signal = IndicatorSignal.BULLISH
            description = f"Short EMA above long EMA ({diff:.2f})"
        elif diff < 0:
            signal = IndicatorSignal.BEARISH
            description = f"Short EMA below long EMA ({diff:.2f})"
        else:
            signal = IndicatorSignal.NEUTRAL
            description = "EMAs equal"
        
        return IndicatorResult(
            name="EMA Crossover",
            signal=signal,
            value=diff,
            description=description
        )
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        
        # Initialize with SMA
        ema[period - 1] = np.mean(data[:period])
        
        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
        
        return ema
    
    def _generate_combined_signal(self, indicators: List[IndicatorResult]) -> Tuple[Signal, int]:
        """
        Generate combined trading signal from all indicators.
        
        Returns:
            Tuple of (Signal, strength)
            Strength is the sum of indicator signals (-4 to +4)
        """
        if not indicators:
            return Signal.HOLD, 0
        
        # Sum up all signals
        total = sum(ind.signal.value for ind in indicators)
        bullish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BULLISH)
        bearish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BEARISH)
        
        min_agreement = self.config.min_signal_agreement
        
        # Strong signals require 3+ indicators agreeing
        if bullish_count >= min_agreement:
            if bullish_count == len(indicators):
                return Signal.STRONG_BUY, total
            return Signal.BUY, total
        elif bearish_count >= min_agreement:
            if bearish_count == len(indicators):
                return Signal.STRONG_SELL, total
            return Signal.SELL, total
        else:
            return Signal.HOLD, total
    
    def should_enter_long(self, result: TechnicalAnalysisResult) -> bool:
        """Check if we should enter a long position."""
        return result.combined_signal in [Signal.BUY, Signal.STRONG_BUY]
    
    def should_enter_short(self, result: TechnicalAnalysisResult) -> bool:
        """Check if we should enter a short position."""
        return result.combined_signal in [Signal.SELL, Signal.STRONG_SELL]
    
    def should_close_long(self, result: TechnicalAnalysisResult) -> bool:
        """Check if we should close a long position."""
        return result.combined_signal in [Signal.SELL, Signal.STRONG_SELL]
    
    def should_close_short(self, result: TechnicalAnalysisResult) -> bool:
        """Check if we should close a short position."""
        return result.combined_signal in [Signal.BUY, Signal.STRONG_BUY]
    
    # =========================================================================
    # ENHANCED INDICATORS FOR MULTI-STRATEGY SYSTEM
    # =========================================================================
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based stops.
        
        ATR measures market volatility by decomposing the entire range
        of an asset price for that period.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ATR period (default 14)
            
        Returns:
            Current ATR value
        """
        if len(close) < period + 1:
            return 0.0
        
        if TALIB_AVAILABLE:
            atr = talib.ATR(high, low, close, timeperiod=period)
            return float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
        
        # Fallback calculation
        tr = np.zeros(len(close))
        
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Simple moving average of TR
        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[1:period+1])
        
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return float(atr[-1])
    
    def calculate_atr_stop(self, entry_price: float, atr: float,
                           side: str, multiplier: float = 2.0) -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: 'long' or 'short'
            multiplier: ATR multiplier (default 2.0)
            
        Returns:
            Stop-loss price
        """
        atr_distance = atr * multiplier
        
        if side.lower() in ['long', 'buy']:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance
    
    def calculate_trailing_stop(self, entry_price: float, 
                                 highest_since_entry: float,
                                 lowest_since_entry: float,
                                 atr: float, side: str,
                                 multiplier: float = 1.5) -> float:
        """
        Calculate trailing stop based on ATR.
        
        Args:
            entry_price: Original entry price
            highest_since_entry: Highest price since entry
            lowest_since_entry: Lowest price since entry
            atr: Current ATR value
            side: 'long' or 'short'
            multiplier: ATR multiplier for trailing distance
            
        Returns:
            Trailing stop price
        """
        trail_distance = atr * multiplier
        
        if side.lower() in ['long', 'buy']:
            # Trail below the highest price
            return highest_since_entry - trail_distance
        else:
            # Trail above the lowest price
            return lowest_since_entry + trail_distance
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        
        ADX measures the strength of a trend (not direction):
        - ADX < 20: Weak trend / choppy market (avoid trading)
        - ADX 20-25: Trend developing
        - ADX 25-50: Strong trend (good for trading)
        - ADX > 50: Very strong trend
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ADX period (default 14)
            
        Returns:
            Current ADX value (0-100)
        """
        if len(close) < period * 2:
            return 0.0
        
        if TALIB_AVAILABLE:
            try:
                adx = talib.ADX(high, low, close, timeperiod=period)
                return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            except:
                pass
        
        # Fallback ADX calculation
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))
        tr = np.zeros(len(close))
        
        for i in range(1, len(close)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
            
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Smooth the values
        atr = self._smooth_data(tr, period)
        plus_di = 100 * self._smooth_data(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._smooth_data(minus_dm, period) / (atr + 1e-10)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._smooth_data(dx, period)
        
        return float(adx[-1]) if len(adx) > 0 else 0.0
    
    def _smooth_data(self, data: np.ndarray, period: int) -> np.ndarray:
        """Smooth data using Wilder's smoothing method."""
        result = np.zeros(len(data))
        result[period] = np.sum(data[1:period+1])
        
        for i in range(period + 1, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        
        return result
    
    def is_trending(self, high: np.ndarray, low: np.ndarray, 
                    close: np.ndarray, min_adx: float = 25.0) -> Tuple[bool, float]:
        """
        Check if market is trending (suitable for trend-following trades).
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            min_adx: Minimum ADX value to consider market trending (default 25)
            
        Returns:
            Tuple of (is_trending, adx_value)
        """
        adx = self.calculate_adx(high, low, close)
        return (adx >= min_adx, adx)
    
    def calculate_volume_signal(self, volume: np.ndarray, 
                                 close: np.ndarray,
                                 period: int = 20,
                                 min_volume_ratio: float = 1.2) -> IndicatorResult:
        """
        Analyze volume to confirm price movements.
        
        High volume on price increase = bullish
        High volume on price decrease = bearish
        Low volume = neutral (no confirmation)
        
        Args:
            volume: Array of volume data
            close: Array of close prices
            period: Period for average volume
            min_volume_ratio: Ratio above average to trigger signal
            
        Returns:
            IndicatorResult with volume signal
        """
        if len(volume) < period + 1:
            return IndicatorResult(
                name="Volume",
                signal=IndicatorSignal.NEUTRAL,
                value=0.0,
                description="Insufficient data"
            )
        
        avg_volume = np.mean(volume[-period-1:-1])
        current_volume = volume[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        price_change = close[-1] - close[-2]
        price_change_pct = price_change / close[-2] if close[-2] > 0 else 0
        
        # Confirmation threshold (reduced for higher frequency)
        if volume_ratio > min_volume_ratio:
            if price_change_pct > 0.0005:  # >0.05% up
                return IndicatorResult(
                    name="Volume",
                    signal=IndicatorSignal.BULLISH,
                    value=volume_ratio,
                    description=f"Volume confirmation ({volume_ratio:.1f}x avg)"
                )
            elif price_change_pct < -0.0005:  # >0.05% down
                return IndicatorResult(
                    name="Volume",
                    signal=IndicatorSignal.BEARISH,
                    value=volume_ratio,
                    description=f"Volume confirmation ({volume_ratio:.1f}x avg)"
                )
        
        return IndicatorResult(
            name="Volume",
            signal=IndicatorSignal.NEUTRAL,
            value=volume_ratio,
            description=f"Normal/Low volume ({volume_ratio:.1f}x avg)"
        )
    
    def get_trend_direction(self, close: np.ndarray, 
                            ema_short: int = 50,
                            ema_long: int = 200) -> str:
        """
        Determine overall trend direction using EMA cross.
        
        Used for multi-timeframe analysis to filter trades.
        
        Args:
            close: Array of close prices
            ema_short: Short EMA period (default 50)
            ema_long: Long EMA period (default 200)
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(close) < ema_long + 1:
            return 'neutral'
        
        if TALIB_AVAILABLE:
            short_ema = talib.EMA(close, timeperiod=ema_short)
            long_ema = talib.EMA(close, timeperiod=ema_long)
        else:
            short_ema = self._ema(close, ema_short)
            long_ema = self._ema(close, ema_long)
        
        current_short = short_ema[-1]
        current_long = long_ema[-1]
        
        if np.isnan(current_short) or np.isnan(current_long):
            return 'neutral'
        
        diff_pct = (current_short - current_long) / current_long
        
        if diff_pct > 0.002:  # Short EMA >0.2% above long
            return 'bullish'
        elif diff_pct < -0.002:  # Short EMA >0.2% below long
            return 'bearish'
        else:
            return 'neutral'
    
    def analyze_with_volume(self, close: np.ndarray, 
                            high: np.ndarray, 
                            low: np.ndarray,
                            volume: np.ndarray,
                            symbol: str = "UNKNOWN") -> TechnicalAnalysisResult:
        """
        Enhanced analysis including volume confirmation.
        
        Args:
            close: Array of close prices
            high: Array of high prices
            low: Array of low prices
            volume: Array of volume data
            symbol: Symbol name
            
        Returns:
            TechnicalAnalysisResult with volume included
        """
        # Run standard analysis
        result = self.analyze(close, high, low, symbol)
        
        # Add volume signal
        volume_signal = self.calculate_volume_signal(volume, close)
        result.indicators.append(volume_signal)
        
        # Recalculate combined signal with volume
        result.combined_signal, result.signal_strength = \
            self._generate_combined_signal(result.indicators)
        result.confidence = abs(result.signal_strength) / len(result.indicators)
        
        return result


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis for trend confirmation.
    
    Uses higher timeframe to determine trend direction,
    and lower timeframe for entry timing.
    """
    
    def __init__(self, client, 
                 higher_tf: str = '4h',
                 entry_tf: str = '15m'):
        """
        Initialize MTF analyzer.
        
        Args:
            client: Delta Exchange client for fetching candles
            higher_tf: Higher timeframe for trend (default 4h)
            entry_tf: Entry timeframe (default 15m)
        """
        self.client = client
        self.higher_tf = higher_tf
        self.entry_tf = entry_tf
        self.analyzer = TechnicalAnalyzer()
    
    def analyze(self, symbol: str) -> dict:
        """
        Perform multi-timeframe analysis.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with analysis results
        """
        from utils.logger import log
        
        try:
            # Fetch higher timeframe candles for trend
            htf_candles = self.client.get_candles(
                symbol=symbol,
                resolution=self.higher_tf
            )
            
            # Fetch entry timeframe candles for signals
            entry_candles = self.client.get_candles(
                symbol=symbol,
                resolution=self.entry_tf
            )
            
            if len(htf_candles) < 50 or len(entry_candles) < 50:
                log.warning(f"Insufficient candles for MTF analysis: {symbol}")
                return {
                    'symbol': symbol,
                    'higher_tf_trend': 'neutral',
                    'entry_signal': None,
                    'aligned': False,
                    'should_trade': False
                }
            
            # Convert to numpy arrays
            htf_close = np.array([c.close for c in htf_candles])
            entry_close = np.array([c.close for c in entry_candles])
            entry_high = np.array([c.high for c in entry_candles])
            entry_low = np.array([c.low for c in entry_candles])
            
            # Get higher timeframe trend
            htf_trend = self.analyzer.get_trend_direction(htf_close)
            
            # Get entry signal
            entry_result = self.analyzer.analyze(
                entry_close, entry_high, entry_low, symbol
            )
            
            # Check alignment
            is_aligned = self._check_alignment(htf_trend, entry_result)
            
            # Calculate ATR for stops
            atr = self.analyzer.calculate_atr(entry_high, entry_low, entry_close)
            
            # Should trade decision - Relaxed for more opportunities
            # Trade if: (aligned OR neutral trend) AND conf >= 0.25
            # We only reject if trend is EXPLICITLY opposite to signal
            should_trade = (is_aligned or htf_trend == 'neutral') and entry_result.confidence >= 0.25
            
            # DEBUG: Log detailed analysis
            log.info(f"[MTF] {symbol} Analysis:")
            log.info(f"  Higher TF ({self.higher_tf}) Trend: {htf_trend.upper()}")
            log.info(f"  Entry TF ({self.entry_tf}) Signal: {entry_result.combined_signal.value}")
            log.info(f"  Entry Confidence: {entry_result.confidence:.2f}")
            log.info(f"  Aligned: {is_aligned}")
            log.info(f"  Should Trade: {should_trade}")
            if not should_trade:
                if not is_aligned and htf_trend != 'neutral':
                    log.info(f"  → NO TRADE: Trend ({htf_trend}) conflicts with entry signal ({entry_result.combined_signal.value})")
                elif entry_result.confidence < 0.25:
                    log.info(f"  → NO TRADE: Confidence {entry_result.confidence:.2f} < 0.25 threshold")
            else:
                if htf_trend == 'neutral':
                    log.info(f"  → READY TO TRADE: {entry_result.combined_signal.value} signal (trend is neutral)")
                else:
                    log.info(f"  → READY TO TRADE: {htf_trend.upper()} trend + {entry_result.combined_signal.value} entry")
            for ind in entry_result.indicators:
                log.info(f"    - {ind.name}: {ind.signal.value}")
            
            return {
                'symbol': symbol,
                'higher_tf': self.higher_tf,
                'entry_tf': self.entry_tf,
                'higher_tf_trend': htf_trend,
                'entry_signal': entry_result.combined_signal.value,
                'entry_confidence': entry_result.confidence,
                'aligned': is_aligned,
                'should_trade': should_trade,
                'atr': atr,
                'current_price': float(entry_close[-1]),
                'indicators': [
                    {'name': ind.name, 'signal': ind.signal.value}
                    for ind in entry_result.indicators
                ]
            }
            
        except Exception as e:
            log.error(f"MTF analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'higher_tf_trend': 'neutral',
                'entry_signal': None,
                'aligned': False,
                'should_trade': False,
                'error': str(e)
            }
    
    def _check_alignment(self, htf_trend: str, 
                         entry_result: TechnicalAnalysisResult) -> bool:
        """
        Check if entry signal aligns with higher timeframe trend.
        
        Only trade:
        - LONG when HTF is bullish AND entry shows BUY
        - SHORT when HTF is bearish AND entry shows SELL
        """
        if htf_trend == 'bullish':
            return entry_result.combined_signal in [Signal.BUY, Signal.STRONG_BUY]
        elif htf_trend == 'bearish':
            return entry_result.combined_signal in [Signal.SELL, Signal.STRONG_SELL]
        
        return False
    
    def should_enter_long(self, symbol: str) -> tuple:
        """
        Check if conditions are right for a long entry.
        
        Returns:
            Tuple of (should_enter, confidence, atr)
        """
        analysis = self.analyze(symbol)
        
        if not analysis.get('should_trade'):
            return False, 0.0, 0.0
        
        if analysis['higher_tf_trend'] == 'bullish' and \
           analysis['entry_signal'] in ['buy', 'strong_buy']:
            return True, analysis['entry_confidence'], analysis['atr']
        
        return False, 0.0, 0.0
    
    def should_enter_short(self, symbol: str) -> tuple:
        """
        Check if conditions are right for a short entry.
        
        Returns:
            Tuple of (should_enter, confidence, atr)
        """
        analysis = self.analyze(symbol)
        
        if not analysis.get('should_trade'):
            return False, 0.0, 0.0
        
        if analysis['higher_tf_trend'] == 'bearish' and \
           analysis['entry_signal'] in ['sell', 'strong_sell']:
            return True, analysis['entry_confidence'], analysis['atr']
        
        return False, 0.0, 0.0

