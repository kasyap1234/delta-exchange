"""
Unified Signal Validation Module.
Ensures consistency between backtesting and live trading logic.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from config.settings import settings
from utils.logger import log

class ValidationResult(Enum):
    PASSED = "passed"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
    REJECTED_WEAK_AGREEMENT = "rejected_weak_agreement"
    REJECTED_WEAK_TREND = "rejected_weak_trend"
    REJECTED_COUNTER_TREND = "rejected_counter_trend"
    REJECTED_OVERBOUGHT = "rejected_overbought"
    REJECTED_OVERSOLD = "rejected_oversold"
    REJECTED_LOW_VOLATILITY = "rejected_low_volatility"
    REJECTED_CHOPPY_MARKET = "rejected_choppy_market"
    REJECTED_LOW_VOLUME = "rejected_low_volume"
    REJECTED_REGIME_CONFLICT = "rejected_regime_conflict"
    REJECTED_COOLDOWN = "rejected_cooldown"

class UnifiedSignalValidator:
    """
    Centralized signal validation used by BOTH backtest and live trading.
    """
    
    def __init__(self):
        # Configuration from settings or sensible defaults
        # Balanced: 0.50 confidence for diverse signal capture
        self.min_confidence = getattr(settings.trading, "min_confidence", 0.50)
        self.min_agreement = getattr(settings.trading, "min_signal_agreement", 1)
        
        # State for whiplash protection
        self.last_loss_time: Dict[str, datetime] = {}
        self.consecutive_losses: Dict[str, int] = {}
        self.min_adx = getattr(settings.signal_filter, "min_adx_for_trend", 15.0)
        self.rsi_overbought = getattr(settings.signal_filter, "rsi_overbought", 65)
        self.rsi_oversold = getattr(settings.signal_filter, "rsi_oversold", 35)
        
    def validate_entry(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        ta_result: Any,
        higher_tf_trend: str = "neutral",
        adx: float = 0.0,
        rsi: Optional[float] = None,
        volume_signal: str = "neutral",
        market_regime: str = "trending",
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, ValidationResult, str]:
        """
        Validate if an entry signal should be taken.
        
        Returns: (is_valid, result_enum, reason_string)
        """
        # 1. Confidence Filter
        if hasattr(ta_result, 'confidence') and ta_result.confidence < self.min_confidence:
            return False, ValidationResult.REJECTED_LOW_CONFIDENCE, f"Confidence {ta_result.confidence:.2f} < {self.min_confidence}"

        # 2. Indicator Agreement Filter
        # signal_strength can be negative for SELL signals, so use absolute value
        if hasattr(ta_result, 'signal_strength'):
            agreement = abs(ta_result.signal_strength)
            if agreement < self.min_agreement:
                return False, ValidationResult.REJECTED_WEAK_AGREEMENT, f"Agreement {agreement} < {self.min_agreement}"

        # 3. ADX Trend Filter
        if adx > 0 and adx < self.min_adx:
            return False, ValidationResult.REJECTED_WEAK_TREND, f"ADX {adx:.1f} < {self.min_adx} (weak trend)"
        
        # 4. Higher Timeframe Trend Alignment (Multi-Timeframe Filter)
        if higher_tf_trend not in ["unknown", "neutral"]:
            # 2026 Enhanced Logic: Allow counter-trend trades IF confidence is high (Specialized Reversal)
            # Relaxed to 0.50 for testing to see if shorts trigger
            is_high_confidence = hasattr(ta_result, 'confidence') and ta_result.confidence >= 0.50
            
            if direction == "long" and higher_tf_trend != "bullish":
                if not is_high_confidence:
                    return False, ValidationResult.REJECTED_COUNTER_TREND, f"Long signal conflicts with {higher_tf_trend} 4h trend (Confidence {ta_result.confidence:.2f} < 0.50)"
                log.info(f"Allowing counter-trend LONG due to high confidence {ta_result.confidence:.2f}")
                
            if direction == "short" and higher_tf_trend != "bearish":
                if not is_high_confidence:
                    return False, ValidationResult.REJECTED_COUNTER_TREND, f"Short signal conflicts with {higher_tf_trend} 4h trend (Confidence {ta_result.confidence:.2f} < 0.50)"
                log.info(f"Allowing counter-trend SHORT due to high confidence {ta_result.confidence:.2f}")

        # 5. RSI Safety Filter
        if rsi is not None:
            if direction == "long" and rsi > self.rsi_overbought:
                return False, ValidationResult.REJECTED_OVERBOUGHT, f"RSI {rsi:.1f} too high for LONG (> {self.rsi_overbought})"
            if direction == "short" and rsi < self.rsi_oversold:
                return False, ValidationResult.REJECTED_OVERSOLD, f"RSI {rsi:.1f} too low for SHORT (< {self.rsi_oversold})"

        # 6. Market Regime Filter
        # Avoid trading in choppy markets for trend-following strategies
        if market_regime in ["choppy", "sideways", "ranging"] and adx < 25:
             # Threshold raised to 0.55 for balanced quality
             if hasattr(ta_result, 'confidence') and ta_result.confidence >= 0.55:
                 log.debug(f"Allowing trade in {market_regime} regime due to decent confidence {ta_result.confidence:.2f}")
             else:
                 return False, ValidationResult.REJECTED_CHOPPY_MARKET, f"Market regime is {market_regime} with weak ADX {adx:.1f} and low confidence"

        # 7. Volume Confirmation Filter
        if volume_signal == "neutral":
             # This is a soft filter - can be made strict if needed
             log.debug(f"Volume signal is neutral for {symbol}, proceeding with caution")
        elif volume_signal == "weak":
             return False, ValidationResult.REJECTED_LOW_VOLUME, f"Volume confirmation failed (signal: {volume_signal})"

        # Filter out very choppy markets (ADX < 15)
        if adx < 15 and volume_signal == "neutral":
            return False, ValidationResult.REJECTED_WEAK_TREND, f"Very weak ADX {adx:.1f} - market too choppy"

        # 9. Loss Cooldown (Whiplash Protection)
        cooldown_mins = getattr(settings.signal_filter, "loss_cooldown_minutes", 30)
        check_time = timestamp or datetime.now()
        
        if symbol in self.last_loss_time:
            time_since_loss = (check_time - self.last_loss_time[symbol]).total_seconds() / 60
            if time_since_loss < cooldown_mins:
                return False, ValidationResult.REJECTED_COOLDOWN, f"In cooldown ({time_since_loss:.1f}/{cooldown_mins}m)"

        return True, ValidationResult.PASSED, "Signal passed all unified filters"

    def record_trade_result(self, symbol: str, pnl: float, timestamp: Optional[datetime] = None):
        """Record trade result to handle cooldowns."""
        if pnl < 0:
            self.last_loss_time[symbol] = timestamp or datetime.now()
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
        else:
            self.consecutive_losses[symbol] = 0
            if symbol in self.last_loss_time:
                del self.last_loss_time[symbol]
