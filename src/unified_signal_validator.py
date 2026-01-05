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
        self.rsi_overbought = getattr(settings.trading, "rsi_overbought", 70)
        self.rsi_oversold = getattr(settings.trading, "rsi_oversold", 30)
        
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

        # 2. Indicator Agreement Filter (Balanced)
        if hasattr(ta_result, 'signal_strength'):
            agreement = abs(ta_result.signal_strength)
            if agreement < 2:
                return False, ValidationResult.REJECTED_WEAK_AGREEMENT, f"Agreement {agreement} < 2"

        # 3. ADX Trend Filter (Stronger - avoid weak trends)
        if adx > 0 and adx < 25.0:
            return False, ValidationResult.REJECTED_WEAK_TREND, f"ADX {adx:.1f} < 25.0 (Trend too weak)"
        
        # 4. Higher Timeframe Trend Alignment (Multi-Timeframe Filter)
        if higher_tf_trend not in ["unknown", "neutral"]:
            # 2026 Enhanced Logic: Allow counter-trend trades ONLY IF confidence is VERY high (Extreme Reversal)
            is_extreme_confidence = hasattr(ta_result, 'confidence') and ta_result.confidence >= 0.75
            
            if direction == "long" and higher_tf_trend != "bullish":
                if not is_extreme_confidence:
                    return False, ValidationResult.REJECTED_COUNTER_TREND, f"Long signal conflicts with {higher_tf_trend} 4h trend (Confidence {ta_result.confidence:.2f} < 0.75)"
                log.info(f"Allowing counter-trend LONG due to EXTREME confidence {ta_result.confidence:.2f}")
                
            if direction == "short" and higher_tf_trend != "bearish":
                if not is_extreme_confidence:
                    return False, ValidationResult.REJECTED_COUNTER_TREND, f"Short signal conflicts with {higher_tf_trend} 4h trend (Confidence {ta_result.confidence:.2f} < 0.75)"
                log.info(f"Allowing counter-trend SHORT due to EXTREME confidence {ta_result.confidence:.2f}")

        # 5. RSI Safety Filter
        if rsi is not None:
            if direction == "long" and rsi > self.rsi_overbought:
                return False, ValidationResult.REJECTED_OVERBOUGHT, f"RSI {rsi:.1f} too high for LONG (> {self.rsi_overbought})"
            if direction == "short" and rsi < self.rsi_oversold:
                return False, ValidationResult.REJECTED_OVERSOLD, f"RSI {rsi:.1f} too low for SHORT (< {self.rsi_oversold})"

        # 6. Market Regime / Choppiness Filter
        # Use choppiness from ta_result if available
        chop_val = getattr(ta_result, 'choppiness', 50.0)
        
        if chop_val > 61.8:
            # Extreme Choppiness - Reject trend trades
            return False, ValidationResult.REJECTED_CHOPPY_MARKET, f"Market extreme chop (CHOP {chop_val:.1f} > 61.8)"
        
        if chop_val > 55.0 and adx < 25.0:
            # High chop and weak trend - require very high confidence
            if hasattr(ta_result, 'confidence') and ta_result.confidence < 0.65:
                return False, ValidationResult.REJECTED_CHOPPY_MARKET, f"Choppy market (CHOP {chop_val:.1f}, ADX {adx:.1f}) - low confidence"

        # 7. Volume Confirmation Filter
        if volume_signal == "neutral":
             # Neutral volume is okay if other factors are strong, but log it
             log.debug(f"Volume signal is neutral for {symbol}")
        elif direction == "long" and volume_signal == "bearish":
             return False, ValidationResult.REJECTED_LOW_VOLUME, "LONG signal with bearish volume bias"
        elif direction == "short" and volume_signal == "bullish":
             return False, ValidationResult.REJECTED_LOW_VOLUME, "SHORT signal with bullish volume bias"

        # 8. Very Weak Trend Filter
        if adx < 15 and chop_val > 50:
            return False, ValidationResult.REJECTED_WEAK_TREND, f"Market too flat (ADX {adx:.1f}, CHOP {chop_val:.1f})"

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
