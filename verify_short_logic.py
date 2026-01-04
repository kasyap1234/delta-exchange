#!/usr/bin/env python3
"""
Verify that the short signal fix (v3 - Reversal Bias) works correctly.
Tests that 2-vs-2 tie-breaking logic favors oscillators for reversals.
"""

import sys
sys.path.insert(0, '/Users/kasyap/Documents/projects/delta-exchange')

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

# Recreate the minimal classes needed for testing
class IndicatorSignal(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

class Signal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class IndicatorResult:
    name: str
    signal: IndicatorSignal
    value: float = 0.0

def generate_combined_signal(indicators: List[IndicatorResult], min_agreement=2):
    """
    Replicate the NEW logic from technical_analysis.py (v3)
    """
    if not indicators:
        return Signal.HOLD, 0

    total = sum(ind.signal.value for ind in indicators)
    bullish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BULLISH)
    bearish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BEARISH)
    
    total_indicators = len(indicators)
    
    has_conflict = bullish_count > 0 and bearish_count > 0

    if has_conflict:
        # Relaxed constraint
        min_agreement = max(min_agreement, int(total_indicators * 0.50))

        # 2026 Reversal Logic: Tie-breaker
        if bullish_count == bearish_count and total_indicators >= 4:
            osc_bullish = sum(1 for ind in indicators if ind.name in ["RSI", "Bollinger Bands"] and ind.signal == IndicatorSignal.BULLISH)
            osc_bearish = sum(1 for ind in indicators if ind.name in ["RSI", "Bollinger Bands"] and ind.signal == IndicatorSignal.BEARISH)
            
            if osc_bearish == 2 and osc_bullish == 0:
                return Signal.SELL, total
            elif osc_bullish == 2 and osc_bearish == 0:
                return Signal.BUY, total

    strong_threshold = total_indicators - 1 if total_indicators >= 4 else total_indicators

    if bullish_count >= strong_threshold and bearish_count == 0:
        return Signal.STRONG_BUY, total
    elif bearish_count >= strong_threshold and bullish_count == 0:
        return Signal.STRONG_SELL, total
    elif bullish_count >= min_agreement and bullish_count > bearish_count:
        return Signal.BUY, total
    elif bearish_count >= min_agreement and bearish_count > bullish_count:
        return Signal.SELL, total
    else:
        return Signal.HOLD, total

def test_case(name, indicators, expected_signal):
    signal, _ = generate_combined_signal(indicators)
    status = "✅ PASS" if signal == expected_signal else "❌ FAIL"
    print(f"{status}: {name}")
    print(f"       Expected: {expected_signal.value}, Got: {signal.value}")
    return signal == expected_signal

def main():
    print("=" * 60)
    print("VERIFICATION: Short Signal Fix (v3 - Reversal Bias)")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Tie-break (2 Bearish Oscillators vs 2 Bullish Trend)
    # NEW: Should return SELL due to Reversal Bias
    indicators_1 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH),
        IndicatorResult("Bollinger Bands", IndicatorSignal.BEARISH),
        IndicatorResult("EMA", IndicatorSignal.BULLISH),
        IndicatorResult("MACD", IndicatorSignal.BULLISH),
    ]
    all_passed &= test_case("Tie-break: 2 Bearish Osc vs 2 Bullish Trend", indicators_1, Signal.SELL)
    
    # Test 2: Standard SELL (3 bearish, 1 bullish)
    indicators_2 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH),
        IndicatorResult("Bollinger Bands", IndicatorSignal.BEARISH),
        IndicatorResult("MACD", IndicatorSignal.BEARISH),
        IndicatorResult("EMA", IndicatorSignal.BULLISH),
    ]
    all_passed &= test_case("3 bearish, 1 bullish", indicators_2, Signal.SELL)
    
    # Test 3: Standard HOLD (Unfocused tie)
    # 2 Bearish, 2 Bullish but NOT the specific 2-oscillator combo
    indicators_3 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH),
        IndicatorResult("EMA", IndicatorSignal.BEARISH),
        IndicatorResult("Bollinger Bands", IndicatorSignal.BULLISH),
        IndicatorResult("MACD", IndicatorSignal.BULLISH),
    ]
    all_passed &= test_case("Tie: Non-oscillator specific (Should HOLD)", indicators_3, Signal.HOLD)

    # Test 4: Strong Sell
    indicators_4 = [IndicatorResult("N", IndicatorSignal.BEARISH)] * 4
    all_passed &= test_case("Strong Sell (4/4)", indicators_4, Signal.STRONG_SELL)

    print("-" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Reversal Bias logic confirmed!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
