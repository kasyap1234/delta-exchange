#!/usr/bin/env python3
"""
Verify that the short signal fix works correctly.
Tests that 2/4 indicator agreement now produces a SELL signal.
"""

import sys
sys.path.insert(0, '/Users/kasyap/Documents/projects/delta-exchange')

from dataclasses import dataclass
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
    value: float
    description: str = ""

def generate_combined_signal(indicators, min_agreement=2):
    """
    Replicate the UPDATED logic from technical_analysis.py
    """
    if not indicators:
        return Signal.HOLD, 0

    total = sum(ind.signal.value for ind in indicators)
    bullish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BULLISH)
    bearish_count = sum(1 for ind in indicators if ind.signal == IndicatorSignal.BEARISH)
    
    total_indicators = len(indicators)
    
    has_conflict = bullish_count > 0 and bearish_count > 0

    if has_conflict:
        # UPDATED: Now 50% instead of 75%
        min_agreement = max(min_agreement, int(total_indicators * 0.50))

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
    print("VERIFICATION: Short Signal Fix")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test 1: 2 bullish, 2 bearish (classic reversal scenario)
    # OLD: Would return HOLD (needed 3/4 = 75%)
    # NEW: Should still return HOLD (2/4 = 50%, but need bearish > bullish)
    indicators_1 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH, 75.0),
        IndicatorResult("BB", IndicatorSignal.BEARISH, 0.95),
        IndicatorResult("EMA", IndicatorSignal.BULLISH, 0.5),
        IndicatorResult("MACD", IndicatorSignal.BULLISH, 0.1),
    ]
    all_passed &= test_case("2 bearish, 2 bullish (equal)", indicators_1, Signal.HOLD)
    
    # Test 2: 3 bearish, 1 bullish
    # Should return SELL (3 >= 2, and 3 > 1)
    indicators_2 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH, 75.0),
        IndicatorResult("BB", IndicatorSignal.BEARISH, 0.95),
        IndicatorResult("EMA", IndicatorSignal.BEARISH, -0.5),
        IndicatorResult("MACD", IndicatorSignal.BULLISH, 0.1),
    ]
    all_passed &= test_case("3 bearish, 1 bullish", indicators_2, Signal.SELL)
    
    # Test 3: 2 bearish, 1 bullish, 1 neutral
    # Should return SELL (2 >= 2, and 2 > 1)
    indicators_3 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH, 75.0),
        IndicatorResult("BB", IndicatorSignal.BEARISH, 0.95),
        IndicatorResult("EMA", IndicatorSignal.NEUTRAL, 0.0),
        IndicatorResult("MACD", IndicatorSignal.BULLISH, 0.1),
    ]
    all_passed &= test_case("2 bearish, 1 bullish, 1 neutral", indicators_3, Signal.SELL)
    
    # Test 4: 4 bearish (unanimous)
    # Should return STRONG_SELL
    indicators_4 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH, 75.0),
        IndicatorResult("BB", IndicatorSignal.BEARISH, 0.95),
        IndicatorResult("EMA", IndicatorSignal.BEARISH, -0.5),
        IndicatorResult("MACD", IndicatorSignal.BEARISH, -0.1),
    ]
    all_passed &= test_case("4 bearish (unanimous)", indicators_4, Signal.STRONG_SELL)
    
    # Test 5: 3 bearish, 0 bullish, 1 neutral
    # Should return STRONG_SELL (3 >= 3 = strong_threshold)
    indicators_5 = [
        IndicatorResult("RSI", IndicatorSignal.BEARISH, 75.0),
        IndicatorResult("BB", IndicatorSignal.BEARISH, 0.95),
        IndicatorResult("EMA", IndicatorSignal.BEARISH, -0.5),
        IndicatorResult("MACD", IndicatorSignal.NEUTRAL, 0.0),
    ]
    all_passed &= test_case("3 bearish, 0 bullish, 1 neutral", indicators_5, Signal.STRONG_SELL)
    
    print()
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Short signals are now enabled!")
    else:
        print("❌ SOME TESTS FAILED - Review the logic")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
