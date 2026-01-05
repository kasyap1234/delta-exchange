#!/usr/bin/env python3
"""
Analyze backtest trades to find patterns in losing trades.
"""

import json
from collections import defaultdict
from datetime import datetime

# Trade data from the backtest (extracted from output)
trades = [
    {"id": 1, "dir": "short", "entry": 129.60, "exit": 132.21, "pnl": -120.61, "pct": -2.01},
    {"id": 2, "dir": "long", "entry": 135.52, "exit": 132.80, "pnl": -119.48, "pct": -2.01},
    {"id": 3, "dir": "short", "entry": 132.29, "exit": 133.51, "pnl": -54.16, "pct": -0.92},
    {"id": 4, "dir": "long", "entry": 134.67, "exit": 136.70, "pnl": 88.41, "pct": 1.51},
    {"id": 5, "dir": "short", "entry": 136.14, "exit": 133.99, "pnl": 93.48, "pct": 1.58},
    {"id": 6, "dir": "long", "entry": 133.91, "exit": 132.74, "pnl": -51.93, "pct": -0.87},
    {"id": 7, "dir": "short", "entry": 132.24, "exit": 133.10, "pnl": -38.75, "pct": -0.65},
    {"id": 8, "dir": "short", "entry": 132.31, "exit": 133.23, "pnl": -41.08, "pct": -0.70},
    {"id": 9, "dir": "short", "entry": 138.28, "exit": 139.74, "pnl": -62.44, "pct": -1.06},
    {"id": 10, "dir": "long", "entry": 138.34, "exit": 136.71, "pnl": -69.22, "pct": -1.18},
    {"id": 11, "dir": "long", "entry": 134.09, "exit": 137.69, "pnl": 156.01, "pct": 2.69},
    {"id": 12, "dir": "long", "entry": 139.12, "exit": 137.43, "pnl": -71.61, "pct": -1.21},
    {"id": 13, "dir": "short", "entry": 132.79, "exit": 131.00, "pnl": 79.19, "pct": 1.35},
    {"id": 14, "dir": "long", "entry": 133.99, "exit": 132.67, "pnl": -57.97, "pct": -0.98},
    {"id": 15, "dir": "short", "entry": 132.28, "exit": 132.89, "pnl": -26.95, "pct": -0.46},
    {"id": 16, "dir": "short", "entry": 132.52, "exit": 132.98, "pnl": -20.12, "pct": -0.34},
    {"id": 17, "dir": "short", "entry": 131.82, "exit": 130.48, "pnl": 59.53, "pct": 1.02},
    {"id": 18, "dir": "short", "entry": 131.42, "exit": 129.47, "pnl": 87.04, "pct": 1.48},
    {"id": 19, "dir": "short", "entry": 129.43, "exit": 130.88, "pnl": -66.26, "pct": -1.12},
    {"id": 20, "dir": "long", "entry": 131.99, "exit": 132.98, "pnl": 44.40, "pct": 0.75},
    {"id": 21, "dir": "long", "entry": 128.39, "exit": 127.37, "pnl": -46.82, "pct": -0.79},
    {"id": 22, "dir": "long", "entry": 128.93, "exit": 128.22, "pnl": -32.26, "pct": -0.55},
    {"id": 23, "dir": "short", "entry": 127.84, "exit": 126.29, "pnl": 71.28, "pct": 1.22},
    {"id": 24, "dir": "short", "entry": 126.59, "exit": 127.80, "pnl": -56.26, "pct": -0.95},
    {"id": 25, "dir": "long", "entry": 129.10, "exit": 126.50, "pnl": -118.05, "pct": -2.01},
    {"id": 26, "dir": "short", "entry": 124.89, "exit": 123.68, "pnl": 56.16, "pct": 0.96},
    {"id": 27, "dir": "long", "entry": 125.45, "exit": 128.36, "pnl": 136.09, "pct": 2.33},
    {"id": 28, "dir": "long", "entry": 127.44, "exit": 124.88, "pnl": -119.27, "pct": -2.01},
    {"id": 29, "dir": "short", "entry": 118.76, "exit": 121.15, "pnl": -119.26, "pct": -2.01},  # Fixed: was listed as trade 30
    {"id": 30, "dir": "short", "entry": 122.95, "exit": 121.09, "pnl": 88.75, "pct": 1.51},  # Fixed
    {"id": 31, "dir": "long", "entry": 121.94, "exit": 125.35, "pnl": 164.29, "pct": 2.79},
    {"id": 32, "dir": "short", "entry": 125.78, "exit": 126.57, "pnl": -37.31, "pct": -0.62},
    {"id": 33, "dir": "long", "entry": 126.75, "exit": 126.02, "pnl": -34.40, "pct": -0.58},
    {"id": 34, "dir": "short", "entry": 125.17, "exit": 125.01, "pnl": 7.49, "pct": 0.13},
    {"id": 35, "dir": "long", "entry": 126.28, "exit": 125.21, "pnl": -50.08, "pct": -0.84},
    {"id": 36, "dir": "short", "entry": 124.98, "exit": 126.05, "pnl": -50.39, "pct": -0.85},
    {"id": 37, "dir": "short", "entry": 124.34, "exit": 125.01, "pnl": -31.57, "pct": -0.54},
    {"id": 38, "dir": "short", "entry": 124.17, "exit": 123.05, "pnl": 52.61, "pct": 0.90},
    {"id": 39, "dir": "short", "entry": 123.08, "exit": 124.62, "pnl": -73.83, "pct": -1.25},
    {"id": 40, "dir": "long", "entry": 124.07, "exit": 123.71, "pnl": -17.23, "pct": -0.29},
    {"id": 41, "dir": "short", "entry": 121.39, "exit": 122.14, "pnl": -36.30, "pct": -0.62},
    {"id": 42, "dir": "long", "entry": 123.52, "exit": 122.97, "pnl": -26.09, "pct": -0.45},
    {"id": 43, "dir": "long", "entry": 122.82, "exit": 124.91, "pnl": 98.38, "pct": 1.70},
    {"id": 44, "dir": "long", "entry": 123.90, "exit": 121.41, "pnl": -117.65, "pct": -2.01},
    {"id": 45, "dir": "short", "entry": 121.63, "exit": 122.51, "pnl": -42.32, "pct": -0.73},
    {"id": 46, "dir": "long", "entry": 123.02, "exit": 122.80, "pnl": -10.54, "pct": -0.18},
    {"id": 47, "dir": "long", "entry": 124.28, "exit": 123.95, "pnl": -15.54, "pct": -0.27},
    {"id": 48, "dir": "long", "entry": 124.84, "exit": 124.31, "pnl": -24.62, "pct": -0.43},
    {"id": 49, "dir": "short", "entry": 123.53, "exit": 124.27, "pnl": -34.32, "pct": -0.60},
    {"id": 50, "dir": "short", "entry": 123.98, "exit": 123.78, "pnl": 9.06, "pct": 0.16},
    {"id": 51, "dir": "long", "entry": 124.78, "exit": 124.17, "pnl": -28.22, "pct": -0.49},
    {"id": 52, "dir": "long", "entry": 126.62, "exit": 127.08, "pnl": 20.95, "pct": 0.37},
    {"id": 53, "dir": "long", "entry": 125.42, "exit": 126.15, "pnl": 33.12, "pct": 0.58},
    {"id": 54, "dir": "long", "entry": 126.54, "exit": 127.42, "pnl": 40.10, "pct": 0.70},
    {"id": 55, "dir": "long", "entry": 127.35, "exit": 128.75, "pnl": 63.36, "pct": 1.10},
    {"id": 56, "dir": "long", "entry": 128.99, "exit": 127.68, "pnl": -58.75, "pct": -1.01},
    {"id": 57, "dir": "long", "entry": 129.03, "exit": 130.72, "pnl": 75.77, "pct": 1.31},
    {"id": 58, "dir": "long", "entry": 131.19, "exit": 132.44, "pnl": 55.59, "pct": 0.96},
    {"id": 59, "dir": "long", "entry": 132.51, "exit": 132.18, "pnl": -14.45, "pct": -0.25},
    {"id": 60, "dir": "short", "entry": 130.59, "exit": 131.30, "pnl": -31.61, "pct": -0.54},
    {"id": 61, "dir": "long", "entry": 133.91, "exit": 133.81, "pnl": -4.25, "pct": -0.07},
]

def analyze_trades():
    print("=" * 70)
    print("TRADE PATTERN ANALYSIS")
    print("=" * 70)
    
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] < 0]
    
    print(f"\nTotal Trades: {len(trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    
    # 1. Direction Analysis
    print("\n" + "-" * 50)
    print("1. DIRECTION ANALYSIS")
    print("-" * 50)
    
    long_trades = [t for t in trades if t["dir"] == "long"]
    short_trades = [t for t in trades if t["dir"] == "short"]
    
    long_winners = len([t for t in long_trades if t["pnl"] > 0])
    short_winners = len([t for t in short_trades if t["pnl"] > 0])
    
    print(f"\nLONG trades: {len(long_trades)}")
    print(f"  - Winners: {long_winners} ({long_winners/len(long_trades)*100:.1f}%)")
    print(f"  - Avg P&L: ${sum(t['pnl'] for t in long_trades)/len(long_trades):.2f}")
    print(f"  - Total P&L: ${sum(t['pnl'] for t in long_trades):.2f}")
    
    print(f"\nSHORT trades: {len(short_trades)}")
    print(f"  - Winners: {short_winners} ({short_winners/len(short_trades)*100:.1f}%)")
    print(f"  - Avg P&L: ${sum(t['pnl'] for t in short_trades)/len(short_trades):.2f}")
    print(f"  - Total P&L: ${sum(t['pnl'] for t in short_trades):.2f}")
    
    # 2. Loss Size Analysis
    print("\n" + "-" * 50)
    print("2. LOSS SIZE DISTRIBUTION")
    print("-" * 50)
    
    small_losses = [t for t in losers if t["pct"] > -0.5]
    medium_losses = [t for t in losers if -1.0 < t["pct"] <= -0.5]
    large_losses = [t for t in losers if t["pct"] <= -1.0]
    stopped_out = [t for t in losers if t["pct"] <= -2.0]
    
    print(f"\nSmall losses (< 0.5%): {len(small_losses)} trades = ${sum(t['pnl'] for t in small_losses):.2f}")
    print(f"Medium losses (0.5-1%): {len(medium_losses)} trades = ${sum(t['pnl'] for t in medium_losses):.2f}")
    print(f"Large losses (1-2%): {len(large_losses)} trades = ${sum(t['pnl'] for t in large_losses):.2f}")
    print(f"Stopped out (2%+): {len(stopped_out)} trades = ${sum(t['pnl'] for t in stopped_out):.2f}")
    
    # 3. Consecutive Loss Pattern
    print("\n" + "-" * 50)
    print("3. CONSECUTIVE LOSS PATTERNS")
    print("-" * 50)
    
    consecutive_losses = 0
    max_consecutive = 0
    current_streak = 0
    loss_streaks = []
    
    for t in trades:
        if t["pnl"] < 0:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        loss_streaks.append(current_streak)
    
    print(f"\nMax consecutive losses: {max_consecutive}")
    print(f"Loss streak distribution: {sorted(loss_streaks, reverse=True)[:5]}")
    
    # 4. Reversal Pattern (losing after flip)
    print("\n" + "-" * 50)
    print("4. DIRECTION REVERSAL ANALYSIS")
    print("-" * 50)
    
    flips = 0
    flip_losses = 0
    for i in range(1, len(trades)):
        if trades[i]["dir"] != trades[i-1]["dir"]:
            flips += 1
            if trades[i]["pnl"] < 0:
                flip_losses += 1
    
    print(f"\nDirection flips: {flips}")
    print(f"Losses after flip: {flip_losses} ({flip_losses/flips*100:.1f}%)")
    
    # 5. Price Level Analysis
    print("\n" + "-" * 50)
    print("5. PRICE LEVEL ANALYSIS (Entry Clusters)")
    print("-" * 50)
    
    # Find entries at similar prices
    for t in losers:
        similar = [t2 for t2 in trades if abs(t2["entry"] - t["entry"]) < 2 and t2["id"] != t["id"]]
        if len(similar) >= 2:
            print(f"Trade #{t['id']} ({t['dir']}) @ {t['entry']:.2f} - {len(similar)} similar entries")
    
    # 6. Key Findings Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
PATTERN 1: HIGH REVERSAL LOSS RATE
- {:.1f}% of trades after direction flip result in losses
- RECOMMENDATION: Add cooldown after position flip (increase loss_cooldown_minutes)

PATTERN 2: SHORTS UNDERPERFORM
- Long win rate: {:.1f}%
- Short win rate: {:.1f}%
- RECOMMENDATION: Consider reducing short trades or requiring stronger confirmation

PATTERN 3: MANY SMALL LOSSES ADD UP
- {} small losses (<0.5%) totaling ${:.2f}
- These "death by a thousand cuts" losses indicate overtrading
- RECOMMENDATION: Increase min_signal_agreement to 2+

PATTERN 4: STOPPED OUT AT EXTREMES
- {} trades hit full 2% stop loss
- Often at trend reversals (entries at local highs/lows)
- RECOMMENDATION: Use ATR-based stops instead of fixed % or widen stops

PATTERN 5: TRADES CLUSTER AT SAME PRICE LEVELS
- Multiple entries near same price = ranging/choppy market
- Bot keeps re-entering a losing zone
- RECOMMENDATION: Increase ADX threshold to filter choppy markets
""".format(
        flip_losses/flips*100 if flips > 0 else 0,
        long_winners/len(long_trades)*100,
        short_winners/len(short_trades)*100,
        len(small_losses),
        sum(t['pnl'] for t in small_losses),
        len(stopped_out)
    ))

if __name__ == "__main__":
    analyze_trades()
