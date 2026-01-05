#!/usr/bin/env python3
"""
Quick parameter sweep to find optimal settings.
"""

import subprocess
import os

# Test configurations
configs = [
    {"MIN_ADX_FOR_TREND": "15", "LOSS_COOLDOWN_MINUTES": "45", "name": "Original"},
    {"MIN_ADX_FOR_TREND": "18", "LOSS_COOLDOWN_MINUTES": "90", "name": "Conservative"},
    {"MIN_ADX_FOR_TREND": "20", "LOSS_COOLDOWN_MINUTES": "120", "name": "Very Conservative"},
    {"MIN_ADX_FOR_TREND": "15", "LOSS_COOLDOWN_MINUTES": "120", "name": "Original ADX + Long Cooldown"},
]

print("=" * 70)
print("PARAMETER OPTIMIZATION SWEEP")
print("=" * 70)

for config in configs:
    # Set environment variables
    env = os.environ.copy()
    env.update(config)
    
    print(f"\n>>> Testing: {config['name']}")
    print(f"    ADX: {config['MIN_ADX_FOR_TREND']}, Cooldown: {config['LOSS_COOLDOWN_MINUTES']}min")
    
    # Run backtest
    result = subprocess.run(
        ["python3", "run_backtest.py", "--days", "30"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/Users/kasyap/Documents/projects/delta-exchange"
    )
    
    # Extract key metrics
    output = result.stdout
    for line in output.split('\n'):
        if "Total Return:" in line or "Win Rate:" in line or "Total Trades:" in line or "Profit Factor:" in line:
            print(f"    {line.strip()}")

print("\n" + "=" * 70)
print("SWEEP COMPLETE")
print("=" * 70)
