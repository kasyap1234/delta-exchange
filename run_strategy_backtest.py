#!/usr/bin/env python3
"""
Strategy Backtest Runner - CLI for comprehensive strategy testing.

Usage:
    python run_strategy_backtest.py                      # Run all strategies, 60 days
    python run_strategy_backtest.py --days 30            # Last 30 days
    python run_strategy_backtest.py --strategy hedging   # Only correlated hedging
    python run_strategy_backtest.py --output report.txt  # Save report
"""

import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_fetcher import HistoricalDataFetcher
from src.backtesting.strategy_backtest import StrategyBacktestRunner, StrategyBacktestResult
from src.delta_client import DeltaExchangeClient
from config.settings import settings
from utils.logger import log


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║      STRATEGY BACKTEST RUNNER - Comprehensive Testing            ║
╠══════════════════════════════════════════════════════════════════╣
║  Tests actual strategy logic against historical data             ║
║  Simulates: Correlated Hedging, Multi-Timeframe, Funding Arb     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_result(result: StrategyBacktestResult):
    """Print formatted backtest result."""
    print(f"\n{'='*60}")
    print(f" {result.strategy_name.upper()} BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"\n  Period: {result.start_date} to {result.end_date}")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Bars Processed: {result.bars_processed:,}")
    
    print(f"\n  {'─'*40}")
    print(f"  CAPITAL")
    print(f"  {'─'*40}")
    print(f"  Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"  Final Capital:    ${result.final_capital:,.2f}")
    print(f"  Total P&L:        ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
    
    print(f"\n  {'─'*40}")
    print(f"  SIGNALS & TRADES")
    print(f"  {'─'*40}")
    print(f"  Signals Generated: {result.signals_generated}")
    print(f"  Signals Executed:  {result.signals_executed}")
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Winning Trades:    {result.winning_trades}")
    print(f"  Losing Trades:     {result.losing_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    
    print(f"\n  {'─'*40}")
    print(f"  RISK METRICS")
    print(f"  {'─'*40}")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    
    # Trade log for non-funding strategies
    if result.trades and result.strategy_name != "Funding Arbitrage":
        print(f"\n  {'─'*40}")
        print(f"  RECENT TRADES (Last 10)")
        print(f"  {'─'*40}")
        for trade in result.trades[-10:]:
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(f"  {emoji} {trade.symbol} {trade.direction.upper()}: "
                  f"${trade.pnl:+.2f} ({trade.exit_reason})")


def run_backtest(
    days: int,
    capital: float,
    leverage: int,
    strategy: str,
    symbols: List[str]
) -> Dict[str, StrategyBacktestResult]:
    """Run the strategy backtest."""
    
    print(f"\n📊 Strategy Backtest Configuration")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Period: Last {days} days")
    print(f"   Capital: ${capital:,.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Strategy: {strategy}")
    print("-" * 60)
    
    # Initialize client and fetcher
    print(f"\n📥 Fetching historical data...")
    client = DeltaExchangeClient()
    fetcher = HistoricalDataFetcher(client)
    
    # Fetch data
    data_dict = fetcher.fetch_multiple(
        symbols=symbols,
        resolution='15m',
        days_back=days
    )
    
    if not data_dict:
        print("❌ Failed to fetch data")
        return {}
    
    for symbol, data in data_dict.items():
        print(f"   ✅ {symbol}: {len(data.bars)} bars")
    
    # Run backtest
    print(f"\n⚙️  Running strategy backtest...")
    runner = StrategyBacktestRunner(
        initial_capital=capital,
        leverage=leverage,
        commission_pct=0.0006,
        slippage_pct=0.0001
    )
    
    results = {}
    
    if strategy in ['all', 'hedging', 'correlated']:
        print("   Running Correlated Hedging...")
        results['correlated_hedging'] = runner.run_correlated_hedging(data_dict)
    
    if strategy in ['all', 'mtf', 'multi-timeframe']:
        print("   Running Multi-Timeframe...")
        results['multi_timeframe'] = runner.run_multi_timeframe(data_dict)
    
    if strategy in ['all', 'funding', 'arbitrage']:
        print("   Running Funding Arbitrage...")
        results['funding_arbitrage'] = runner.run_funding_arbitrage(data_dict)
    
    if strategy == 'all':
        # Calculate combined
        total_pnl = sum(r.total_pnl for r in results.values())
        total_trades = sum(r.total_trades for r in results.values())
        total_wins = sum(r.winning_trades for r in results.values())
        
        results['combined'] = StrategyBacktestResult(
            strategy_name="Combined Portfolio",
            symbols=symbols,
            start_date=list(results.values())[0].start_date,
            end_date=list(results.values())[0].end_date,
            initial_capital=capital,
            final_capital=capital + total_pnl,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / capital) * 100,
            total_trades=total_trades,
            winning_trades=total_wins,
            losing_trades=total_trades - total_wins,
            win_rate=(total_wins / total_trades * 100) if total_trades > 0 else 0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            bars_processed=list(results.values())[0].bars_processed
        )
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Strategy Backtest Runner")
    
    parser.add_argument('--days', '-d', type=int, default=60,
                        help='Number of days to backtest (default: 60)')
    parser.add_argument('--capital', '-c', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--leverage', '-l', type=int, default=5,
                        help='Leverage (default: 5)')
    parser.add_argument('--strategy', '-s', type=str, default='all',
                        choices=['all', 'hedging', 'mtf', 'funding'],
                        help='Strategy to test (default: all)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbols (default: from settings)')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = settings.trading.trading_pairs
    
    try:
        results = run_backtest(
            days=args.days,
            capital=args.capital,
            leverage=args.leverage,
            strategy=args.strategy,
            symbols=symbols
        )
        
        # Print results
        for name, result in results.items():
            print_result(result)
        
        # Summary
        if 'combined' in results:
            combined = results['combined']
            print(f"\n{'='*60}")
            print(f" PORTFOLIO SUMMARY")
            print(f"{'='*60}")
            print(f"\n  Initial Capital: ${combined.initial_capital:,.2f}")
            print(f"  Final Capital:   ${combined.final_capital:,.2f}")
            print(f"  Total Return:    {combined.total_pnl_pct:+.2f}%")
            print(f"  Total Trades:    {combined.total_trades}")
            print(f"  Win Rate:        {combined.win_rate:.1f}%")
        
        # Save to file
        if args.output:
            with open(args.output, 'w') as f:
                for name, result in results.items():
                    f.write(f"{result.strategy_name}\n")
                    for k, v in result.to_dict().items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")
            print(f"\n📁 Results saved to {args.output}")
        
        print("\n✅ Backtest complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
