#!/usr/bin/env python3
"""
Backtest Runner - CLI interface for running backtests.

Usage:
    python run_backtest.py                      # Run with defaults (30 days, all pairs)
    python run_backtest.py --days 60            # Backtest last 60 days
    python run_backtest.py --symbol BTCUSD     # Backtest single symbol
    python run_backtest.py --capital 5000       # Start with $5000
    python run_backtest.py --leverage 10        # Use 10x leverage
    python run_backtest.py --output report.txt  # Save report to file
"""

import argparse
import sys
import os
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.data_fetcher import HistoricalDataFetcher
from src.backtesting.backtest_engine import BacktestEngine, MultiStrategyBacktest
from src.backtesting.performance import PerformanceAnalyzer, ReportGenerator
from src.delta_client import DeltaExchangeClient
from config.settings import settings
from utils.logger import log


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         DELTA EXCHANGE TRADING BOT - BACKTESTING ENGINE          ║
╠══════════════════════════════════════════════════════════════════╣
║  Simulates trading strategies on historical data                 ║
║  Calculates P&L, risk metrics, and performance statistics        ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_single_symbol_backtest(
    symbol: str, days: int, capital: float, leverage: int, resolution: str
):
    """Run backtest for a single symbol."""
    print(f"\n📊 Running backtest for {symbol}")
    print(f"   Period: Last {days} days")
    print(f"   Capital: ${capital:,.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Resolution: {resolution}")
    print("-" * 60)

    # Initialize components
    client = DeltaExchangeClient()
    fetcher = HistoricalDataFetcher(client)

    # Fetch data
    print(f"\n📥 Fetching historical data for {symbol}...")
    data = fetcher.fetch(symbol=symbol, resolution=resolution, days_back=days)

    if not data.bars:
        print(f"❌ No data available for {symbol}")
        return None, None

    print(f"   ✅ Loaded {len(data.bars)} bars")
    print(f"   Date range: {data.start_date} to {data.end_date}")

    # Run backtest
    print(f"\n⚙️  Running backtest simulation...")
    engine = BacktestEngine(initial_capital=capital, leverage=leverage)

    result = engine.run(data, strategy="technical")

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(result.equity_curve, result.trades, days)

    # Generate and print report
    report = ReportGenerator.generate_summary(result, metrics)
    print(report)

    # Print trade log if trades exist
    if result.trades:
        print("\n" + ReportGenerator.generate_trade_log(result.trades))

    return result, metrics


def run_multi_symbol_backtest(
    symbols: List[str], days: int, capital: float, leverage: int, resolution: str
):
    """Run backtest for multiple symbols."""
    print(f"\n📊 Running multi-symbol backtest")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Period: Last {days} days")
    print(f"   Capital: ${capital:,.2f}")
    print(f"   Leverage: {leverage}x")
    print("-" * 60)

    # Initialize components
    client = DeltaExchangeClient()
    fetcher = HistoricalDataFetcher(client)

    # Fetch data for all symbols
    print(f"\n📥 Fetching historical data...")
    data_dict = fetcher.fetch_multiple(
        symbols=symbols, resolution=resolution, days_back=days
    )

    if not data_dict:
        print("❌ No data available")
        return None

    for symbol, data in data_dict.items():
        print(f"   ✅ {symbol}: {len(data.bars)} bars")

    multi_bt = MultiStrategyBacktest(
        initial_capital=capital,
        allocation={
            "funding_arbitrage": settings.strategy_allocation.funding_arbitrage,
            "correlated_hedging": settings.strategy_allocation.correlated_hedging,
            "multi_timeframe": settings.strategy_allocation.multi_timeframe,
        },
    )

    results = multi_bt.run(data_dict)

    # Generate combined report
    report = ReportGenerator.generate_combined_report(results)
    print(report)

    # Detailed per-symbol analysis
    print("\n" + "=" * 70)
    print("DETAILED PER-SYMBOL ANALYSIS")
    print("=" * 70)

    analyzer = PerformanceAnalyzer()

    for strategy, strategy_results in results.items():
        if strategy in ["combined", "funding_arbitrage"]:
            continue

        if isinstance(strategy_results, dict):
            for symbol, result in strategy_results.items():
                if hasattr(result, "equity_curve"):
                    metrics = analyzer.analyze(result.equity_curve, result.trades, days)
                    summary = ReportGenerator.generate_summary(result, metrics)
                    print(f"\n{summary}")

    return results


def run_quick_backtest(days: int = 30, capital: float = 10000.0) -> None:
    """Run a quick backtest with sensible defaults."""
    print_banner()

    symbols = settings.trading.trading_pairs
    leverage = settings.trading.leverage
    resolution = settings.trading.candle_interval

    print(f"\n🚀 Quick Backtest Mode")
    print(f"   Using settings from config:")
    print(f"   - Symbols: {symbols}")
    print(f"   - Leverage: {leverage}x")
    print(f"   - Resolution: {resolution}")

    run_multi_symbol_backtest(
        symbols=symbols,
        days=days,
        capital=capital,
        leverage=leverage,
        resolution=resolution,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Delta Exchange Trading Bot Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                         # Quick test, 30 days
  python run_backtest.py --days 60               # Last 60 days
  python run_backtest.py --symbol BTCUSD        # Single symbol
  python run_backtest.py --capital 5000          # $5000 starting capital
  python run_backtest.py --all                   # All configured pairs
  python run_backtest.py --output results.txt   # Save to file
        """,
    )

    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=30,
        help="Number of days to backtest (default: 30)",
    )

    parser.add_argument(
        "--symbol", "-s", type=str, help="Single symbol to backtest (e.g., BTCUSD)"
    )

    parser.add_argument(
        "--all", "-a", action="store_true", help="Backtest all configured trading pairs"
    )

    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000)",
    )

    parser.add_argument(
        "--leverage",
        "-l",
        type=int,
        default=None,
        help="Leverage multiplier (default: from settings)",
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=None,
        help="Candle resolution (default: from settings, e.g., 15m, 1h, 4h)",
    )

    parser.add_argument("--output", "-o", type=str, help="Output file for results")

    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if not args.quiet:
        print_banner()

    # Get settings
    leverage = args.leverage or settings.trading.leverage
    resolution = args.resolution or settings.trading.candle_interval

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.all:
        symbols = settings.trading.trading_pairs
    else:
        symbols = settings.trading.trading_pairs

    # Run backtest
    result = None
    metrics = None
    results = None

    try:
        if len(symbols) == 1:
            backtest_output = run_single_symbol_backtest(
                symbol=symbols[0],
                days=args.days,
                capital=args.capital,
                leverage=leverage,
                resolution=resolution,
            )
            if backtest_output:
                result, metrics = backtest_output
        else:
            results = run_multi_symbol_backtest(
                symbols=symbols,
                days=args.days,
                capital=args.capital,
                leverage=leverage,
                resolution=resolution,
            )

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                if len(symbols) == 1 and result and metrics:
                    f.write(ReportGenerator.generate_summary(result, metrics))
                elif results:
                    f.write(ReportGenerator.generate_combined_report(results))
            print(f"\n📁 Results saved to {args.output}")

        print("\n✅ Backtest complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
