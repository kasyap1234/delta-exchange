#!/usr/bin/env python3
"""
Delta Exchange Multi-Strategy Automated Trading Bot
Main entry point for the trading bot with 3-tier strategy system.

Strategies:
    Tier 1: Delta-Neutral Funding Rate Arbitrage (40% capital)
    Tier 2: Correlated Pair Hedging (40% capital)
    Tier 3: Multi-Timeframe Trend Following (20% capital)

Usage:
    python main_v2.py                    # Run with default settings
    python main_v2.py --testnet          # Use testnet environment
    python main_v2.py --dry-run          # Log trades without executing
    python main_v2.py --once             # Run analysis once and exit
    python main_v2.py --interval 300     # Custom interval in seconds
    python main_v2.py --legacy           # Use legacy single-strategy mode

Configuration:
    Copy .env.example to .env and fill in your API credentials.
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from src.delta_client import DeltaExchangeClient
from src.strategies.strategy_manager import StrategyManager, TradingState
from src.strategies.base_strategy import StrategyType
from utils.logger import log


class MultiStrategyBot:
    """
    Multi-strategy trading bot orchestrator.
    
    Manages three strategy tiers:
    - Tier 1: Funding Rate Arbitrage (safe, passive income)
    - Tier 2: Correlated Pair Hedging (medium risk)
    - Tier 3: Multi-Timeframe Trend (higher risk)
    
    Features:
    - Automatic capital allocation
    - Daily loss limits
    - Graceful shutdown
    - Performance tracking
    """
    
    def __init__(self, dry_run: bool = False, interval_seconds: int = 300):
        """
        Initialize the multi-strategy bot.
        
        Args:
            dry_run: If True, log trades but don't execute
            interval_seconds: Interval between analysis cycles
        """
        self.dry_run = dry_run
        self.interval_seconds = interval_seconds
        self.running = False
        
        # Components
        self.client: Optional[DeltaExchangeClient] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.scheduler: Optional[BlockingScheduler] = None
        
        # Statistics
        self.cycle_count = 0
        self.start_time: Optional[datetime] = None
    
    def initialize(self) -> bool:
        """
        Initialize all bot components and verify connectivity.
        
        Returns:
            True if initialization successful
        """
        log.info("=" * 70)
        log.info("Delta Exchange Multi-Strategy Trading Bot v2.0")
        log.info("=" * 70)
        
        # Validate configuration
        if not settings.delta.validate():
            log.error("Invalid API credentials. Please check your .env file.")
            log.error("Copy .env.example to .env and fill in your API key and secret.")
            return False
        
        # Log configuration
        log.info(f"Environment: {settings.delta.environment}")
        log.info(f"Region: {settings.delta.region}")
        log.info(f"Base URL: {settings.delta.base_url}")
        log.info(f"Trading Pairs: {settings.trading.trading_pairs}")
        log.info(f"Dry Run: {self.dry_run}")
        
        # Log strategy allocation
        log.info("-" * 40)
        log.info("Strategy Allocation:")
        log.info(f"  Tier 1 (Funding Arbitrage): {settings.strategy_allocation.funding_arbitrage:.0%}")
        log.info(f"  Tier 2 (Correlated Hedging): {settings.strategy_allocation.correlated_hedging:.0%}")
        log.info(f"  Tier 3 (Multi-Timeframe): {settings.strategy_allocation.multi_timeframe:.0%}")
        
        # Log risk settings
        log.info("-" * 40)
        log.info("Risk Management:")
        log.info(f"  Daily Loss Limit: {settings.enhanced_risk.daily_loss_limit_pct:.1%}")
        log.info(f"  ATR Stop Multiplier: {settings.enhanced_risk.atr_stop_multiplier}x")
        log.info(f"  Trailing Stops: {settings.enhanced_risk.trailing_enabled}")
        log.info(f"  Profit Ladder: {settings.enhanced_risk.profit_ladder_enabled}")
        
        # Initialize API client
        log.info("-" * 40)
        log.info("Initializing Delta Exchange client...")
        self.client = DeltaExchangeClient()
        
        # Test connection
        log.info("Testing API connection...")
        if not self.client.test_connection():
            log.error("Failed to connect to Delta Exchange API")
            return False
        
        # Build strategy allocation
        allocation = {
            StrategyType.FUNDING_ARBITRAGE: settings.strategy_allocation.funding_arbitrage,
            StrategyType.CORRELATED_HEDGING: settings.strategy_allocation.correlated_hedging,
            StrategyType.MULTI_TIMEFRAME: settings.strategy_allocation.multi_timeframe,
        }
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager(
            client=self.client,
            allocation=allocation,
            daily_loss_limit=settings.enhanced_risk.daily_loss_limit_pct,
            dry_run=self.dry_run
        )
        
        log.info("=" * 70)
        log.info("Initialization complete!")
        log.info("=" * 70)
        return True
    
    def run_trading_cycle(self) -> None:
        """
        Execute one complete trading cycle across all strategies.
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        log.info("-" * 50)
        log.info(f"Trading Cycle #{self.cycle_count} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("-" * 50)
        
        try:
            # Run the strategy manager cycle
            signals = self.strategy_manager.run_cycle()
            
            # Log results
            if signals:
                log.info(f"Generated {len(signals)} signals")
                for signal in signals:
                    log.info(f"  {signal.strategy_type.value}: {signal.direction.value} "
                            f"{signal.symbol} @ {signal.entry_price:.2f}")
            else:
                log.info("No actionable signals - holding positions")
            
            # Log strategy status
            status = self.strategy_manager.get_status()
            log.info(f"Trading State: {status['state']}")
            
            if status['today_stats']['pnl'] != 0:
                log.info(f"Today's PnL: ${status['today_stats']['pnl']:.2f} "
                        f"({status['today_stats']['pnl_pct']})")
            
            # Cycle timing
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            log.info(f"Cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            log.error(f"Trading cycle failed: {e}")
            import traceback
            log.error(traceback.format_exc())
    
    def run_once(self) -> None:
        """Run a single trading cycle and exit."""
        if not self.initialize():
            sys.exit(1)
        
        self.start_time = datetime.now()
        self.run_trading_cycle()
        
        # Print performance summary
        self._print_summary()
        log.info("Single cycle complete. Exiting.")
    
    def run_continuous(self) -> None:
        """Run the bot continuously on a schedule."""
        if not self.initialize():
            sys.exit(1)
        
        self.start_time = datetime.now()
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Run first cycle immediately
        self.run_trading_cycle()
        
        # Set up scheduler
        log.info(f"Starting scheduler with {self.interval_seconds}s interval")
        self.scheduler = BlockingScheduler()
        
        self.scheduler.add_job(
            self.run_trading_cycle,
            trigger=IntervalTrigger(seconds=self.interval_seconds),
            id='trading_cycle',
            name='Multi-Strategy Trading Cycle',
            replace_existing=True
        )
        
        try:
            log.info("Bot is running. Press Ctrl+C to stop.")
            log.info("=" * 70)
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        log.info(f"Received signal {signum}, shutting down...")
        self._shutdown()
    
    def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        self.running = False
        
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        
        self._print_summary()
        
        log.info("Shutdown complete.")
        sys.exit(0)
    
    def _print_summary(self) -> None:
        """Print performance summary."""
        log.info("=" * 70)
        log.info("PERFORMANCE SUMMARY")
        log.info("=" * 70)
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            log.info(f"Runtime: {runtime}")
        
        log.info(f"Total cycles: {self.cycle_count}")
        
        if self.strategy_manager:
            perf = self.strategy_manager.get_performance_summary()
            
            log.info(f"Total trades: {perf['total_trades']}")
            log.info(f"Total P&L: ${perf['total_pnl']:.2f}")
            log.info(f"Funding earned: ${perf['total_funding_earned']:.2f}")
            log.info(f"Combined return: ${perf['combined_return']:.2f}")
            
            log.info("-" * 40)
            log.info("Per-Strategy Performance:")
            for strategy_name, stats in perf['strategies'].items():
                log.info(f"  {strategy_name}:")
                log.info(f"    Trades: {stats['trades']}, Win Rate: {stats['win_rate']}")
                log.info(f"    P&L: ${stats['pnl']:.2f}, Funding: ${stats['funding']:.2f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Delta Exchange Multi-Strategy Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  Tier 1: Funding Rate Arbitrage - Delta-neutral, earns from funding
  Tier 2: Correlated Hedging - Trades with 30% hedge using BTC/ETH
  Tier 3: Multi-Timeframe - 4H trend + 15m entry confirmation

Examples:
  python main_v2.py --dry-run          # Test without executing trades
  python main_v2.py --testnet          # Use testnet environment
  python main_v2.py --once             # Run once and exit
  python main_v2.py --interval 60      # Run every 60 seconds

Configuration:
  Copy .env.example to .env and set your API credentials.
  Adjust strategy allocation in .env (ALLOC_FUNDING_ARB, etc.)
        """
    )
    
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Use testnet environment (overrides .env setting)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log trades without executing them'
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one trading cycle and exit'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Interval between trading cycles in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy single-strategy mode (original bot)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Override environment if testnet flag is set
    if args.testnet:
        settings.delta.environment = 'testnet'
        log.info("Using testnet environment (command line override)")
    
    # Check for legacy mode
    if args.legacy:
        log.info("Running in legacy single-strategy mode...")
        # Import and run the original main
        from main import main as legacy_main
        legacy_main()
        return
    
    # Create and run multi-strategy bot
    bot = MultiStrategyBot(
        dry_run=args.dry_run,
        interval_seconds=args.interval
    )
    
    if args.once:
        bot.run_once()
    else:
        bot.run_continuous()


if __name__ == '__main__':
    main()
