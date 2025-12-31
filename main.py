#!/usr/bin/env python3
"""
Delta Exchange Automated Trading Bot
Main entry point for the trading bot background service.

Usage:
    python main.py                    # Run with default settings
    python main.py --testnet          # Use testnet environment
    python main.py --dry-run          # Log trades without executing
    python main.py --once             # Run analysis once and exit
    python main.py --interval 300     # Custom interval in seconds

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
from src.strategy import TradingStrategy, TradeAction
from src.trader import TradeExecutor
from utils.logger import log


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Handles:
    - Initialization and authentication
    - Scheduled market analysis
    - Trade execution
    - Graceful shutdown
    """
    
    def __init__(self, dry_run: bool = False, interval_seconds: int = 300):
        """
        Initialize the trading bot.
        
        Args:
            dry_run: If True, log trades but don't execute
            interval_seconds: Interval between analysis cycles
        """
        self.dry_run = dry_run
        self.interval_seconds = interval_seconds
        self.running = False
        
        # Initialize components
        self.client: Optional[DeltaExchangeClient] = None
        self.strategy: Optional[TradingStrategy] = None
        self.executor: Optional[TradeExecutor] = None
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
        log.info("=" * 60)
        log.info("Delta Exchange Trading Bot Starting")
        log.info("=" * 60)
        
        # Validate configuration
        if not settings.delta.validate():
            log.error("Invalid API credentials. Please check your .env file.")
            log.error("Copy .env.example to .env and fill in your API key and secret.")
            return False
        
        log.info(f"Environment: {settings.delta.environment}")
        log.info(f"Region: {settings.delta.region}")
        log.info(f"Base URL: {settings.delta.base_url}")
        log.info(f"Trading Pairs: {settings.trading.trading_pairs}")
        log.info(f"Candle Interval: {settings.trading.candle_interval}")
        log.info(f"Max Capital Per Trade: {settings.trading.max_capital_per_trade * 100}%")
        log.info(f"Stop Loss: {settings.trading.stop_loss_pct * 100}%")
        log.info(f"Take Profit: {settings.trading.take_profit_pct * 100}%")
        log.info(f"Dry Run: {self.dry_run}")
        
        # Initialize API client
        log.info("Initializing Delta Exchange client...")
        self.client = DeltaExchangeClient()
        
        # Test connection
        log.info("Testing API connection...")
        if not self.client.test_connection():
            log.error("Failed to connect to Delta Exchange API")
            return False
        
        # Initialize executor FIRST, then pass to strategy
        self.executor = TradeExecutor(self.client, dry_run=self.dry_run)
        self.strategy = TradingStrategy(self.client, self.executor, dry_run=self.dry_run)
        
        log.info("Initialization complete!")
        return True
    
    def run_trading_cycle(self) -> None:
        """
        Execute one complete trading cycle.
        
        1. Fetch account balance
        2. Fetch open positions
        3. Analyze all trading pairs
        4. Execute actionable decisions
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        log.info("-" * 40)
        log.info(f"Trading Cycle #{self.cycle_count} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("-" * 40)
        
        try:
            # Get account balance
            from src.utils.balance_utils import get_usd_balance
            available_balance = get_usd_balance(self.client)
            log.info(f"Available Balance: ${available_balance:.2f}")
            
            # Get open positions
            positions = self.client.get_positions()
            log.info(f"Open Positions: {len(positions)}")
            
            for pos in positions:
                log.info(f"  {pos.product_symbol}: {pos.size} @ {pos.entry_price:.2f} "
                         f"(PnL: {pos.unrealized_pnl:.2f})")
            
            # Check for hedging opportunities (auto-protect losing positions)
            log.info("Evaluating hedging opportunities...")
            new_hedges = self.strategy.evaluate_hedging_opportunities(positions)
            if new_hedges:
                log.info(f"Created {len(new_hedges)} new hedge position(s)")
                for hedge in new_hedges:
                    log.info(f"  {hedge.primary_side.upper()} {hedge.primary_symbol} "
                             f"hedged with {hedge.hedge_side.upper()} {hedge.hedge_symbol} "
                             f"(ratio: {hedge.hedge_ratio:.0%})")
            
            # Log hedge status
            hedge_status = self.strategy.get_hedge_status()
            if hedge_status['active_positions'] > 0:
                log.info(f"Active hedged positions: {hedge_status['active_positions']}, "
                         f"Net exposure: ${hedge_status['total_exposure']:.2f}")
            
            # Analyze all pairs
            log.info("Analyzing markets...")
            decisions = self.strategy.analyze_all_pairs(available_balance, positions)
            
            # Get actionable decisions
            actionable = self.strategy.get_actionable_decisions(decisions)
            
            if actionable:
                log.info(f"Actionable decisions: {len(actionable)}")
                for decision in actionable:
                    self.executor.execute_decision(decision)
            else:
                log.info("No actionable signals - holding positions")
            
            # Log cycle summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            log.info(f"Cycle completed in {cycle_duration:.2f}s")
            
            # Log trade stats
            stats = self.executor.get_trade_stats()
            log.info(f"Session Stats: {stats['total_trades']} trades "
                     f"({stats['buy_trades']} buys, {stats['sell_trades']} sells)")
            
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
            name='Main Trading Cycle',
            replace_existing=True
        )
        
        try:
            log.info("Bot is running. Press Ctrl+C to stop.")
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
        
        # Log final stats
        if self.start_time:
            runtime = datetime.now() - self.start_time
            log.info(f"Bot ran for {runtime}")
        
        log.info(f"Total cycles: {self.cycle_count}")
        
        if self.executor:
            stats = self.executor.get_trade_stats()
            log.info(f"Total trades: {stats['total_trades']}")
        
        log.info("Shutdown complete.")
        sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Delta Exchange Automated Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dry-run          # Test without executing trades
  python main.py --testnet          # Use testnet environment
  python main.py --once             # Run once and exit
  python main.py --interval 60      # Run every 60 seconds

Configuration:
  Copy .env.example to .env and set your API credentials.
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Override environment if testnet flag is set
    if args.testnet:
        settings.delta.environment = 'testnet'
        log.info("Using testnet environment (command line override)")
    
    # Create and run bot
    bot = TradingBot(
        dry_run=args.dry_run,
        interval_seconds=args.interval
    )
    
    if args.once:
        bot.run_once()
    else:
        bot.run_continuous()


if __name__ == '__main__':
    main()
