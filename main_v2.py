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
    python main_v2.py --paper-trade      # Paper trade with P&L tracking
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
from src.strategies.base_strategy import StrategyType, SignalDirection
from src.paper_trader import PaperTradingSimulator
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
    
    def __init__(self, dry_run: bool = False, paper_trade: bool = False, 
                 interval_seconds: int = 300):
        """
        Initialize the multi-strategy bot.
        
        Args:
            dry_run: If True, log trades but don't execute
            paper_trade: If True, simulate trades with P&L tracking
            interval_seconds: Interval between analysis cycles
        """
        self.dry_run = dry_run
        self.paper_trade = paper_trade
        self.interval_seconds = interval_seconds
        self.running = False
        
        # Components
        self.client: Optional[DeltaExchangeClient] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.scheduler: Optional[BlockingScheduler] = None
        self.paper_simulator: Optional[PaperTradingSimulator] = None
        
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
        log.info(f"Paper Trade: {self.paper_trade}")
        
        # Initialize paper trading simulator if enabled
        if self.paper_trade:
            # Get wallet balance for initial virtual balance
            from src.utils.balance_utils import get_usd_balance
            try:
                if hasattr(self, 'client') and self.client:
                    initial_balance = get_usd_balance(self.client)
                    if initial_balance == 0:
                        initial_balance = 200.0  # Default if balance not found
                else:
                    initial_balance = 200.0  # Default if client not available
            except Exception as e:
                log.warning(f"Could not get balance for paper trading: {e}, using default 200.0")
                initial_balance = 200.0
            
            # Get leverage from settings
            leverage = getattr(settings.trading, 'leverage', 5)
            
            self.paper_simulator = PaperTradingSimulator(
                initial_balance=initial_balance,
                leverage=leverage,
                use_maker_fee=True  # Use limit orders (maker fee 0.04% vs taker 0.06%)
            )
            log.info(f"Paper Trading enabled with ${initial_balance:.2f} @ {leverage}x leverage")
        
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
            if self.paper_trade or self.dry_run:
                log.warning("Private API connection failed, but proceeding with Paper Trading mode")
                if not self.client.test_public_connection():
                    log.error("Failed to connect to Delta Exchange Market Data API")
                    return False
                log.info("Public API connection OK - Running in Unauthenticated Mode")
            else:
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
            # Paper Trading position override
            positions_override = None
            if self.paper_trade and self.paper_simulator:
                positions_override = self.paper_simulator.get_positions_as_delta_objects()
                
            # Run the strategy manager cycle
            signals = self.strategy_manager.run_cycle(override_positions=positions_override)
            
            # Paper Trading: Process entries and exits
            if self.paper_trade and self.paper_simulator:
                # Get current prices and ATRs for all trading pairs
                prices = {}
                atrs = {}
                current_signals = {}
                for pair in settings.trading.trading_pairs:
                    try:
                        ticker = self.client.get_ticker(pair)
                        prices[pair] = float(ticker.get('mark_price', 0))
                        
                        # Fetch candles for ATR calculation (required for trailing stops)
                        candles = self.client.get_candles(symbol=pair, resolution='15m')
                        if len(candles) >= 15:
                            highs = np.array([c.high for c in candles])
                            lows = np.array([c.low for c in candles])
                            closes = np.array([c.close for c in candles])
                            atrs[pair] = self.strategy_manager.analyzer.calculate_atr(highs, lows, closes)
                            
                    except (KeyError, ValueError, TypeError, AttributeError) as e:
                        log.debug(f"Could not get data for {pair}: {e}")
                        continue
                    except Exception as e:
                        log.warning(f"Unexpected error getting data for {pair}: {e}")
                        continue
                
                # Update prices and check exit conditions with ATRs
                self.paper_simulator.update_prices(prices, atrs=atrs)
                
                # Build current signals map for reversal detection
                for signal in signals:
                    current_signals[signal.symbol] = signal.direction.value
                
                # Check and execute exits (stop-loss, take-profit, reversal)
                self.paper_simulator.process_exits(prices, current_signals)
                
                # Process new entries
                for signal in signals:
                    # Skip if already have position
                    if self.paper_simulator.has_position(signal.symbol):
                        continue
                    
                    # Only process entry signals (not exit)
                    if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
                        side = 'long' if signal.direction == SignalDirection.LONG else 'short'
                        self.paper_simulator.open_position(
                            symbol=signal.symbol,
                            side=side,
                            size=signal.position_size,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss if signal.stop_loss else signal.entry_price * 0.98,
                            take_profit=signal.take_profit if signal.take_profit else signal.entry_price * 1.04,
                            strategy=signal.strategy_type.value
                        )
                
                # Display current positions and P&L
                self.paper_simulator.print_positions()
            
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
        
        # Stop WebSocket if running
        if self.strategy_manager:
            self.strategy_manager._stop_websocket()
        
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
        
        # Paper trading summary
        if self.paper_trade and self.paper_simulator:
            self.paper_simulator.print_summary()


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
  python main_v2.py --paper-trade      # Simulate with P&L tracking
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
        default=120,
        help='Interval between trading cycles in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy single-strategy mode (original bot)'
    )
    
    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Paper trade with virtual P&L tracking (includes --dry-run)'
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
    # Paper trade implies dry-run (no real trades)
    dry_run = args.dry_run or args.paper_trade
    
    bot = MultiStrategyBot(
        dry_run=dry_run,
        paper_trade=args.paper_trade,
        interval_seconds=args.interval
    )
    
    if args.once:
        bot.run_once()
    else:
        bot.run_continuous()


if __name__ == '__main__':
    main()
