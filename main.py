#!/usr/bin/env python3
"""
Delta Exchange Automated Trading Bot (Async Event-Driven)
Main entry point for the trading bot background service.

This version uses an event-driven architecture with WebSockets
for real-time responsiveness and improved reliability in the India region.

Usage:
    python main.py                    # Run with default settings
    python main.py --testnet          # Use testnet environment
    python main.py --dry-run          # Log trades without executing

Configuration:
    Copy .env.example to .env and fill in your API credentials.
"""

import argparse
import asyncio
import signal
import sys
from typing import Optional

from config.settings import settings
from src.strategies.async_strategy_manager import AsyncStrategyManager
from utils.logger import log


class AsyncTradingBot:
    """
    Main Orchestrator for Event-Driven Trading.
    
    Wraps the AsyncStrategyManager and handles signal interruption
    and high-level lifecycle.
    """
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.manager: Optional[AsyncStrategyManager] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Initialize and start the bot."""
        log.info("=" * 60)
        log.info("Delta Exchange Async Bot Starting")
        log.info(f"Region: {settings.delta.region}")
        log.info("=" * 60)
        
        # Verify settings
        if not settings.delta.validate():
            log.error("Invalid API credentials. Check .env")
            sys.exit(1)
            
        # Initialize Manager
        self.manager = AsyncStrategyManager(
            api_key=settings.delta.api_key,
            api_secret=settings.delta.api_secret,
            region=settings.delta.region,
            dry_run=self.dry_run
        )
        
        # Setup Signal Handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            
        try:
            # Run Manager
            await self.manager.start()
            
            # Keep alive until shutdown signal
            await self._shutdown_event.wait()
            
        except asyncio.CancelledError:
            log.info("Bot task cancelled")
        except Exception as e:
            log.critical(f"Bot crashed: {e}")
            import traceback
            log.error(traceback.format_exc())
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown sequence."""
        log.info("Shutting down bot...")
        if self.manager:
            self.manager.stop()
        self._shutdown_event.set()
        
        # Allow time for pending tasks to cancel
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Shutdown complete.")
        sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Delta Exchange Async Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    return parser.parse_args()


# Wrapper for asyncio entry point
def main():
    args = parse_args()
    
    if args.testnet:
        settings.delta.environment = 'testnet'
        log.info("Using testnet environment")
        
    bot = AsyncTradingBot(dry_run=args.dry_run)
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        # Handled by signal handler, this is just a fallback
        pass

if __name__ == '__main__':
    main()
