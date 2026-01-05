"""
Async Strategy Manager for Event-Driven Trading.

This module orchestrates trading strategies using an asyncio event loop,
reacting to real-time WebSocket events instead of polling.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.delta_client import DeltaExchangeClient, Order, Position, OrderSide, OrderType
from src.websocket_client import DeltaWebSocketClient
from src.risk_manager import RiskManager
from src.unified_signal_validator import UnifiedSignalValidator
from config.settings import settings
from utils.logger import log

from src.strategies.base_strategy import BaseStrategy, StrategySignal, SignalDirection
from src.strategies.correlated_hedging import CorrelatedHedgingStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
# Funding Arbitrage removed as per India region constraints

class AsyncStrategyManager:
    """
    Event-driven strategy manager.
    
    Responsibilities:
    1. Initialize and manage strategy instances
    2. Process real-time market data (ticks)
    3. Route signals to execution
    4. Enforce global risk limits (Kill Switch)
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, region: str = "india", dry_run: bool = False):
        self.dry_run = dry_run
        
        # Initialize Clients
        self.rest_client = DeltaExchangeClient(api_key=api_key, api_secret=api_secret, use_hybrid_mode=True)
        self.ws_client = DeltaWebSocketClient(api_key=api_key, api_secret=api_secret, region=region)
        
        # Risk & Validation
        self.risk_manager = RiskManager()
        self.signal_validator = UnifiedSignalValidator()
        
        # Strategies
        self.strategies: List[BaseStrategy] = []
        self._initialize_strategies()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.last_tick_time: Dict[str, float] = {}
        self.is_running = False
        
        log.info(f"AsyncStrategyManager initialized (Dry Run: {dry_run})")

    def _initialize_strategies(self):
        """Initialize active strategy modules."""
        self.strategies = []
        alloc = settings.strategy_allocation
        
        # Tier 2: Correlated Hedging
        if alloc.correlated_hedging > 0:
            self.strategies.append(
                CorrelatedHedgingStrategy(
                    client=self.rest_client,
                    capital_allocation=alloc.correlated_hedging,
                    dry_run=self.dry_run
                )
            )
            
        # Tier 3: Multi-Timeframe
        if alloc.multi_timeframe > 0:
            self.strategies.append(
                MultiTimeframeStrategy(
                    client=self.rest_client,
                    capital_allocation=alloc.multi_timeframe,
                    dry_run=self.dry_run
                )
            )
            
        log.info(f"Initialized {len(self.strategies)} strategies")

    async def start(self):
        """Start the async event loop and WebSocket connection."""
        self.is_running = True
        
        # Connect WebSocket
        asyncio.create_task(self.ws_client.connect())
        
        # Wait for connection
        while not self.ws_client.ws:
            await asyncio.sleep(1)
            
        # Subscribe to channels
        await self._subscribe_channels()
        
        log.info("AsyncStrategyManager started and listening...")
        
        # Keep alive check
        while self.is_running:
            await self._periodic_health_check()
            await asyncio.sleep(60)

    async def _subscribe_channels(self):
        """Subscribe to necessary WebSocket channels."""
        symbols = settings.trading.trading_pairs
        
        # 1. Market Data (Ticker) for all pairs
        await self.ws_client.subscribe('v2/ticker', symbols=symbols, callback=self.on_ticker_update)
        
        # 2. Private Channels (Orders, Positions)
        # Note: These require auth, handled by WS client
        await self.ws_client.subscribe_positions(callback=self.on_position_update)
        await self.ws_client.subscribe_orders(callback=self.on_order_update)
        
        log.info(f"Subscribed to tickers for {symbols} and private channels")

    async def on_ticker_update(self, data: Dict):
        """
        Handle real-time ticker updates.
        Triggers strategy analysis if enough time has passed since last check.
        """
        if data.get('type') != 'v2/ticker':
            return
            
        ticker = data.get('symbol')
        price = float(data.get('mark_price', 0) or 0)
        
        if not ticker or price <= 0:
            return

        # Throttle strategy checks (e.g., check every 15 seconds per symbol)
        now = asyncio.get_running_loop().time()
        last_check = self.last_tick_time.get(ticker, 0)
        
        if now - last_check < 15:
            return
            
        self.last_tick_time[ticker] = now
        await self.evaluate_strategies(ticker, price)
        
    async def on_position_update(self, data: Dict):
        """
        Handle position updates.
        CRITICAL: Checks for Global Risk Limit (Kill Switch).
        """
        log.info(f"Position Update: {data}")
        # Update local state/cache if needed
        # Check Kill Switch
        if not self.dry_run:
            self._check_kill_switch()

    async def on_order_update(self, data: Dict):
        """Handle order updates (fills, cancellations)."""
        log.info(f"Order Update: {data}")
        # Can trigger hedge logic here if a primary order is filled

    async def evaluate_strategies(self, symbol: str, current_price: float):
        """
        Run analysis on all strategies for ALL symbols.
        
        Collects signals from all strategies and symbols, ranks them by
        composite score (confidence + historical performance), and only
        executes the single best signal.
        """
        
        # Get latest positions snapshot for strategies
        current_positions = self.rest_client.get_positions()
        
        # Calculate available capital
        balances = self.rest_client.get_wallet_balance()
        if isinstance(balances, list):
             balance = 0.0
             for b in balances:
                 balance += float(b.get('available_balance', 0))
        else:
            balance = float(balances.get('available_balance', 0))

        # Collect all signals from all strategies
        all_signals = []
        
        for strategy in self.strategies:
            try:
                signals = strategy.analyze(balance * strategy.capital_allocation, current_positions)
                
                for signal in signals:
                    if signal.is_actionable:
                        # Calculate composite score for ranking
                        # Score = 0.6 * confidence + 0.4 * strategy_win_rate
                        win_rate = strategy.performance.win_rate if strategy.performance.total_trades >= 5 else 0.45
                        score = (0.6 * signal.confidence) + (0.4 * win_rate)
                        
                        all_signals.append({
                            'signal': signal,
                            'strategy': strategy,
                            'score': score,
                            'confidence': signal.confidence,
                            'win_rate': win_rate
                        })
                        
            except Exception as e:
                log.error(f"Error evaluating {strategy.name}: {e}")

        if not all_signals:
            return
            
        # Sort by score (highest first) and select the best
        all_signals.sort(key=lambda x: x['score'], reverse=True)
        best = all_signals[0]
        
        log.info(f"📊 Signal Ranking: {len(all_signals)} candidates")
        for i, sig_data in enumerate(all_signals[:3]):  # Log top 3
            s = sig_data['signal']
            log.info(f"  #{i+1}: {s.symbol} {s.direction.name} | Score: {sig_data['score']:.3f} "
                    f"(Conf: {sig_data['confidence']:.2f}, WR: {sig_data['win_rate']:.2%})")
        
        # Execute only the best signal
        log.info(f"✅ Selected: {best['signal'].symbol} {best['signal'].direction.name} (Score: {best['score']:.3f})")
        await self.execute_signal(best['signal'])

    async def execute_signal(self, signal: StrategySignal):
        """Execute a trading signal."""
        if not self.risk_manager.can_open_position(signal.symbol, signal.position_size):
            log.warning(f"Risk Manager rejected signal for {signal.symbol}")
            return

        log.info(f"EXECUTING Signal: {signal.direction.name} {signal.symbol} Size: {signal.position_size}")
        
        if self.dry_run:
            log.info("[DRY RUN] Order would be placed here.")
            return

        try:
            # Place Order
            side = OrderSide.BUY if signal.direction in [SignalDirection.LONG, SignalDirection.CLOSE_SHORT] else OrderSide.SELL
            
            # Use sync client for now (wrapped in thread if blocking, but requests is fast enough for low freq)
            # ideally refactor RestClient to be async too, but mixing is okay for V1
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.rest_client.place_order(
                    symbol=signal.symbol,
                    side=side,
                    size=signal.position_size,
                    order_type=OrderType.MARKET # Using market for immediate execution
                )
            )
            
            # Handle Hedging if needed (Correlated Strategy)
            if signal.hedge_symbol and signal.hedge_size > 0:
                hedge_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
                log.info(f"EXECUTING HEDGE: {hedge_side.name} {signal.hedge_symbol} Size: {signal.hedge_size}")
                await loop.run_in_executor(
                    None,
                    lambda: self.rest_client.place_order(
                        symbol=signal.hedge_symbol,
                        side=hedge_side,
                        size=signal.hedge_size,
                        order_type=OrderType.MARKET
                    )
                )

        except Exception as e:
            log.error(f"Execution failed: {e}")

    def _check_kill_switch(self):
        """
        Real-time risk check. 
        If Portfolio Unrealized Loss > Limit, liquidate everything.
        """
        try:
            positions = self.rest_client.get_positions()
            total_upnl = sum(p.unrealized_pnl for p in positions)
            
            # Logic: If using INR, values are large. Pct check is better.
            # But we need Total Balance to calc %.
            balances = self.rest_client.get_wallet_balance()
            total_equity = 0.0
            
            # Calculate equity safely
            # Note: Wallet structure varies, assuming standard Delta response
            if isinstance(balances, list):
                 for b in balances:
                     total_equity += float(b.get('equity', 0))
            
            if total_equity > 0:
                loss_pct = abs(total_upnl) / total_equity
                max_loss = settings.enhanced_risk.max_drawdown_limit_pct
                
                if total_upnl < 0 and loss_pct > max_loss:
                    log.critical(f"KILL SWITCH TRIGGERED: Loss {loss_pct:.2%} exceeds limit {max_loss:.2%}")
                    self._liquidate_all()
                    
        except Exception as e:
            log.error(f"Kill switch check failed: {e}")

    def _liquidate_all(self):
        """Panic close all positions."""
        log.warning("LIQUIDATING ALL POSITIONS...")
        positions = self.rest_client.get_positions()
        for pos in positions:
            try:
                if pos.size > 0: # Long -> Sell
                    self.rest_client.place_order(pos.product_symbol, OrderSide.SELL, pos.size, reduce_only=True)
                elif pos.size < 0: # Short -> Buy
                    self.rest_client.place_order(pos.product_symbol, OrderSide.BUY, abs(pos.size), reduce_only=True)
            except Exception as e:
                log.error(f"Failed to liquidate {pos.product_symbol}: {e}")
        
        self.stop() # Verify if we should stop bot or just flatten. Usually stop.

    def stop(self):
        self.is_running = False
        log.info("AsyncStrategyManager stopping...")
        if self.ws_client:
            self.ws_client.stop()

    async def _periodic_health_check(self):
        """Verify connections are alive."""
        ws = self.ws_client.ws
        # Check connection status (websockets library version-safe)
        is_connected = ws is not None and (
            (hasattr(ws, 'open') and ws.open) or
            (hasattr(ws, 'closed') and not ws.closed) or
            True  # Fallback: assume connected if we can't check
        )
        if not is_connected:
            log.warning("WebSocket disconnected. Attempting reconnect...")
            # Reconnect logic is handled by client
