"""
Strategy Manager Module.
Orchestrates multiple trading strategies with capital allocation and risk management.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import threading
import asyncio

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    StrategyType,
    SignalDirection,
)
from src.strategies.funding_arbitrage import FundingArbitrageStrategy
from src.strategies.correlated_hedging import CorrelatedHedgingStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
from src.delta_client import DeltaExchangeClient, Position
from src.websocket_client import DeltaWebSocketClient
from src.position_sync import PositionSyncManager
from config.settings import settings
from utils.logger import log
from src.utils.persistence_manager import PersistenceManager


class TradingState(str, Enum):
    """Overall trading state."""

    ACTIVE = "active"
    PAUSED = "paused"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    MAX_DRAWDOWN_HIT = "max_drawdown_hit"
    ERROR = "error"


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: date
    starting_balance: float
    current_balance: float
    peak_balance: float = 0.0  # Track highest balance for drawdown calculation
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_executed: int = 0
    funding_earned: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.funding_earned

    @property
    def pnl_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return self.total_pnl / self.starting_balance

    @property
    def is_losing_day(self) -> bool:
        return self.total_pnl < 0

    @property
    def drawdown_pct(self) -> float:
        """Calculate current drawdown from peak balance."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance


class StrategyManager:
    """
    Orchestrates multiple trading strategies.

    Responsibilities:
    1. Allocate capital across strategies
    2. Collect signals from all strategies
    3. Enforce daily loss limits
    4. Track overall performance
    5. Manage strategy lifecycle
    6. Handle real-time WebSocket updates

    Capital Allocation (default):
    - Tier 1 (Funding Arbitrage): 40%
    - Tier 2 (Correlated Hedging): 40%
    - Tier 3 (Multi-Timeframe): 20%
    """

    # Default capital allocation
    DEFAULT_ALLOCATION = {
        StrategyType.FUNDING_ARBITRAGE: 0.40,
        StrategyType.CORRELATED_HEDGING: 0.40,
        StrategyType.MULTI_TIMEFRAME: 0.20,
    }

    # Daily loss limit (default 3%)
    DEFAULT_DAILY_LOSS_LIMIT = 0.03

    def __init__(
        self,
        client: DeltaExchangeClient,
        allocation: Optional[Dict[StrategyType, float]] = None,
        daily_loss_limit: float = 0.03,
        dry_run: bool = False,
    ):
        """
        Initialize strategy manager.

        Args:
            client: Delta Exchange API client
            allocation: Custom capital allocation per strategy
            daily_loss_limit: Maximum daily loss before halting (default 3%)
            dry_run: If True, don't execute real trades
        """
        self.client = client
        self.allocation = allocation or self.DEFAULT_ALLOCATION
        self.daily_loss_limit = daily_loss_limit
        self.dry_run = dry_run

        # State
        self.state = TradingState.ACTIVE
        self.persistence = PersistenceManager()
        
        # Position sync manager - SINGLE SOURCE OF TRUTH for positions
        self.position_sync = PositionSyncManager(client)

        # WebSocket Client
        self.ws_client = DeltaWebSocketClient()
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self.ws_connected = False
        # Don't start WebSocket immediately - let it be optional
        # self._start_websocket_thread()
        self.today_stats: Optional[DailyStats] = None

        # Initialize strategies
        self.strategies: Dict[StrategyType, BaseStrategy] = {}
        self._initialize_strategies()

        # Load persisted state on startup
        self._reconcile_with_exchange()

        log.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
        log.info(f"Allocation: {self._format_allocation()}")
        log.info(f"Daily loss limit: {daily_loss_limit:.1%}")

    def _start_websocket_thread(self) -> None:
        """Start the WebSocket client in a separate thread."""
        if self.ws_thread and self.ws_thread.is_alive():
            log.warning("WebSocket thread already running")
            return
        
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
        log.info("WebSocket thread started")

    def _run_websocket_loop(self) -> None:
        """Run the asyncio event loop for WebSocket in a thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.ws_loop = loop

            async def setup_and_connect():
                """Setup subscriptions and connect."""
                # Set up subscriptions first (they'll be stored for resubscription after connection)
                # These calls store the subscription info but don't send yet (ws is None)
                try:
                    # Subscribe to balance/position updates for risk management
                    await self.ws_client.subscribe_positions(self._on_ws_positions_update)
                    
                    # Subscribe to order updates for execution tracking
                    await self.ws_client.subscribe_orders(self._on_ws_orders_update)
                    
                    # Subscribe to funding rates for Tier 1 strategy
                    symbols = getattr(settings.trading, "trading_pairs", ["BTCUSD", "ETHUSD"])
                    await self.ws_client.subscribe_funding_rate(symbols, self._on_ws_funding_update)
                    
                    # Subscribe to portfolio margin for liquidation alerts
                    await self.ws_client.subscribe(
                        "portfolio_margins", callback=self._on_ws_margin_update
                    )
                    
                    log.info("WebSocket subscriptions configured (will activate on connect)")
                except Exception as e:
                    log.warning(f"Failed to set up WebSocket subscriptions: {e}")
                
                # Connect (which will establish connection and resubscribe automatically)
                await self.ws_client.connect()
            
            # Run the connection
            loop.run_until_complete(setup_and_connect())
        except Exception as e:
            log.error(f"WebSocket loop error: {e}")
            import traceback
            log.error(traceback.format_exc())
        finally:
            self.ws_loop = None

    def _stop_websocket(self) -> None:
        """Stop the WebSocket client and thread."""
        if self.ws_client:
            self.ws_client.stop()
        
        if self.ws_loop and self.ws_loop.is_running():
            # Schedule loop stop
            self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
        
        if self.ws_thread and self.ws_thread.is_alive():
            # Thread should stop when ws_client.stop() is called
            self.ws_thread.join(timeout=5.0)
            if self.ws_thread.is_alive():
                log.warning("WebSocket thread did not stop within timeout")

    def _on_ws_positions_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time position updates."""
        # In a production bot, we'd update our internal state immediately
        # and maybe trigger emergency exits if SL/TP hit on exchange.
        log.debug(f"WS Position Update: {data}")

    def _on_ws_orders_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time order updates."""
        log.debug(f"WS Order Update: {data}")

    def _on_ws_funding_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time funding rate updates."""
        # Tier 1 strategy can react instantly to funding changes
        log.debug(f"WS Funding Update: {data}")

    def _on_ws_margin_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time portfolio margin updates."""
        # Phase 4.2: Liquidation risk alerts
        if data.get("liquidation_risk"):
            log.critical(f"LIQUIDATION RISK ALERT! Data: {data}")
            # In emergency, we could trigger self._emergency_reduce_positions()

    def _initialize_strategies(self) -> None:
        """Initialize all strategy instances."""
        # =====================================================================
        # DISABLED: Funding Arbitrage requires spot trading, which Delta Exchange
        # India does not support via API. The strategy tries to:
        #   1. Long spot (BTC/USDT) 
        #   2. Short perpetual (BTCUSD)
        # But spot trading calls will fail. Enable only if spot support is added.
        # =====================================================================
        # if StrategyType.FUNDING_ARBITRAGE in self.allocation:
        #     self.strategies[StrategyType.FUNDING_ARBITRAGE] = FundingArbitrageStrategy(
        #         client=self.client,
        #         capital_allocation=self.allocation[StrategyType.FUNDING_ARBITRAGE],
        #         dry_run=self.dry_run,
        #     )
        if StrategyType.FUNDING_ARBITRAGE in self.allocation:
            log.warning("FundingArbitrageStrategy DISABLED - Delta India does not support spot trading API")

        # Tier 2: Correlated Hedging
        if StrategyType.CORRELATED_HEDGING in self.allocation:
            self.strategies[StrategyType.CORRELATED_HEDGING] = (
                CorrelatedHedgingStrategy(
                    client=self.client,
                    capital_allocation=self.allocation[StrategyType.CORRELATED_HEDGING],
                    position_sync=self.position_sync,
                    dry_run=self.dry_run,
                )
            )

        # Tier 3: Multi-Timeframe
        if StrategyType.MULTI_TIMEFRAME in self.allocation:
            self.strategies[StrategyType.MULTI_TIMEFRAME] = MultiTimeframeStrategy(
                client=self.client,
                capital_allocation=self.allocation[StrategyType.MULTI_TIMEFRAME],
                dry_run=self.dry_run,
            )

    def _format_allocation(self) -> str:
        """Format allocation for logging."""
        return ", ".join(
            f"{st.value}: {pct:.0%}" for st, pct in self.allocation.items()
        )

    def run_cycle(
        self, override_positions: Optional[List[Position]] = None
    ) -> List[StrategySignal]:
        """
        Run one complete trading cycle across all strategies.

        Args:
            override_positions: Optional list of positions to use instead of fetching from client
                               (useful for paper trading)

        1. Check daily loss limit
        2. Get account balance
        3. Get current positions
        4. Collect signals from each strategy
        5. Execute signals

        Returns:
            List of generated signals
        """
        all_signals = []

        # Check if we should trade
        if not self._should_trade():
            log.info(f"Trading halted: {self.state.value}")
            return all_signals

        try:
            # Get account balance
            from src.utils.balance_utils import get_usd_balance
            try:
                total_balance = get_usd_balance(self.client)
            except Exception as e:
                if self.dry_run:
                    log.warning(f"Could not get wallet balance (dry run mode): {e}")
                    total_balance = 0.0
                else:
                    raise

            # If zero balance on testnet/dry-run, use a default for paper trading
            if total_balance == 0 and self.dry_run:
                total_balance = 200.0
                log.info(
                    f"Zero or inaccessible balance. Using default paper trading balance: ${total_balance:.2f}"
                )

            log.debug(f"Total balance: ${total_balance:.2f}")

            # Initialize or check daily stats
            self._update_daily_stats(total_balance)

            # Check daily loss limit
            if self._is_daily_limit_hit():
                log.warning(
                    f"Daily loss limit hit! PnL: {self.today_stats.pnl_pct:.2%}"
                )
                self.state = TradingState.DAILY_LIMIT_HIT
                return all_signals

            # Check max drawdown limit
            if self._is_max_drawdown_hit():
                log.warning(
                    f"Maximum drawdown limit hit! Drawdown: {self.today_stats.drawdown_pct:.2%}"
                )
                self.state = TradingState.MAX_DRAWDOWN_HIT
                return all_signals

            # Get current positions (use override if provided for paper trading)
            if override_positions is not None:
                positions = override_positions
                log.info(f"Using {len(positions)} override positions for analysis")
            else:
                # SYNC POSITIONS FROM EXCHANGE - Single Source of Truth
                snapshot = self.position_sync.sync_positions()
                if not snapshot.sync_successful:
                    log.error(f"Position sync failed: {snapshot.error}")
                    if not self.dry_run:
                        return all_signals  # Don't trade if we can't verify positions
                
                positions = self.position_sync.get_all_positions()
                if positions:
                    log.info(f"Retrieved {len(positions)} positions from exchange: {[p.product_symbol for p in positions]}")

            # Run each strategy
            for strategy_type, strategy in self.strategies.items():
                if not strategy.is_active:
                    continue

                # Calculate allocated capital for this strategy
                allocated_capital = strategy.get_allocated_capital(total_balance)

                try:
                    signals = strategy.analyze(allocated_capital, positions)

                    for signal in signals:
                        log.info(
                            f"Signal from {strategy_type.value}: "
                            f"{signal.direction.value} {signal.symbol} "
                            f"(confidence: {signal.confidence:.0%})"
                        )

                    # Filter signals by minimum size/notional
                    valid_signals = []
                    min_notional = getattr(settings.trading, "min_order_notional", 10.0)

                    for signal in signals:
                        notional = signal.position_size * signal.entry_price
                        if notional < min_notional:
                            log.warning(
                                f"{strategy_type.value}: {signal.symbol} signal rejected - notional ${notional:.2f} < ${min_notional} min"
                            )
                            continue
                        valid_signals.append(signal)

                    all_signals.extend(valid_signals)

                except Exception as e:
                    log.error(
                        f"Strategy {strategy_type.value} failed during analysis: {e}",
                        exc_info=True
                    )

            # Execute signals
            if all_signals:
                self._execute_signals(all_signals)

            # Save state after each cycle
            self._save_current_state()

            return all_signals

        except Exception as e:
            log.error(f"Trading cycle failed: {e}")
            self.state = TradingState.ERROR
            return all_signals

    def _should_trade(self) -> bool:
        """Check if trading should continue."""
        if self.state == TradingState.DAILY_LIMIT_HIT:
            # Check if it's a new day
            if self.today_stats and self.today_stats.date != date.today():
                self.state = TradingState.ACTIVE
                self.today_stats = None
                log.info("New day - resetting trading state")
                return True
            return False

        if self.state == TradingState.MAX_DRAWDOWN_HIT:
            # Check if we've recovered from drawdown
            if (
                self.today_stats and self.today_stats.drawdown_pct < 0.02
            ):  # Under 2% drawdown
                self.state = TradingState.ACTIVE
                log.info("Recovered from max drawdown - resuming trading")
                return True
            return False

        return self.state == TradingState.ACTIVE

    def _update_daily_stats(self, current_balance: float) -> None:
        """Update or initialize daily statistics."""
        today = date.today()

        if self.today_stats is None or self.today_stats.date != today:
            self.today_stats = DailyStats(
                date=today,
                starting_balance=current_balance,
                current_balance=current_balance,
                peak_balance=current_balance,  # Initialize peak
            )
            log.info(f"New trading day started. Balance: ${current_balance:.2f}")
        else:
            self.today_stats.current_balance = current_balance
            # Update peak balance for drawdown calculation
            if current_balance > self.today_stats.peak_balance:
                self.today_stats.peak_balance = current_balance

    def _is_daily_limit_hit(self) -> bool:
        """Check if daily loss limit has been exceeded."""
        if self.today_stats is None:
            return False

        loss_pct = -self.today_stats.pnl_pct  # Negative of PnL
        return loss_pct >= self.daily_loss_limit

    def _is_max_drawdown_hit(self) -> bool:
        """Check if maximum drawdown limit has been exceeded."""
        if self.today_stats is None:
            return False

        drawdown = self.today_stats.drawdown_pct
        max_dd_limit = getattr(settings.enhanced_risk, "max_drawdown_limit_pct", 0.10)
        return drawdown >= max_dd_limit

    def _execute_signals(self, signals: List[StrategySignal]) -> None:
        """Execute trading signals."""
        for signal in signals:
            if not signal.is_actionable:
                continue

            strategy = self.strategies.get(signal.strategy_type)
            if not strategy:
                continue

            try:
                # Log the execution
                log.info(
                    f"Executing: {signal.direction.value} {signal.position_size:.6f} "
                    f"{signal.symbol} @ {signal.entry_price:.2f}"
                )

                if signal.has_hedge:
                    log.info(
                        f"  Hedge: {signal.hedge_direction.value} "
                        f"{signal.hedge_size:.6f} {signal.hedge_symbol}"
                    )

                # Strategy-specific execution
                if signal.direction == SignalDirection.CLOSE_PARTIAL:
                    self._execute_partial_exit(signal, strategy)
                elif signal.strategy_type == StrategyType.FUNDING_ARBITRAGE:
                    self._execute_funding_arb_signal(signal, strategy)
                elif signal.strategy_type == StrategyType.CORRELATED_HEDGING:
                    self._execute_hedged_signal(signal, strategy)
                elif signal.strategy_type == StrategyType.MULTI_TIMEFRAME:
                    self._execute_mtf_signal(signal, strategy)

                # Update stats
                if self.today_stats:
                    self.today_stats.trades_executed += 1

            except Exception as e:
                log.error(
                    f"Failed to execute signal {signal.direction.value} for {signal.symbol}: {e}",
                    exc_info=True
                )

    def _execute_funding_arb_signal(
        self, signal: StrategySignal, strategy: FundingArbitrageStrategy
    ) -> None:
        """Execute funding arbitrage signal."""
        action = signal.metadata.get("action")

        if action == "enter_arbitrage":
            funding_rate = signal.metadata.get("funding_rate", 0)
            strategy.enter_arbitrage(signal.symbol, signal.position_size, funding_rate)
        elif action == "exit_arbitrage":
            strategy.exit_arbitrage(signal.symbol)

    def _execute_hedged_signal(
        self, signal: StrategySignal, strategy: CorrelatedHedgingStrategy
    ) -> None:
        """Execute correlated hedging signal."""
        if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            strategy.execute_entry(signal)
        else:
            strategy.execute_exit(signal)

    def _execute_mtf_signal(
        self, signal: StrategySignal, strategy: MultiTimeframeStrategy
    ) -> None:
        """Execute multi-timeframe signal."""
        from src.strategies.base_strategy import SignalDirection

        if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            strategy.execute_entry(signal)
        else:
            strategy.execute_exit(signal)

    def _execute_partial_exit(
        self, signal: StrategySignal, strategy: BaseStrategy
    ) -> None:
        """Execute a partial position exit."""
        log.info(
            f"Executing partial exit for {signal.symbol}: {signal.position_size:.6f}"
        )

        # Determine side for the closing order (opposite of current)
        from src.delta_client import OrderSide, OrderType

        side = None
        # 1. Explicitly handle via metadata
        if signal.metadata:
            original_side = signal.metadata.get("original_side")
            if original_side == "long":
                side = OrderSide.SELL
            elif original_side == "short":
                side = OrderSide.BUY

        # 2. If metadata absent, try to infer from direction or position
        if not side:
            if signal.direction == SignalDirection.CLOSE_LONG:
                side = OrderSide.SELL
            elif signal.direction == SignalDirection.CLOSE_SHORT:
                side = OrderSide.BUY
            elif hasattr(signal, "position") and getattr(signal, "position"):
                # Handle potential dynamic position field on signal
                pos = getattr(signal, "position")
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
            elif hasattr(strategy, "get_position"):
                # Try to get position from strategy's internal tracking
                pos = strategy.get_position(signal.symbol)
                if pos and pos.size != 0:
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY

        # 3. If inference impossible, raise error
        if not side:
            error_msg = (
                f"Cannot determine order side for {signal.direction.value} on {signal.symbol}. "
                f"Provide 'original_side' in metadata or ensure a position exists."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        if not self.dry_run:
            self.client.place_order(
                symbol=signal.symbol,
                side=side,
                size=signal.position_size,
                order_type=OrderType.MARKET,
                reduce_only=True,
            )
            # Update strategy internal state
            if hasattr(strategy, "apply_partial_exit"):
                strategy.apply_partial_exit(signal.symbol, signal.position_size)
        else:
            log.info(
                f"[DRY RUN] Would execute partial {side.value} for {signal.position_size} {signal.symbol}"
            )
            if hasattr(strategy, "apply_partial_exit"):
                strategy.apply_partial_exit(signal.symbol, signal.position_size)

    def _save_current_state(self) -> None:
        """Save the current bot state to persistent storage."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
            "today_stats": {
                "date": self.today_stats.date.isoformat() if self.today_stats else None,
                "starting_balance": self.today_stats.starting_balance
                if self.today_stats
                else 0,
                "realized_pnl": self.today_stats.realized_pnl
                if self.today_stats
                else 0,
                "trades_executed": self.today_stats.trades_executed
                if self.today_stats
                else 0,
                "funding_earned": self.today_stats.funding_earned
                if self.today_stats
                else 0,
            },
            "strategies": {},
        }

        for st_type, strategy in self.strategies.items():
            state["strategies"][st_type.value] = {
                "performance": {
                    "total_trades": strategy.performance.total_trades,
                    "winning_trades": strategy.performance.winning_trades,
                    "losing_trades": strategy.performance.losing_trades,
                    "total_pnl": strategy.performance.total_pnl,
                    "total_funding_earned": strategy.performance.total_funding_earned,
                },
                # Strategy-specific active position state
                "active_state": getattr(strategy, "get_active_state", lambda: {})(),
            }

        self.persistence.save_state(state)

    def _reconcile_with_exchange(self) -> None:
        """Reconcile internal state with exchange reality using persisted data."""
        persisted_state = self.persistence.load_state()
        if not persisted_state:
            return

        log.info("Reconciling internal state with persisted data...")

        # Restore daily stats if same day
        stats_data = persisted_state.get("today_stats", {})
        if stats_data.get("date") == date.today().isoformat():
            self.today_stats = DailyStats(
                date=date.today(),
                starting_balance=stats_data.get("starting_balance", 0),
                current_balance=stats_data.get(
                    "starting_balance", 0
                ),  # Will be updated in cycle
                realized_pnl=stats_data.get("realized_pnl", 0),
                trades_executed=stats_data.get("trades_executed", 0),
                funding_earned=stats_data.get("funding_earned", 0),
            )
            log.info(f"Restored daily stats for {date.today()}")

        # Restore strategy performance and active states
        strategies_data = persisted_state.get("strategies", {})
        for st_type_val, st_data in strategies_data.items():
            try:
                st_type = StrategyType(st_type_val)
                strategy = self.strategies.get(st_type)
                if not strategy:
                    continue

                perf_data = st_data.get("performance", {})
                strategy.performance.total_trades = perf_data.get("total_trades", 0)
                strategy.performance.winning_trades = perf_data.get("winning_trades", 0)
                strategy.performance.losing_trades = perf_data.get("losing_trades", 0)
                strategy.performance.total_pnl = perf_data.get("total_pnl", 0)
                strategy.performance.total_funding_earned = perf_data.get(
                    "total_funding_earned", 0
                )

                # Restore active state (e.g., partial exits for MTF)
                if hasattr(strategy, "restore_active_state"):
                    strategy.restore_active_state(st_data.get("active_state", {}))

            except ValueError:
                continue

    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """Get a specific strategy instance."""
        return self.strategies.get(strategy_type)

    def pause_strategy(self, strategy_type: StrategyType) -> bool:
        """Pause a specific strategy."""
        strategy = self.strategies.get(strategy_type)
        if strategy:
            strategy.pause()
            log.info(f"Strategy paused: {strategy_type.value}")
            return True
        return False

    def resume_strategy(self, strategy_type: StrategyType) -> bool:
        """Resume a specific strategy."""
        strategy = self.strategies.get(strategy_type)
        if strategy:
            strategy.resume()
            log.info(f"Strategy resumed: {strategy_type.value}")
            return True
        return False

    def pause_all(self) -> None:
        """Pause all strategies."""
        self.state = TradingState.PAUSED
        for strategy in self.strategies.values():
            strategy.pause()
        log.info("All strategies paused")

    def resume_all(self) -> None:
        """Resume all strategies."""
        self.state = TradingState.ACTIVE
        for strategy in self.strategies.values():
            strategy.resume()
        log.info("All strategies resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all strategies."""
        return {
            "state": self.state.value,
            "dry_run": self.dry_run,
            "daily_loss_limit": f"{self.daily_loss_limit:.1%}",
            "allocation": {
                st.value: f"{pct:.0%}" for st, pct in self.allocation.items()
            },
            "today_stats": {
                "date": self.today_stats.date.isoformat() if self.today_stats else None,
                "starting_balance": self.today_stats.starting_balance
                if self.today_stats
                else 0,
                "current_balance": self.today_stats.current_balance
                if self.today_stats
                else 0,
                "pnl": self.today_stats.total_pnl if self.today_stats else 0,
                "pnl_pct": f"{self.today_stats.pnl_pct:.2%}"
                if self.today_stats
                else "0%",
                "trades": self.today_stats.trades_executed if self.today_stats else 0,
            },
            "strategies": {
                st.value: strategy.get_status()
                for st, strategy in self.strategies.items()
            },
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all strategies."""
        total_trades = 0
        total_pnl = 0.0
        total_funding = 0.0

        for strategy in self.strategies.values():
            perf = strategy.performance
            total_trades += perf.total_trades
            total_pnl += perf.total_pnl
            total_funding += perf.total_funding_earned

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "total_funding_earned": total_funding,
            "combined_return": total_pnl + total_funding,
            "strategies": {
                st.value: {
                    "trades": strategy.performance.total_trades,
                    "pnl": strategy.performance.total_pnl,
                    "win_rate": f"{strategy.performance.win_rate:.1%}",
                    "funding": strategy.performance.total_funding_earned,
                }
                for st, strategy in self.strategies.items()
            },
        }


# Import for signal direction
from src.strategies.base_strategy import SignalDirection
