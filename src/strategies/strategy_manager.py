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

from src.strategies.base_strategy import BaseStrategy, StrategySignal, StrategyType
from src.strategies.funding_arbitrage import FundingArbitrageStrategy
from src.strategies.correlated_hedging import CorrelatedHedgingStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
from src.delta_client import DeltaExchangeClient, Position
from src.websocket_client import DeltaWebSocketClient
from config.settings import settings
from utils.logger import log


class TradingState(str, Enum):
    """Overall trading state."""
    ACTIVE = "active"
    PAUSED = "paused"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    ERROR = "error"


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float
    current_balance: float
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
    
    def __init__(self, client: DeltaExchangeClient,
                 allocation: Optional[Dict[StrategyType, float]] = None,
                 daily_loss_limit: float = 0.03,
                 dry_run: bool = False):
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
        
        # WebSocket Client
        self.ws_client = DeltaWebSocketClient()
        self.ws_thread = None
        self._start_websocket_thread()
        self.today_stats: Optional[DailyStats] = None
        
        # Initialize strategies
        self.strategies: Dict[StrategyType, BaseStrategy] = {}
        self._initialize_strategies()
        
        log.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
        log.info(f"Allocation: {self._format_allocation()}")
        log.info(f"Daily loss limit: {daily_loss_limit:.1%}")

    def _start_websocket_thread(self) -> None:
        """Start the WebSocket client in a separate thread."""
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
        log.info("WebSocket thread started")

    def _run_websocket_loop(self) -> None:
        """Run the asyncio event loop for WebSocket in a thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Setup callbacks
        loop.run_until_complete(self._setup_websocket_subscriptions())
        
        # Run the connection
        try:
            loop.run_until_complete(self.ws_client.connect())
        except Exception as e:
            log.error(f"WebSocket loop error: {e}")

    async def _setup_websocket_subscriptions(self) -> None:
        """Subscribe to real-time channels."""
        # Subscribe to balance/position updates for risk management
        await self.ws_client.subscribe_positions(self._on_ws_positions_update)
        
        # Subscribe to order updates for execution tracking
        await self.ws_client.subscribe_orders(self._on_ws_orders_update)
        
        # Subscribe to funding rates for Tier 1 strategy
        symbols = getattr(settings.trading, 'trading_pairs', ['BTCUSDT', 'ETHUSDT'])
        await self.ws_client.subscribe_funding_rate(symbols, self._on_ws_funding_update)
        
        # Subscribe to portfolio margin for liquidation alerts (Phase 4.2)
        await self.ws_client.subscribe('portfolio_margins', callback=self._on_ws_margin_update)
        
        log.info("WebSocket subscriptions set up")

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
        if data.get('liquidation_risk'):
            log.critical(f"LIQUIDATION RISK ALERT! Data: {data}")
            # In emergency, we could trigger self._emergency_reduce_positions()

    
    def _initialize_strategies(self) -> None:
        """Initialize all strategy instances."""
        # Tier 1: Funding Arbitrage
        if StrategyType.FUNDING_ARBITRAGE in self.allocation:
            self.strategies[StrategyType.FUNDING_ARBITRAGE] = FundingArbitrageStrategy(
                client=self.client,
                capital_allocation=self.allocation[StrategyType.FUNDING_ARBITRAGE],
                dry_run=self.dry_run
            )
        
        # Tier 2: Correlated Hedging
        if StrategyType.CORRELATED_HEDGING in self.allocation:
            self.strategies[StrategyType.CORRELATED_HEDGING] = CorrelatedHedgingStrategy(
                client=self.client,
                capital_allocation=self.allocation[StrategyType.CORRELATED_HEDGING],
                dry_run=self.dry_run
            )
        
        # Tier 3: Multi-Timeframe
        if StrategyType.MULTI_TIMEFRAME in self.allocation:
            self.strategies[StrategyType.MULTI_TIMEFRAME] = MultiTimeframeStrategy(
                client=self.client,
                capital_allocation=self.allocation[StrategyType.MULTI_TIMEFRAME],
                dry_run=self.dry_run
            )
    
    def _format_allocation(self) -> str:
        """Format allocation for logging."""
        return ", ".join(
            f"{st.value}: {pct:.0%}" 
            for st, pct in self.allocation.items()
        )
    
    def run_cycle(self) -> List[StrategySignal]:
        """
        Run one complete trading cycle across all strategies.
        
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
            balance_data = self.client.get_wallet_balance()
            
            # Balance API returns a list of balances - find USDT balance
            total_balance = 0.0
            if isinstance(balance_data, list):
                for wallet in balance_data:
                    asset = wallet.get('asset_symbol', '') or wallet.get('asset', {}).get('symbol', '')
                    if asset in ['USDT', 'USD']:
                        total_balance = float(wallet.get('available_balance', 0) or 
                                              wallet.get('balance', 0) or 0)
                        break
                # If no USDT found, sum all available balances
                if total_balance == 0:
                    for wallet in balance_data:
                        total_balance += float(wallet.get('available_balance', 0) or 0)
            elif isinstance(balance_data, dict):
                total_balance = float(balance_data.get('available_balance', 0))
            
            log.debug(f"Total balance: ${total_balance:.2f}")
            
            # Initialize or check daily stats
            self._update_daily_stats(total_balance)
            
            # Check daily loss limit
            if self._is_daily_limit_hit():
                log.warning(f"Daily loss limit hit! PnL: {self.today_stats.pnl_pct:.2%}")
                self.state = TradingState.DAILY_LIMIT_HIT
                return all_signals
            
            # Get current positions
            positions = self.client.get_positions()
            
            # Run each strategy
            for strategy_type, strategy in self.strategies.items():
                if not strategy.is_active:
                    continue
                
                # Calculate allocated capital for this strategy
                allocated_capital = strategy.get_allocated_capital(total_balance)
                
                try:
                    signals = strategy.analyze(allocated_capital, positions)
                    
                    for signal in signals:
                        log.info(f"Signal from {strategy_type.value}: "
                                f"{signal.direction.value} {signal.symbol} "
                                f"(confidence: {signal.confidence:.0%})")
                    
                    all_signals.extend(signals)
                    
                except Exception as e:
                    log.error(f"Strategy {strategy_type.value} failed: {e}")
            
            # Execute signals
            if all_signals:
                self._execute_signals(all_signals)
            
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
        
        return self.state == TradingState.ACTIVE
    
    def _update_daily_stats(self, current_balance: float) -> None:
        """Update or initialize daily statistics."""
        today = date.today()
        
        if self.today_stats is None or self.today_stats.date != today:
            self.today_stats = DailyStats(
                date=today,
                starting_balance=current_balance,
                current_balance=current_balance
            )
            log.info(f"New trading day started. Balance: ${current_balance:.2f}")
        else:
            self.today_stats.current_balance = current_balance
    
    def _is_daily_limit_hit(self) -> bool:
        """Check if daily loss limit has been exceeded."""
        if self.today_stats is None:
            return False
        
        loss_pct = -self.today_stats.pnl_pct  # Negative of PnL
        return loss_pct >= self.daily_loss_limit
    
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
                log.info(f"Executing: {signal.direction.value} {signal.position_size:.6f} "
                        f"{signal.symbol} @ {signal.entry_price:.2f}")
                
                if signal.has_hedge:
                    log.info(f"  Hedge: {signal.hedge_direction.value} "
                            f"{signal.hedge_size:.6f} {signal.hedge_symbol}")
                
                # Strategy-specific execution
                if signal.strategy_type == StrategyType.FUNDING_ARBITRAGE:
                    self._execute_funding_arb_signal(signal, strategy)
                elif signal.strategy_type == StrategyType.CORRELATED_HEDGING:
                    self._execute_hedged_signal(signal, strategy)
                elif signal.strategy_type == StrategyType.MULTI_TIMEFRAME:
                    self._execute_mtf_signal(signal, strategy)
                
                # Update stats
                if self.today_stats:
                    self.today_stats.trades_executed += 1
                
            except Exception as e:
                log.error(f"Failed to execute signal: {e}")
    
    def _execute_funding_arb_signal(self, signal: StrategySignal, 
                                    strategy: FundingArbitrageStrategy) -> None:
        """Execute funding arbitrage signal."""
        action = signal.metadata.get('action')
        
        if action == 'enter_arbitrage':
            funding_rate = signal.metadata.get('funding_rate', 0)
            strategy.enter_arbitrage(
                signal.symbol, signal.position_size, funding_rate
            )
        elif action == 'exit_arbitrage':
            strategy.exit_arbitrage(signal.symbol)
    
    def _execute_hedged_signal(self, signal: StrategySignal,
                               strategy: CorrelatedHedgingStrategy) -> None:
        """Execute correlated hedging signal."""
        if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            strategy.execute_entry(signal)
        else:
            strategy.execute_exit(signal)
    
    def _execute_mtf_signal(self, signal: StrategySignal,
                            strategy: MultiTimeframeStrategy) -> None:
        """Execute multi-timeframe signal."""
        from src.strategies.base_strategy import SignalDirection
        
        if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            strategy.execute_entry(signal)
        else:
            strategy.execute_exit(signal)
    
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
            'state': self.state.value,
            'dry_run': self.dry_run,
            'daily_loss_limit': f"{self.daily_loss_limit:.1%}",
            'allocation': {
                st.value: f"{pct:.0%}" 
                for st, pct in self.allocation.items()
            },
            'today_stats': {
                'date': self.today_stats.date.isoformat() if self.today_stats else None,
                'starting_balance': self.today_stats.starting_balance if self.today_stats else 0,
                'current_balance': self.today_stats.current_balance if self.today_stats else 0,
                'pnl': self.today_stats.total_pnl if self.today_stats else 0,
                'pnl_pct': f"{self.today_stats.pnl_pct:.2%}" if self.today_stats else "0%",
                'trades': self.today_stats.trades_executed if self.today_stats else 0
            },
            'strategies': {
                st.value: strategy.get_status()
                for st, strategy in self.strategies.items()
            }
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
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_funding_earned': total_funding,
            'combined_return': total_pnl + total_funding,
            'strategies': {
                st.value: {
                    'trades': strategy.performance.total_trades,
                    'pnl': strategy.performance.total_pnl,
                    'win_rate': f"{strategy.performance.win_rate:.1%}",
                    'funding': strategy.performance.total_funding_earned
                }
                for st, strategy in self.strategies.items()
            }
        }


# Import for signal direction
from src.strategies.base_strategy import SignalDirection
