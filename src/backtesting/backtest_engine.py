"""
Backtest Engine Module.
Simulates trading strategies on historical data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np

from src.backtesting.data_fetcher import HistoricalData, OHLCVBar
from src.technical_analysis import TechnicalAnalyzer, TechnicalAnalysisResult, Signal
from src.risk_manager import RiskManager, PositionSizing
from src.hedging.hedge_manager import HedgeManager, HedgedPosition
from src.hedging.correlation import CorrelationCalculator
from src.delta_client import DeltaExchangeClient  # Needed for types, even if mocked
from src.unified_signal_validator import UnifiedSignalValidator, ValidationResult
from config.settings import settings
from utils.logger import log


class TradeDirection(str, Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    """Trade status."""

    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"


@dataclass
class BacktestTrade:
    """Represents a single trade in the backtest."""

    id: int
    symbol: str
    direction: TradeDirection
    entry_time: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float

    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    # Additional metadata
    signal_strength: int = 0
    confidence: float = 0.0
    strategy: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate P&L percentage."""
        if self.direction == TradeDirection.LONG:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def close(self, exit_price: float, exit_time: str, reason: str) -> None:
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.pnl = self.calculate_pnl(exit_price)
        self.pnl_pct = self.calculate_pnl_pct(exit_price)

        if "stop" in reason.lower():
            self.status = TradeStatus.STOPPED_OUT
        elif "profit" in reason.lower():
            self.status = TradeStatus.TAKE_PROFIT
        else:
            self.status = TradeStatus.CLOSED


@dataclass
class BacktestResult:
    """Complete backtest results."""

    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0

    # Trade details
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Additional info
    bars_processed: int = 0
    signals_generated: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def avg_win(self) -> float:
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(winners) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        losers = [t.pnl for t in self.trades if t.pnl < 0]
        return np.mean(losers) if losers else 0.0

    @property
    def risk_reward_ratio(self) -> float:
        if self.avg_loss == 0:
            return 0.0
        return abs(self.avg_win / self.avg_loss)

    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "bars_processed": self.bars_processed,
        }


class BacktestEngine:
    """
    Main backtesting engine.

    Simulates trading on historical data with:
    - Technical analysis signal generation
    - Position sizing based on risk management
    - Stop-loss and take-profit execution
    - Equity curve tracking
    - Performance metrics calculation
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 5,
        commission_pct: float = 0.0006,  # 0.06% taker fee
        slippage_pct: float = 0.0001,
    ):  # 0.01% slippage
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital in USD
            leverage: Leverage multiplier
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Trading components
        self.analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.signal_validator = UnifiedSignalValidator()
        
        # Initialize Hedging Components (Mock Client)
        class MockClient(DeltaExchangeClient):
            def __init__(self): pass
            def get_candles(self, *args, **kwargs): return []
            def get_ticker(self, symbol): return {'mark_price': 0.0} # Will be patched
            def place_order(self, *args, **kwargs): return type('obj', (object,), {'id': 'mock_id'})
            def wait_for_order_fill(self, *args, **kwargs): return type('obj', (object,), {'state': 'filled'})
            def get_positions(self): return []

        self.mock_client = MockClient()
        self.correlation_calc = CorrelationCalculator(self.mock_client)
        
        # We need a custom HedgeManager that doesn't rely on live API for positions
        self.hedge_manager = HedgeManager(
            self.mock_client, 
            self.correlation_calc, 
            dry_run=True  # Always dry run in backtest to avoid API calls
        )

        # State
        self.capital = initial_capital
        self.open_positions: Dict[str, BacktestTrade] = {}
        # Track active hedges: primary_symbol -> HedgedPosition
        self.active_hedges: Dict[str, HedgedPosition] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.trade_counter = 0

        # Settings
        self.min_signal_strength = settings.trading.min_signal_agreement
        self.max_positions = settings.trading.max_open_positions

    def reset(self) -> None:
        """Reset engine state for new backtest."""
        self.capital = self.initial_capital
        self.open_positions = {}
        self.active_hedges = {}
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]
        self.trade_counter = 0

    def run(
        self, data: HistoricalData, strategy: str = "technical", warmup_bars: int = 50
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Historical OHLCV data
            strategy: Strategy name for identification
            warmup_bars: Number of bars to skip for indicator warmup

        Returns:
            BacktestResult with complete analysis
        """
        self.reset()

        log.info(f"Starting backtest for {data.symbol}")
        log.info(f"Data range: {data.start_date} to {data.end_date}")
        log.info(f"Total bars: {len(data.bars)}, Warmup: {warmup_bars}")

        signals_generated = 0

        # Process each bar
        for i in range(warmup_bars, len(data.bars)):
            current_bar = data.bars[i]

            # Get historical data up to current bar for analysis
            historical_closes = data.closes[: i + 1]
            historical_highs = data.highs[: i + 1]
            historical_lows = data.lows[: i + 1]
            historical_volumes = data.volumes[: i + 1]

            # --- HEDGING LOGIC (Pre-check) ---
            # Check for intra-bar hedge triggers BEFORE stop-loss kills the trade
            loss_threshold_pct = -0.02 # -2%

            for symbol, trade in list(self.open_positions.items()):
                if symbol in self.active_hedges:
                    continue

                # Check if price dipped below threshold during this bar
                # For LONG: Check Low. For SHORT: Check High.
                current_pnl_pct = 0.0
                
                if trade.direction == TradeDirection.LONG:
                    # Lowest point of bar
                    worst_case_val = (current_bar.low - trade.entry_price) * trade.size
                    entry_val = trade.entry_price * trade.size
                    if entry_val > 0:
                        current_pnl_pct = worst_case_val / entry_val
                
                elif trade.direction == TradeDirection.SHORT:
                     # Highest point of bar (worst for short)
                    worst_case_val = (trade.entry_price - current_bar.high) * trade.size
                    entry_val = trade.entry_price * trade.size
                    if entry_val > 0:
                        current_pnl_pct = worst_case_val / entry_val

                if current_pnl_pct <= loss_threshold_pct:
                     log.debug(f"Hedge triggered for {symbol} (Intra-bar worse case: {current_pnl_pct:.2%})")
                     self.active_hedges[symbol] = True

            # Update open positions (check stop-loss/take-profit)
            atr = None
            if len(historical_closes) >= 14:
                 atr = self.analyzer.calculate_atr(historical_highs, historical_lows, historical_closes)

            self._update_positions(
                current_price=current_bar.close,
                low=current_bar.low,
                high=current_bar.high,
                bar_time=current_bar.datetime,
                atr=atr
            )

            # --- FUNDING ARBITRAGE SIMULATION ---
            # Simulate low-risk funding arb income (0.03% daily or ~11% APY)
            # We apply it to 40% of the capital (standard allocation)
            daily_yield = 0.0003
            bars_per_day = 96 # 15m candles
            funding_income = (self.capital * 0.4 * daily_yield) / bars_per_day
            self.capital += funding_income

            # Update mock client for correlation calc (if needed for other logic)
            self.mock_client.get_ticker = lambda s: {'mark_price': current_bar.close} if s == data.symbol else {'mark_price': 0.0}

            # Generate signals
            ta_result = self.analyzer.analyze(
                historical_closes, historical_highs, historical_lows, data.symbol
            )

            if ta_result and ta_result.combined_signal != Signal.HOLD:
                signals_generated += 1

            # Check for entry signals (or position flip)
            # Always call _check_entry - it handles position flipping internally
            self._check_entry(
                data.symbol, current_bar, ta_result,
                historical_highs, historical_lows, historical_closes,
                historical_volumes=historical_volumes
            )

            # Update equity curve
            equity = self._calculate_equity(current_bar.close)
            self.equity_curve.append(equity)

        # Close any remaining positions at last price
        if self.open_positions:
            last_bar = data.bars[-1]
            for symbol in list(self.open_positions.keys()):
                self._close_position(
                    symbol, last_bar.close, last_bar.datetime, "End of backtest"
                )

        # Calculate final metrics
        result = self._calculate_results(data, strategy, signals_generated)

        log.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"P&L: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2f}%)"
        )

        return result

    def run_multi_symbol(
        self, data_dict: Dict[str, HistoricalData], strategy: str = "technical"
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple symbols.

        Args:
            data_dict: Dictionary of symbol -> HistoricalData
            strategy: Strategy name

        Returns:
            Dictionary of symbol -> BacktestResult
        """
        results = {}

        for symbol, data in data_dict.items():
            try:
                result = self.run(data, strategy)
                results[symbol] = result
            except Exception as e:
                log.error(f"Backtest failed for {symbol}: {e}")

        return results

    def _can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        if symbol in self.open_positions:
            log.debug(f"Already have open position in {symbol}, skipping signal")
            return False
        if len(self.open_positions) >= self.max_positions:
            log.warning(f"Max positions ({self.max_positions}) reached, skipping signal for {symbol}")
            return False
        return True

    def _should_flip_position(self, symbol: str, new_direction: str) -> bool:
        """Check if we should close an existing position and flip to the opposite direction."""
        if symbol not in self.open_positions:
            return False
        
        existing_trade = self.open_positions[symbol]
        existing_direction = existing_trade.direction.value
        
        # Flip if we have a LONG and want to go SHORT, or vice versa
        return existing_direction != new_direction


    def _check_entry(
        self, symbol: str, bar: OHLCVBar, ta_result: Optional[TechnicalAnalysisResult],
        historical_highs: Optional[np.ndarray] = None,
        historical_lows: Optional[np.ndarray] = None,
        historical_closes: Optional[np.ndarray] = None,
        historical_volumes: Optional[np.ndarray] = None,
    ) -> None:
        """Check for entry signals and open position if valid."""
        if ta_result is None:
            return

        # Determine direction
        direction = None
        if self.analyzer.should_enter_long(ta_result):
            direction = "long"
        elif self.analyzer.should_enter_short(ta_result):
            direction = "short"

        if direction is None:
            return

        log.info(f"ENTRY CHECK for {symbol}: direction={direction}, signal={ta_result.combined_signal.value}, confidence={ta_result.confidence:.2f}")

        # Position Flipping: If we have an open position in the OPPOSITE direction, close it first
        if symbol in self.open_positions:
            if self._should_flip_position(symbol, direction):
                log.info(f"Flipping position on {symbol}: Closing {self.open_positions[symbol].direction.value} to open {direction}")
                self._close_position(symbol, bar.close, bar.datetime, f"Position flip to {direction}")
            else:
                # Same direction, skip (already have a position)
                return


        # Use Unified Signal Validator for consistent logic across systems
        # We need ADX, RSI, Volume, and Regime for the validator
        adx = 0.0
        rsi = None
        volume_signal = "neutral"
        market_regime = "trending"
        
        if historical_highs is not None and historical_lows is not None and historical_closes is not None:
            adx = self.analyzer.calculate_adx(historical_highs, historical_lows, historical_closes)
            rsi = self.analyzer.calculate_rsi(historical_closes)
            
            if historical_volumes is not None:
                vol_res = self.analyzer.calculate_volume_signal(historical_volumes, historical_closes)
                volume_signal = vol_res.signal.value if hasattr(vol_res.signal, 'value') else str(vol_res.signal)
                
                # Market regime (Trend vs Range)
                # Since AdvancedIndicators is usually an attribute of analyzer if it's there
                # Or we can use a simpler version if needed.
                # TechnicalAnalyzer.is_trending is already a good proxy
                is_trending, _ = self.analyzer.is_trending(historical_highs, historical_lows, historical_closes)
                market_regime = "trending" if is_trending else "ranging"

        # In backtest, we might not have higher_tf_trend easily available per bar unless pre-calculated
        # For now, we'll use neutral if not provided, but we can pass it if we expand the engine.
        higher_tf_trend = "neutral" 
        
        is_valid, validation_result, reason = self.signal_validator.validate_entry(
            symbol=symbol,
            direction=direction,
            ta_result=ta_result,
            higher_tf_trend=higher_tf_trend,
            adx=adx,
            rsi=rsi,
            volume_signal=volume_signal,
            market_regime=market_regime,
            timestamp=datetime.fromisoformat(bar.datetime) if isinstance(bar.datetime, str) else bar.datetime
        )
        
        if not is_valid:
            log.info(f"Backtest: {symbol} {direction} REJECTED: {reason}")
            return

        log.info(f"Backtest: {symbol} entry approved: {reason}")

        # Map back to TradeDirection enum if needed, or use strings
        trade_dir = TradeDirection.LONG if direction == "long" else TradeDirection.SHORT

        # Calculate position size
        entry_price = bar.close * (
            1 + self.slippage_pct
            if trade_dir == TradeDirection.LONG
            else 1 - self.slippage_pct
        )

        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            side="buy" if trade_dir == TradeDirection.LONG else "sell",
            entry_price=entry_price,
            available_balance=self.capital,
        )

        # Apply leverage
        position_value = sizing.size * entry_price
        leveraged_size = sizing.size * self.leverage

        # Check minimum position
        if position_value < 10:
            return

        # Create trade
        self.trade_counter += 1
        trade = BacktestTrade(
            id=self.trade_counter,
            symbol=symbol,
            direction=trade_dir,
            entry_time=bar.datetime,
            entry_price=entry_price,
            size=leveraged_size,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            signal_strength=ta_result.signal_strength,
            confidence=ta_result.confidence,
            strategy="technical",
        )

        # Deduct commission
        commission = position_value * self.commission_pct
        self.capital -= commission

        self.open_positions[symbol] = trade
        log.debug(f"Opened {direction} position on {symbol} @ {entry_price:.2f}")

    def _update_positions(self, current_price: float, low: float, high: float, bar_time: str, atr: Optional[float] = None) -> None:
        """Update open positions and check exit/trailing conditions."""
        for symbol in list(self.open_positions.keys()):
            trade = self.open_positions[symbol]

            # 1. Update Trailing Stop / Break-Even
            # Calculate current R-multiple using RiskManager
            r_multiple = self.risk_manager.get_r_multiple(
                entry_price=trade.entry_price,
                current_price=current_price,
                stop_loss=trade.stop_loss, # Note: this should ideally use INITIAL stop loss for R tracking
                side="buy" if trade.direction == TradeDirection.LONG else "sell"
            )
            
            # Use original risk for R-multiple consistency if stored
            # (In this backtestTrade, stop_loss is mutated, so we might want to store initial_sl)
            # For now, let's assume trade.stop_loss is what we use.

            # 1. Break-Even: After 1.0R profit, move SL to entry
            if r_multiple >= 1.0 and trade.stop_loss != trade.entry_price:
                new_sl = self.risk_manager.calculate_break_even_stop(
                    entry_price=trade.entry_price,
                    side="buy" if trade.direction == TradeDirection.LONG else "sell"
                )
                
                # Only move if it improves the stop
                if trade.direction == TradeDirection.LONG:
                    if new_sl > trade.stop_loss:
                        trade.stop_loss = new_sl
                        log.debug(f"Moved SL to Break-Even for {symbol} @ {new_sl:.2f}")
                else:
                    if new_sl < trade.stop_loss:
                        trade.stop_loss = new_sl
                        log.debug(f"Moved SL to Break-Even for {symbol} @ {new_sl:.2f}")

            # 2. Trailing: After 1.5R, trail at ATR distance
            if r_multiple >= 1.5 and atr is not None:
                new_trail_sl = self.risk_manager.calculate_trailing_stop(
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    atr=atr,
                    side="buy" if trade.direction == TradeDirection.LONG else "sell",
                    multiplier=getattr(settings.enhanced_risk, "atr_stop_multiplier", 2.0)
                )
                
                if trade.direction == TradeDirection.LONG:
                    if new_trail_sl > trade.stop_loss:
                        trade.stop_loss = new_trail_sl
                else:
                    if new_trail_sl < trade.stop_loss:
                        trade.stop_loss = new_trail_sl

            # 2. Check Exits (SL/TP)
            if trade.direction == TradeDirection.LONG:
                if low <= trade.stop_loss:
                    self._close_position(
                        symbol, trade.stop_loss, bar_time, "Stop-loss triggered"
                    )
                    continue
                if high >= trade.take_profit:
                    self._close_position(
                        symbol, trade.take_profit, bar_time, "Take-profit triggered"
                    )
                    continue
            else:  # SHORT
                if high >= trade.stop_loss:
                    self._close_position(
                        symbol, trade.stop_loss, bar_time, "Stop-loss triggered"
                    )
                    continue
                if low <= trade.take_profit:
                    self._close_position(
                        symbol, trade.take_profit, bar_time, "Take-profit triggered"
                    )
                    continue

    def _close_position(
        self, symbol: str, exit_price: float, exit_time: str, reason: str
    ) -> None:
        """Close an open position."""
        if symbol not in self.open_positions:
            return

        trade = self.open_positions[symbol]

        # Apply slippage
        if trade.direction == TradeDirection.LONG:
            actual_exit = exit_price * (1 - self.slippage_pct)
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)

        trade.close(actual_exit, exit_time, reason)

        # Calculate and apply P&L
        pnl = trade.pnl
        
        # --- HEDGING SIMULATION (Simplified) ---
        # If this trade was hedged, calculate the offset from the short hedge.
        # Since we don't have multi-symbol data here, we assume:
        # 1. Hedge Ratio = 0.3 (Default)
        # 2. Correlation = 0.8 (Strong)
        # 3. Hedge moves opposite to Primary (Profit when Primary Loses)
        hedge_pnl = 0.0
        if symbol in self.active_hedges and pnl < 0:
            # Net Hedge Profit ~= |Loss| * Ratio * Correlation
            # Example: Loss $100. Hedge covers 30% ($30). Moves 80% correlated.
            # Hedge Profit = $100 * 0.3 * 0.8 = $24. Net Loss = $76.
            hedge_offset_factor = 0.3 * 0.8 
            hedge_pnl = abs(pnl) * hedge_offset_factor
            log.debug(f"Applied simulated hedge profit: +${hedge_pnl:.2f}")

        commission = abs(trade.size * actual_exit / self.leverage) * self.commission_pct
        net_pnl = pnl + hedge_pnl - commission

        self.capital += net_pnl
        
        # Record result for cooldowns (Whiplash protection)
        # Use bar time for backtest consistency
        ts = datetime.fromisoformat(trade.exit_time) if isinstance(trade.exit_time, str) else trade.exit_time
        self.signal_validator.record_trade_result(symbol, net_pnl, timestamp=ts)

        # Move to closed trades
        # Update trade P&L for reporting (optional, but good for stats)
        # trade.pnl += hedge_pnl  # Don't mutate trade.pnl directly to keep raw stats accurate, but strictly net_pnl is what matters for equity.

        del self.open_positions[symbol]
        if symbol in self.active_hedges:
            del self.active_hedges[symbol]
            
        self.closed_trades.append(trade)

        log.debug(
            f"Closed {trade.direction.value} on {symbol} @ {actual_exit:.2f}, "
            f"P&L: ${net_pnl:.2f} (incl. hedge +${hedge_pnl:.2f}), Reason: {reason}"
        )

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L."""
        equity = self.capital

        for trade in self.open_positions.values():
            unrealized = trade.calculate_pnl(current_price)
            equity += unrealized

        return equity

    def _calculate_results(
        self, data: HistoricalData, strategy: str, signals_generated: int
    ) -> BacktestResult:
        """Calculate final backtest results."""
        all_trades = self.closed_trades

        # Basic stats
        total_trades = len(all_trades)
        winning_trades = len([t for t in all_trades if t.pnl > 0])
        losing_trades = len([t for t in all_trades if t.pnl < 0])

        # P&L
        total_pnl = sum(t.pnl for t in all_trades)
        total_pnl_pct = (
            (self.capital - self.initial_capital) / self.initial_capital
        ) * 100

        # Drawdown
        max_drawdown, max_drawdown_pct = self._calculate_drawdown()

        # Profit factor
        gross_profit = sum(t.pnl for t in all_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified daily)
        sharpe = self._calculate_sharpe()

        return BacktestResult(
            symbol=data.symbol,
            strategy=strategy,
            start_date=data.start_date,
            end_date=data.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            trades=all_trades,
            equity_curve=self.equity_curve,
            bars_processed=len(data.bars),
            signals_generated=signals_generated,
        )

    def _calculate_drawdown(self) -> tuple[float, float]:
        """Calculate maximum drawdown."""
        if not self.equity_curve:
            return 0.0, 0.0

        peak = self.equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity

            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[
                i - 1
            ]
            returns.append(ret)

        if not returns:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming 15-minute bars, ~35,000 bars/year)
        periods_per_year = 35000
        annualized_return = avg_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)

        sharpe = (annualized_return - risk_free_rate) / annualized_std

        return sharpe


class MultiStrategyBacktest:
    """
    Backtests multiple strategies with capital allocation.

    Supports:
    - Tier 1: Funding Arbitrage (simulated)
    - Tier 2: Correlated Hedging
    - Tier 3: Multi-Timeframe
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        allocation: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-strategy backtest.

        Args:
            initial_capital: Total starting capital
            allocation: Capital allocation per strategy
        """
        self.initial_capital = initial_capital
        self.allocation = allocation or {
            "funding_arbitrage": 0.40,
            "correlated_hedging": 0.40,
            "multi_timeframe": 0.20,
        }

    def run(
        self,
        data_dict: Dict[str, HistoricalData],
        higher_tf_data: Optional[Dict[str, HistoricalData]] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-strategy backtest.

        Args:
            data_dict: Symbol -> HistoricalData (entry timeframe)
            higher_tf_data: Symbol -> HistoricalData (higher timeframe for MTF)

        Returns:
            Combined results dictionary
        """
        results = {}

        # Tier 2: Correlated Hedging (main strategy for backtest)
        tier2_capital = self.initial_capital * self.allocation["correlated_hedging"]
        tier2_engine = BacktestEngine(initial_capital=tier2_capital)

        tier2_results = {}
        for symbol, data in data_dict.items():
            result = tier2_engine.run(data, strategy="correlated_hedging")
            tier2_results[symbol] = result

        results["correlated_hedging"] = tier2_results

        # Tier 3: Multi-Timeframe (if higher TF data provided)
        if higher_tf_data:
            tier3_capital = self.initial_capital * self.allocation["multi_timeframe"]
            tier3_engine = BacktestEngine(initial_capital=tier3_capital)

            tier3_results = {}
            for symbol, data in data_dict.items():
                result = tier3_engine.run(data, strategy="multi_timeframe")
                tier3_results[symbol] = result

            results["multi_timeframe"] = tier3_results

        # Tier 1: Funding Arbitrage (simulated - use historical funding rates)
        tier1_capital = self.initial_capital * self.allocation["funding_arbitrage"]
        tier1_result = self._simulate_funding_arbitrage(tier1_capital, 30)  # 30 days
        results["funding_arbitrage"] = tier1_result

        # Calculate combined results
        results["combined"] = self._calculate_combined_results(results)

        return results

    def _simulate_funding_arbitrage(self, capital: float, days: int) -> Dict:
        """
        Simulate funding arbitrage returns.

        Uses historical average funding rates for crypto perpetuals.
        Average funding: 0.01% per 8 hours = 0.03% per day = ~11% APY
        """
        # Conservative estimate: 0.02% daily return on delta-neutral position
        daily_return = 0.0002
        total_return = capital * daily_return * days

        return {
            "strategy": "funding_arbitrage",
            "initial_capital": capital,
            "final_capital": capital + total_return,
            "total_pnl": total_return,
            "total_pnl_pct": (total_return / capital) * 100,
            "estimated_apy": daily_return * 365 * 100,
            "note": "Simulated based on historical average funding rates",
        }

    def _calculate_combined_results(self, results: Dict) -> Dict:
        """Calculate combined results across all strategies."""
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

        # Sum up results from each strategy
        for strategy, strategy_results in results.items():
            if strategy == "combined":
                continue

            if strategy == "funding_arbitrage":
                total_pnl += strategy_results["total_pnl"]
            elif isinstance(strategy_results, dict):
                for symbol, result in strategy_results.items():
                    if hasattr(result, "total_pnl"):
                        total_pnl += result.total_pnl
                        total_trades += result.total_trades
                        total_wins += result.winning_trades

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.initial_capital + total_pnl,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.initial_capital) * 100,
            "total_trades": total_trades,
            "winning_trades": total_wins,
            "win_rate": (total_wins / total_trades * 100) if total_trades > 0 else 0,
        }
