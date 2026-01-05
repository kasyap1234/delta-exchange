"""
Unified Backtest Runner.
Uses actual strategy classes with BacktestDeltaClient for accurate backtesting.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from src.backtesting.data_fetcher import HistoricalDataFetcher, HistoricalData
from src.backtesting.backtest_data_provider import BacktestDataProvider
from src.backtesting.backtest_client import BacktestDeltaClient, TradeRecord
from src.strategies.correlated_hedging import CorrelatedHedgingStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.base_strategy import StrategySignal, SignalDirection
from config.settings import settings
from utils.logger import log


@dataclass
class UnifiedBacktestResult:
    """Result from unified backtest."""
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy_name,
            'period': f"{self.start_date} to {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_pnl': round(self.total_pnl, 2),
            'return_pct': round(self.total_pnl_pct, 2),
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2)
        }


class UnifiedBacktestRunner:
    """
    Runs backtests using the SAME strategy classes as live trading.
    
    This ensures backtest results accurately reflect live trading behavior.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 5,
        commission_pct: float = 0.0006,
        slippage_pct: float = 0.0001
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def run(
        self,
        data_dict: Dict[str, HistoricalData],
        warmup_bars: int = 50,
        strategy_type: str = "correlated_hedging"
    ) -> UnifiedBacktestResult:
        """
        Run backtest using actual strategy class.
        
        Args:
            data_dict: Symbol -> HistoricalData mapping
            warmup_bars: Number of bars to skip for indicator warmup
            strategy_type: 'correlated_hedging' or 'multi_timeframe'
        """
        log.info(f"Starting Unified Backtest: {strategy_type}")
        
        # Setup providers
        data_provider = BacktestDataProvider(data_dict)
        mock_client = BacktestDeltaClient(
            data_provider,
            self.initial_capital,
            self.leverage,
            self.commission_pct,
            self.slippage_pct
        )
        
        # Initialize ACTUAL strategy class with mock client
        if strategy_type == "correlated_hedging":
            strategy = CorrelatedHedgingStrategy(
                client=mock_client,
                capital_allocation=1.0,  # Full allocation for this backtest
                dry_run=False  # We want to execute trades
            )
        elif strategy_type == "multi_timeframe":
            strategy = MultiTimeframeStrategy(
                client=mock_client,
                capital_allocation=1.0,
                dry_run=False
            )
        elif strategy_type == "pairs_trading":
            strategy = PairsTradingStrategy(
                client=mock_client
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        symbols = data_provider.get_symbols()
        total_bars = data_provider.total_bars
        signals_generated = 0
        signals_executed = 0
        
        log.info(f"Processing {total_bars} bars for {symbols}")
        
        # Main simulation loop
        for bar_idx in range(warmup_bars, total_bars):
            data_provider.set_current_bar(bar_idx)
            
            # Check bracket orders (stop-loss/take-profit)
            mock_client.check_bracket_orders()
            
            # Get current positions for strategy
            current_positions = mock_client.get_positions()
            
            # Use strategy's analyze method - THE SAME CODE AS LIVE
            try:
                signals = strategy.analyze(
                    available_capital=mock_client.capital,
                    current_positions=current_positions
                )
                
                for signal in signals:
                    signals_generated += 1
                    
                    if signal.is_actionable:
                        # Execute the signal
                        result = self._execute_signal(mock_client, signal)
                        if result.get('success'):
                            signals_executed += 1
                            
            except Exception as e:
                log.debug(f"Strategy error at bar {bar_idx}: {e}")
            
            # Update equity curve
            mock_client.update_equity()
        
        # Close remaining positions
        for symbol in list(mock_client.positions.keys()):
            mock_client.close_position(symbol)
        
        # Calculate results
        return self._calculate_results(
            strategy_name=strategy_type,
            mock_client=mock_client,
            data_dict=data_dict,
            signals_generated=signals_generated,
            signals_executed=signals_executed
        )
    
    def _execute_signal(
        self, 
        client: BacktestDeltaClient, 
        signal: StrategySignal
    ) -> Dict[str, Any]:
        """Execute a strategy signal through the mock client."""
        
        if signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            side = 'buy' if signal.direction == SignalDirection.LONG else 'sell'
            return client.place_order(
                product_symbol=signal.symbol,
                side=side,
                size=signal.position_size,
                order_type='market',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
        
        elif signal.direction in [SignalDirection.CLOSE_LONG, SignalDirection.CLOSE_SHORT]:
            return client.close_position(signal.symbol)
        
        return {'success': False, 'error': 'Unknown signal direction'}
    
    def _calculate_results(
        self,
        strategy_name: str,
        mock_client: BacktestDeltaClient,
        data_dict: Dict[str, HistoricalData],
        signals_generated: int,
        signals_executed: int
    ) -> UnifiedBacktestResult:
        """Calculate backtest results from mock client state."""
        trades = mock_client.closed_trades
        equity_curve = mock_client.equity_curve
        
        # Get date range
        first_symbol = list(data_dict.keys())[0]
        start_date = data_dict[first_symbol].start_date
        end_date = data_dict[first_symbol].end_date
        
        # Calculate stats
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        total_pnl = sum(t.pnl for t in trades)
        final_capital = mock_client.initial_capital + total_pnl
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        max_dd, max_dd_pct = self._calculate_drawdown(equity_curve)
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe(equity_curve)
        
        log.info(f"Backtest complete: {total_trades} trades, "
                f"P&L: ${total_pnl:.2f} ({total_pnl/mock_client.initial_capital*100:.2f}%)")
        
        return UnifiedBacktestResult(
            strategy_name=strategy_name,
            symbols=list(data_dict.keys()),
            start_date=start_date,
            end_date=end_date,
            initial_capital=mock_client.initial_capital,
            final_capital=final_capital,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / mock_client.initial_capital) * 100,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=(winning_trades / total_trades * 100) if total_trades > 0 else 0,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> tuple:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0, 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        return max_dd, max_dd_pct
    
    def _calculate_sharpe(self, equity_curve: List[float], risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret == 0:
            return 0.0
        
        # Annualize (assuming 15m bars)
        periods_per_year = 35000
        ann_ret = avg_ret * periods_per_year
        ann_std = std_ret * np.sqrt(periods_per_year)
        
        return (ann_ret - risk_free) / ann_std


def run_unified_backtest(
    symbols: List[str] = None,
    days: int = 60,
    warmup_bars: int = 1000,
    strategy: str = "correlated_hedging",
    initial_capital: float = 10000.0
) -> UnifiedBacktestResult:
    """
    Convenience function to run unified backtest.
    
    Args:
        symbols: List of symbols to backtest
        days: Number of days of actual SIMULATION (after warmup)
        warmup_bars: Number of bars for indicator warmup
        strategy: Strategy to backtest
        initial_capital: Starting capital
    """
    if symbols is None:
        symbols = settings.trading.trading_pairs
    
    # Calculate total days needed (warmup + simulation)
    # 15m resolution: 96 bars per day
    warmup_days = int(np.ceil(warmup_bars / 96))
    total_days = days + warmup_days
    
    # Fetch data
    fetcher = HistoricalDataFetcher()
    data_dict = {}
    
    for symbol in symbols:
        data = fetcher.fetch(symbol, resolution='15m', days_back=total_days)
        if data and data.bars:
            data_dict[symbol] = data
    
    if not data_dict:
        raise ValueError("No data fetched")
    
    # Run backtest
    runner = UnifiedBacktestRunner(initial_capital=initial_capital)
    return runner.run(data_dict, warmup_bars=warmup_bars, strategy_type=strategy)


if __name__ == "__main__":
    # Quick test
    result = run_unified_backtest(days=30, strategy="correlated_hedging")
    print(f"\n=== Unified Backtest Result ===")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
