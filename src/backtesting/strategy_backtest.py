"""
Strategy Backtest Runner Module.
Runs actual strategy classes against historical data using mock client.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from src.backtesting.backtest_data_provider import BacktestDataProvider
from src.backtesting.backtest_client import BacktestDeltaClient, TradeRecord
from src.backtesting.data_fetcher import HistoricalData
from src.backtesting.performance import PerformanceMetrics, PerformanceAnalyzer
from src.strategies.base_strategy import StrategySignal, SignalDirection
from src.technical_analysis import TechnicalAnalyzer, Signal
from src.risk_manager import RiskManager
from src.strategy_core import StrategyCore
from config.settings import settings
from utils.logger import log


@dataclass
class StrategyBacktestResult:
    """Result from a strategy backtest."""
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
    signals_generated: int = 0
    signals_executed: int = 0
    bars_processed: int = 0
    
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
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed
        }


class StrategyBacktestRunner:
    """
    Runs actual strategy classes against historical data.
    
    Uses BacktestDeltaClient to simulate the exchange,
    allowing strategies to run their actual analysis logic.
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
        
        # Analysis tools
        self.analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = prices[:period].mean()
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _calculate_macd_histogram(self, prices: np.ndarray) -> float:
        """Calculate MACD histogram (MACD line - Signal line)."""
        if len(prices) < 26:
            return 0.0
        
        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-period EMA of MACD)
        # Simplified: use current MACD approximation
        macd_values = []
        for i in range(max(26, len(prices) - 20), len(prices)):
            e12 = self._calculate_ema(prices[:i+1], 12)
            e26 = self._calculate_ema(prices[:i+1], 26)
            macd_values.append(e12 - e26)
        
        if len(macd_values) >= 9:
            signal_line = np.mean(macd_values[-9:])
        else:
            signal_line = np.mean(macd_values) if macd_values else 0
        
        return macd_line - signal_line
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return (high[-1] - low[-1])
        
        tr_values = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_values.append(tr)
        
        return np.mean(tr_values[-period:])
    
    def run_correlated_hedging(
        self, 
        data_dict: Dict[str, HistoricalData],
        warmup_bars: int = 50
    ) -> StrategyBacktestResult:
        """
        Run Correlated Hedging Strategy backtest.
        
        Simulates the strategy logic without importing the actual strategy
        to avoid circular dependencies and API call issues.
        """
        log.info("Starting Correlated Hedging Backtest")
        
        # Setup data provider and mock client
        data_provider = BacktestDataProvider(data_dict)
        mock_client = BacktestDeltaClient(
            data_provider,
            self.initial_capital * 0.4,  # 40% allocation
            self.leverage,
            self.commission_pct,
            self.slippage_pct
        )
        
        symbols = data_provider.get_symbols()
        total_bars = data_provider.total_bars
        signals_generated = 0
        signals_executed = 0
        
        # Strategy parameters - WITH NEW FILTERS
        MIN_INDICATOR_AGREEMENT = 2
        RSI_PULLBACK_LONG = 55   # Not too overbought for longs
        RSI_PULLBACK_SHORT = 45  # Not too oversold for shorts
        MIN_ADX = 20  # Trend strength
        ATR_STOP_MULT = 3.0  # Wider stops
        ATR_TP_MULT = 4.5    # 1.5x risk:reward
        
        log.info(f"Processing {total_bars} bars for {symbols}")
        
        # Main simulation loop
        for bar_idx in range(warmup_bars, total_bars):
            data_provider.set_current_bar(bar_idx)
            
            # Check bracket orders (stop-loss/take-profit)
            mock_client.check_bracket_orders()
            
            # Analyze each symbol
            for symbol in symbols:
                # Skip if already have position
                if symbol in mock_client.positions:
                    continue
                
                # Get price data
                candles = data_provider.get_candles_up_to(symbol)
                if len(candles) < 200:  # Need 200 bars for EMA200
                    continue
                
                close = np.array([c.close for c in candles])
                high = np.array([c.high for c in candles])
                low = np.array([c.low for c in candles])
                current_price = close[-1]
                
                # === FIX 1: EMA200 Trend Filter ===
                ema200 = self._calculate_ema(close, 200)
                price_above_ema200 = current_price > ema200
                
                # Run technical analysis
                ta_result = self.analyzer.analyze(close, high, low, symbol)
                if ta_result is None:
                    continue
                
                # Check indicator agreement
                bullish = sum(1 for ind in ta_result.indicators if ind.signal.value == 1)
                bearish = sum(1 for ind in ta_result.indicators if ind.signal.value == -1)
                
                # Determine direction based on indicator majority
                direction = None
                if bullish >= MIN_INDICATOR_AGREEMENT and bullish > bearish:
                    direction = 'long'
                elif bearish >= MIN_INDICATOR_AGREEMENT and bearish > bullish:
                    direction = 'short'
                
                if direction is None:
                    continue
                
                # === FIX 1b: Only trade WITH the EMA200 trend ===
                if direction == 'long' and not price_above_ema200:
                    continue  # Skip longs in downtrend
                if direction == 'short' and price_above_ema200:
                    continue  # Skip shorts in uptrend
                
                # === FIX 2: MACD Histogram Momentum Filter ===
                macd_hist = self._calculate_macd_histogram(close)
                if direction == 'long' and macd_hist <= 0:
                    continue  # No bullish momentum
                if direction == 'short' and macd_hist >= 0:
                    continue  # No bearish momentum
                
                # === FIX 4: RSI Pullback Filter (don't chase) ===
                rsi_value = None
                for ind in ta_result.indicators:
                    if ind.name == "RSI":
                        rsi_value = ind.value
                        break
                
                if rsi_value:
                    if direction == 'long' and rsi_value > RSI_PULLBACK_LONG:
                        continue  # Too overbought, wait for pullback
                    if direction == 'short' and rsi_value < RSI_PULLBACK_SHORT:
                        continue  # Too oversold, wait for pullback
                
                # ADX trend strength filter
                try:
                    adx = self.analyzer.calculate_adx(high, low, close)
                    if adx < MIN_ADX:
                        continue
                except:
                    pass
                
                # Check higher timeframe trend alignment
                htf_candles = data_provider.get_higher_tf_candles(symbol, '4h')
                if len(htf_candles) >= 20:
                    htf_close = np.array([c.close for c in htf_candles])
                    htf_trend = self.analyzer.get_trend_direction(htf_close)
                    
                    if direction == 'long' and htf_trend == 'bearish':
                        continue
                    if direction == 'short' and htf_trend == 'bullish':
                        continue
                
                signals_generated += 1
                
                # Calculate position size
                balance = mock_client.get_balance()
                available = balance['available_balance']
                
                # === FIX 3: Wider ATR-based stops ===
                atr = self._calculate_atr(high, low, close, 14)
                
                if direction == 'long':
                    stop_loss = current_price - (atr * ATR_STOP_MULT)
                    take_profit = current_price + (atr * ATR_TP_MULT)
                else:
                    stop_loss = current_price + (atr * ATR_STOP_MULT)
                    take_profit = current_price - (atr * ATR_TP_MULT)
                
                # Calculate size based on risk
                risk_per_trade = available * 0.02  # 2% risk
                stop_distance = abs(current_price - stop_loss)
                size = risk_per_trade / stop_distance if stop_distance > 0 else 0
                
                if size <= 0:
                    continue
                
                # Execute trade with ATR-based bracket orders
                side = 'buy' if direction == 'long' else 'sell'
                result = mock_client.place_order(
                    product_symbol=symbol,
                    side=side,
                    size=size * self.leverage,
                    order_type='market',
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if result.get('success'):
                    signals_executed += 1
                    log.debug(f"Trade executed: {side} {symbol} @ {result['filled_price']:.2f}")
            
            # Update equity curve
            mock_client.update_equity()
        
        # Close any remaining positions
        for symbol in list(mock_client.positions.keys()):
            mock_client.close_position(symbol)
        
        # Calculate results
        return self._calculate_results(
            strategy_name="Correlated Hedging",
            mock_client=mock_client,
            data_dict=data_dict,
            signals_generated=signals_generated,
            signals_executed=signals_executed,
            bars_processed=total_bars - warmup_bars
        )
    
    def run_multi_timeframe(
        self,
        data_dict: Dict[str, HistoricalData],
        warmup_bars: int = 50
    ) -> StrategyBacktestResult:
        """
        Run Multi-Timeframe Strategy backtest.
        
        Uses 4h for trend, 15m for entry timing.
        """
        log.info("Starting Multi-Timeframe Backtest")
        
        data_provider = BacktestDataProvider(data_dict)
        mock_client = BacktestDeltaClient(
            data_provider,
            self.initial_capital * 0.2,  # 20% allocation
            self.leverage,
            self.commission_pct,
            self.slippage_pct
        )
        
        symbols = data_provider.get_symbols()
        total_bars = data_provider.total_bars
        signals_generated = 0
        signals_executed = 0
        
        # MTF parameters - WITH NEW FILTERS
        MIN_ADX = 22  # Slightly stricter for trend following
        MIN_INDICATOR_AGREEMENT = 2
        RSI_PULLBACK_LONG = 55
        RSI_PULLBACK_SHORT = 45
        ATR_STOP_MULT = 3.0
        ATR_TP_MULT = 4.5
        
        for bar_idx in range(warmup_bars, total_bars):
            data_provider.set_current_bar(bar_idx)
            mock_client.check_bracket_orders()
            
            for symbol in symbols:
                if symbol in mock_client.positions:
                    continue
                
                # Get base timeframe data
                candles = data_provider.get_candles_up_to(symbol)
                if len(candles) < 200:  # Need 200 for EMA200
                    continue
                
                close = np.array([c.close for c in candles])
                high = np.array([c.high for c in candles])
                low = np.array([c.low for c in candles])
                current_price = close[-1]
                
                # === FIX 1: EMA200 Trend Filter ===
                ema200 = self._calculate_ema(close, 200)
                price_above_ema200 = current_price > ema200
                
                # Get higher timeframe trend
                htf_candles = data_provider.get_higher_tf_candles(symbol, '4h')
                if len(htf_candles) < 30:
                    continue
                
                htf_close = np.array([c.close for c in htf_candles])
                htf_trend = self.analyzer.get_trend_direction(htf_close)
                
                if htf_trend == 'neutral':
                    continue  # MTF needs clear trend
                
                # Entry timing on lower timeframe
                ta_result = self.analyzer.analyze(close, high, low, symbol)
                if ta_result is None:
                    continue
                
                bullish = sum(1 for ind in ta_result.indicators if ind.signal.value == 1)
                bearish = sum(1 for ind in ta_result.indicators if ind.signal.value == -1)
                
                # Only enter WITH the HTF trend when LTF confirms
                direction = None
                if htf_trend == 'bullish' and bullish >= MIN_INDICATOR_AGREEMENT and bullish > bearish:
                    direction = 'long'
                elif htf_trend == 'bearish' and bearish >= MIN_INDICATOR_AGREEMENT and bearish > bullish:
                    direction = 'short'
                
                if direction is None:
                    continue
                
                # === FIX 1b: EMA200 alignment ===
                if direction == 'long' and not price_above_ema200:
                    continue
                if direction == 'short' and price_above_ema200:
                    continue
                
                # === FIX 2: MACD Histogram ===
                macd_hist = self._calculate_macd_histogram(close)
                if direction == 'long' and macd_hist <= 0:
                    continue
                if direction == 'short' and macd_hist >= 0:
                    continue
                
                # === FIX 4: RSI Pullback ===
                rsi_value = None
                for ind in ta_result.indicators:
                    if ind.name == "RSI":
                        rsi_value = ind.value
                        break
                
                if rsi_value:
                    if direction == 'long' and rsi_value > RSI_PULLBACK_LONG:
                        continue
                    if direction == 'short' and rsi_value < RSI_PULLBACK_SHORT:
                        continue
                
                # ADX filter
                try:
                    adx = self.analyzer.calculate_adx(high, low, close)
                    if adx < MIN_ADX:
                        continue
                except:
                    pass
                
                signals_generated += 1
                
                # === FIX 3: ATR-based stops ===
                balance = mock_client.get_balance()
                available = balance['available_balance']
                atr = self._calculate_atr(high, low, close, 14)
                
                if direction == 'long':
                    stop_loss = current_price - (atr * ATR_STOP_MULT)
                    take_profit = current_price + (atr * ATR_TP_MULT)
                else:
                    stop_loss = current_price + (atr * ATR_STOP_MULT)
                    take_profit = current_price - (atr * ATR_TP_MULT)
                
                risk_per_trade = available * 0.02
                stop_distance = abs(current_price - stop_loss)
                size = risk_per_trade / stop_distance if stop_distance > 0 else 0
                
                if size <= 0:
                    continue
                
                side = 'buy' if direction == 'long' else 'sell'
                result = mock_client.place_order(
                    product_symbol=symbol,
                    side=side,
                    size=size * self.leverage,
                    order_type='market',
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if result.get('success'):
                    signals_executed += 1
            
            mock_client.update_equity()
        
        for symbol in list(mock_client.positions.keys()):
            mock_client.close_position(symbol)
        
        return self._calculate_results(
            strategy_name="Multi-Timeframe",
            mock_client=mock_client,
            data_dict=data_dict,
            signals_generated=signals_generated,
            signals_executed=signals_executed,
            bars_processed=total_bars - warmup_bars
        )
    
    def run_funding_arbitrage(
        self,
        data_dict: Dict[str, HistoricalData],
        funding_rate: float = 0.0001,  # 0.01% per 8 hours avg
        warmup_bars: int = 50
    ) -> StrategyBacktestResult:
        """
        Funding Arbitrage - DISABLED.
        
        This strategy is disabled because:
        1. It requires capital on both spot AND perpetual markets
        2. Funding rates are highly variable and can be negative
        3. Real implementation needs historical funding rate data
        
        To enable: Set ALLOC_FUNDING_ARB > 0 in .env and provide real funding data.
        """
        log.info("Funding Arbitrage: DISABLED (no simulation)")
        
        data_provider = BacktestDataProvider(data_dict)
        capital = 0  # No allocation
        
        first_symbol = list(data_dict.keys())[0]
        start_date = data_dict[first_symbol].start_date
        end_date = data_dict[first_symbol].end_date
        
        return StrategyBacktestResult(
            strategy_name="Funding Arbitrage (Disabled)",
            symbols=list(data_dict.keys()),
            start_date=start_date,
            end_date=end_date,
            initial_capital=0,
            final_capital=0,
            total_pnl=0,
            total_pnl_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            equity_curve=[0],
            signals_generated=0,
            signals_executed=0,
            bars_processed=data_provider.total_bars - warmup_bars
        )
    
    def run_all_strategies(
        self,
        data_dict: Dict[str, HistoricalData]
    ) -> Dict[str, StrategyBacktestResult]:
        """Run all strategies and return combined results."""
        results = {}
        
        # Run each strategy
        results['correlated_hedging'] = self.run_correlated_hedging(data_dict)
        results['multi_timeframe'] = self.run_multi_timeframe(data_dict)
        results['funding_arbitrage'] = self.run_funding_arbitrage(data_dict)
        
        # Calculate combined results
        total_pnl = sum(r.total_pnl for r in results.values())
        total_trades = sum(r.total_trades for r in results.values())
        total_wins = sum(r.winning_trades for r in results.values())
        
        results['combined'] = StrategyBacktestResult(
            strategy_name="Combined",
            symbols=list(data_dict.keys()),
            start_date=results['correlated_hedging'].start_date,
            end_date=results['correlated_hedging'].end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital + total_pnl,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / self.initial_capital) * 100,
            total_trades=total_trades,
            winning_trades=total_wins,
            losing_trades=total_trades - total_wins,
            win_rate=(total_wins / total_trades * 100) if total_trades > 0 else 0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            bars_processed=results['correlated_hedging'].bars_processed
        )
        
        return results
    
    def _calculate_results(
        self,
        strategy_name: str,
        mock_client: BacktestDeltaClient,
        data_dict: Dict[str, HistoricalData],
        signals_generated: int,
        signals_executed: int,
        bars_processed: int
    ) -> StrategyBacktestResult:
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
        
        return StrategyBacktestResult(
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
            equity_curve=equity_curve,
            signals_generated=signals_generated,
            signals_executed=signals_executed,
            bars_processed=bars_processed
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
