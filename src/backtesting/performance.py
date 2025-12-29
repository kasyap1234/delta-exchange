"""
Performance Analysis Module.
Provides detailed performance metrics and reporting for backtests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from utils.logger import log


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0

    # Other
    expectancy: float = 0.0
    avg_trade_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return": round(self.annualized_return, 2),
            "volatility": round(self.volatility, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
        }


class PerformanceAnalyzer:
    """
    Analyzes backtest results and calculates performance metrics.

    Provides:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Trade statistics
    - Equity curve analysis
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    def analyze(
        self, equity_curve: List[float], trades: List[Any], days: int = 30
    ) -> PerformanceMetrics:
        """
        Analyze equity curve and trades.

        Args:
            equity_curve: List of equity values over time
            trades: List of closed trades
            days: Number of trading days

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if not equity_curve or len(equity_curve) < 2:
            return metrics

        initial = equity_curve[0]
        final = equity_curve[-1]

        # Basic returns
        metrics.total_return = final - initial
        metrics.total_return_pct = ((final - initial) / initial) * 100

        # Annualized return
        periods_per_year = 365 / days if days > 0 else 1
        total_return_decimal = (final - initial) / initial
        metrics.annualized_return = (
            (1 + total_return_decimal) ** periods_per_year - 1
        ) * 100

        # Calculate returns series
        returns = self._calculate_returns(equity_curve)

        # Volatility (annualized)
        if returns:
            daily_vol = float(np.std(returns))
            metrics.volatility = daily_vol * np.sqrt(365) * 100

        # Drawdown
        max_dd, max_dd_pct, avg_dd = self._calculate_drawdown_stats(equity_curve)
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_pct = max_dd_pct
        metrics.avg_drawdown = avg_dd

        # Risk-adjusted ratios
        metrics.sharpe_ratio = self._calculate_sharpe(returns, days)
        metrics.sortino_ratio = self._calculate_sortino(returns, days)
        metrics.calmar_ratio = self._calculate_calmar(
            metrics.annualized_return, metrics.max_drawdown_pct
        )

        # Trade statistics
        if trades:
            self._analyze_trades(trades, metrics)

        return metrics

    def _calculate_returns(self, equity_curve: List[float]) -> List[float]:
        """Calculate period returns from equity curve."""
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] != 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)
        return returns

    def _calculate_drawdown_stats(self, equity_curve: List[float]) -> tuple:
        """Calculate drawdown statistics."""
        if not equity_curve:
            return 0.0, 0.0, 0.0

        peak = equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        drawdowns = []

        for equity in equity_curve:
            if equity > peak:
                peak = equity

            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0

            if dd > 0:
                drawdowns.append(dd_pct)

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        avg_dd = float(np.mean(drawdowns)) if drawdowns else 0.0

        return max_dd, max_dd_pct, avg_dd

    def _calculate_sharpe(self, returns: List[float], days: int) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = float(np.mean(returns))
        std_return = float(np.std(returns))

        if std_return == 0:
            return 0.0

        # Annualize
        periods_per_year = 365 / days if days > 0 else 365
        annualized_return = avg_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)

        sharpe = (annualized_return - self.risk_free_rate) / annualized_std

        return float(sharpe)

    def _calculate_sortino(self, returns: List[float], days: int) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = float(np.mean(returns))

        # Downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return 0.0

        downside_std = float(np.std(downside_returns))

        if downside_std == 0:
            return 0.0

        # Annualize
        periods_per_year = 365 / days if days > 0 else 365
        annualized_return = avg_return * periods_per_year
        annualized_downside_std = downside_std * np.sqrt(periods_per_year)

        sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std

        return float(sortino)

    def _calculate_calmar(
        self, annualized_return: float, max_drawdown_pct: float
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown_pct == 0:
            return 0.0

        return annualized_return / max_drawdown_pct

    def _analyze_trades(self, trades: List[Any], metrics: PerformanceMetrics) -> None:
        """Analyze trade statistics."""
        pnls = [t.pnl for t in trades if hasattr(t, "pnl")]

        if not pnls:
            return

        metrics.total_trades = len(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = (len(wins) / len(pnls)) * 100 if pnls else 0

        if wins:
            metrics.avg_win = float(np.mean(wins))
            metrics.largest_win = max(wins)

        if losses:
            metrics.avg_loss = float(np.mean(losses))
            metrics.largest_loss = min(losses)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy
        if metrics.total_trades > 0:
            win_rate_decimal = metrics.win_rate / 100
            if metrics.avg_loss != 0:
                metrics.expectancy = (
                    win_rate_decimal * metrics.avg_win
                    + (1 - win_rate_decimal) * metrics.avg_loss
                )


class ReportGenerator:
    """Generates formatted reports from backtest results."""

    @staticmethod
    def generate_summary(result: Any, metrics: PerformanceMetrics) -> str:
        """Generate text summary report."""
        lines = [
            "=" * 60,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Symbol: {result.symbol}",
            f"Strategy: {result.strategy}",
            f"Period: {result.start_date} to {result.end_date}",
            f"Bars Processed: {result.bars_processed}",
            "",
            "-" * 40,
            "CAPITAL",
            "-" * 40,
            f"Initial Capital: ${result.initial_capital:,.2f}",
            f"Final Capital: ${result.final_capital:,.2f}",
            f"Total P&L: ${metrics.total_return:,.2f}",
            f"Total Return: {metrics.total_return_pct:.2f}%",
            f"Annualized Return: {metrics.annualized_return:.2f}%",
            "",
            "-" * 40,
            "RISK METRICS",
            "-" * 40,
            f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2f}%)",
            f"Volatility (Ann.): {metrics.volatility:.2f}%",
            f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"Sortino Ratio: {metrics.sortino_ratio:.2f}",
            f"Calmar Ratio: {metrics.calmar_ratio:.2f}",
            "",
            "-" * 40,
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades: {metrics.total_trades}",
            f"Winning Trades: {metrics.winning_trades}",
            f"Losing Trades: {metrics.losing_trades}",
            f"Win Rate: {metrics.win_rate:.2f}%",
            f"Profit Factor: {metrics.profit_factor:.2f}",
            "",
            f"Avg Win: ${metrics.avg_win:,.2f}",
            f"Avg Loss: ${metrics.avg_loss:,.2f}",
            f"Largest Win: ${metrics.largest_win:,.2f}",
            f"Largest Loss: ${metrics.largest_loss:,.2f}",
            f"Expectancy: ${metrics.expectancy:,.2f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    @staticmethod
    def generate_trade_log(trades: List[Any]) -> str:
        """Generate trade-by-trade log."""
        lines = [
            "=" * 80,
            "TRADE LOG",
            "=" * 80,
            f"{'#':<4} {'Symbol':<10} {'Dir':<6} {'Entry':<12} {'Exit':<12} {'P&L':>10} {'%':>8}",
            "-" * 80,
        ]

        for trade in trades:
            if hasattr(trade, "exit_price") and trade.exit_price:
                lines.append(
                    f"{trade.id:<4} {trade.symbol:<10} {trade.direction.value:<6} "
                    f"{trade.entry_price:<12.2f} {trade.exit_price:<12.2f} "
                    f"${trade.pnl:>9.2f} {trade.pnl_pct:>7.2f}%"
                )

        lines.append("-" * 80)

        return "\n".join(lines)

    @staticmethod
    def generate_combined_report(results: Dict[str, Any]) -> str:
        """Generate combined multi-strategy report."""
        lines = [
            "=" * 70,
            "MULTI-STRATEGY BACKTEST REPORT",
            "=" * 70,
            "",
        ]

        # Per-strategy results
        for strategy, strategy_results in results.items():
            if strategy == "combined":
                continue

            lines.append(f"\n--- {strategy.upper()} ---")

            if isinstance(strategy_results, dict):
                if "total_pnl" in strategy_results:
                    # Single result (like funding_arbitrage)
                    lines.append(f"  P&L: ${strategy_results['total_pnl']:.2f}")
                    lines.append(f"  Return: {strategy_results['total_pnl_pct']:.2f}%")
                else:
                    # Multiple symbol results
                    for symbol, result in strategy_results.items():
                        if hasattr(result, "total_pnl"):
                            lines.append(f"  {symbol}:")
                            lines.append(f"    P&L: ${result.total_pnl:.2f}")
                            lines.append(f"    Trades: {result.total_trades}")
                            lines.append(f"    Win Rate: {result.win_rate:.1f}%")

        # Combined results
        if "combined" in results:
            combined = results["combined"]
            lines.extend(
                [
                    "",
                    "=" * 70,
                    "COMBINED RESULTS",
                    "=" * 70,
                    f"Initial Capital: ${combined['initial_capital']:,.2f}",
                    f"Final Capital: ${combined['final_capital']:,.2f}",
                    f"Total P&L: ${combined['total_pnl']:,.2f}",
                    f"Total Return: {combined['total_pnl_pct']:.2f}%",
                    f"Total Trades: {combined['total_trades']}",
                    f"Win Rate: {combined['win_rate']:.1f}%",
                    "",
                ]
            )

        return "\n".join(lines)
