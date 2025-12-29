"""
Backtesting Framework for Delta Exchange Trading Bot.
Provides historical data fetching, trade simulation, and performance analysis.
"""


# Lazy imports to avoid circular import issues
def get_fetcher():
    from src.backtesting.data_fetcher import HistoricalDataFetcher

    return HistoricalDataFetcher


def get_engine():
    from src.backtesting.backtest_engine import BacktestEngine, BacktestResult

    return BacktestEngine, BacktestResult


def get_analyzer():
    from src.backtesting.performance import PerformanceAnalyzer, PerformanceMetrics

    return PerformanceAnalyzer, PerformanceMetrics


__all__ = [
    "HistoricalDataFetcher",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
]
