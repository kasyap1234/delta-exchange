"""
Hedging Module.
Provides tools for managing hedge positions, correlation analysis, and funding rate monitoring.
"""

from src.hedging.correlation import CorrelationCalculator
from src.hedging.hedge_manager import HedgeManager
from src.hedging.funding_monitor import FundingMonitor

__all__ = [
    'CorrelationCalculator',
    'HedgeManager',
    'FundingMonitor',
]
