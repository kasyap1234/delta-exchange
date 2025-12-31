"""
Funding Rate Monitor Module.
Monitors and tracks funding rates for perpetual futures contracts.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.delta_client import DeltaExchangeClient
from utils.logger import log


class FundingDirection(str, Enum):
    """Direction of funding payment."""
    LONGS_PAY = "longs_pay"  # Positive funding - shorts receive
    SHORTS_PAY = "shorts_pay"  # Negative funding - longs receive
    NEUTRAL = "neutral"


@dataclass
class FundingRateData:
    """Data about a funding rate."""
    symbol: str
    funding_rate: float  # Per 8-hour rate
    funding_rate_8h: float  # Annualized from 8h rate
    predicted_rate: Optional[float]
    next_funding_time: datetime
    timestamp: datetime
    
    @property
    def direction(self) -> FundingDirection:
        """Get the direction of funding payment."""
        if self.funding_rate > 0.0001:  # Positive threshold
            return FundingDirection.LONGS_PAY
        elif self.funding_rate < -0.0001:  # Negative threshold
            return FundingDirection.SHORTS_PAY
        return FundingDirection.NEUTRAL
    
    @property
    def annualized_rate(self) -> float:
        """
        Calculate annualized funding rate.
        
        Funding is charged 3x per day (every 8 hours).
        Annual rate = (8h_rate) * 3 * 365
        """
        return self.funding_rate * 3 * 365
    
    @property
    def is_favorable_for_short(self) -> bool:
        """Check if funding is favorable for holding short positions."""
        return self.funding_rate > 0.0005  # > 0.05% per 8h
    
    @property
    def is_favorable_for_long(self) -> bool:
        """Check if funding is favorable for holding long positions."""
        return self.funding_rate < -0.0005  # < -0.05% per 8h
    
    def estimate_daily_income(self, position_size: float) -> float:
        """
        Estimate daily income from funding.
        
        Args:
            position_size: Size of position in USD
            
        Returns:
            Estimated daily income from funding
        """
        # 3 funding periods per day
        return abs(self.funding_rate) * position_size * 3


@dataclass
class FundingHistory:
    """Historical funding data for analysis."""
    symbol: str
    rates: List[FundingRateData] = field(default_factory=list)
    
    @property
    def average_rate(self) -> float:
        """Calculate average funding rate."""
        if not self.rates:
            return 0.0
        return sum(r.funding_rate for r in self.rates) / len(self.rates)
    
    @property
    def is_consistently_positive(self) -> bool:
        """Check if funding has been consistently positive."""
        if len(self.rates) < 3:
            return False
        return all(r.funding_rate > 0 for r in self.rates[-3:])
    
    @property
    def is_consistently_negative(self) -> bool:
        """Check if funding has been consistently negative."""
        if len(self.rates) < 3:
            return False
        return all(r.funding_rate < 0 for r in self.rates[-3:])


class FundingMonitor:
    """
    Monitors funding rates for perpetual futures.
    
    Used by the funding arbitrage strategy to:
    1. Track current funding rates
    2. Predict upcoming funding
    3. Calculate potential arbitrage income
    4. Alert on significant funding rate changes
    """
    
    # Minimum funding rate to consider for arbitrage (0.005% = 0.00005)
    MIN_FUNDING_THRESHOLD = 0.00005
    
    # Symbols to monitor
    DEFAULT_SYMBOLS = ['BTCUSD', 'ETHUSD', 'SOLUSD']
    
    def __init__(self, client: DeltaExchangeClient,
                 funding_threshold: float = 0.0005,
                 symbols: Optional[List[str]] = None):
        """
        Initialize funding monitor.
        
        Args:
            client: Delta Exchange API client
            funding_threshold: Minimum rate to trigger arbitrage
            symbols: Symbols to monitor (defaults to BTC, ETH, SOL)
        """
        self.client = client
        self.funding_threshold = funding_threshold
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        
        # Cache for funding data
        self._current_rates: Dict[str, FundingRateData] = {}
        self._history: Dict[str, FundingHistory] = {}
        self._last_update: Optional[datetime] = None
        self._update_interval = timedelta(minutes=5)
        
        # Track funding income
        self.total_funding_earned: float = 0.0
    
    def get_funding_rate(self, symbol: str) -> Optional[FundingRateData]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            
        Returns:
            FundingRateData or None if not available
        """
        # Check if update needed
        if self._should_update():
            self._update_all_rates()
        
        return self._current_rates.get(symbol)
    
    def get_all_rates(self) -> Dict[str, FundingRateData]:
        """
        Get funding rates for all monitored symbols.
        
        Returns:
            Dictionary of symbol -> FundingRateData
        """
        if self._should_update():
            self._update_all_rates()
        
        return self._current_rates.copy()
    
    def _should_update(self) -> bool:
        """Check if we should fetch fresh data."""
        if self._last_update is None:
            return True
        return datetime.now() - self._last_update > self._update_interval
    
    def _update_all_rates(self) -> None:
        """Fetch and update funding rates for all symbols."""
        for symbol in self.symbols:
            try:
                rate_data = self._fetch_funding_rate(symbol)
                if rate_data:
                    self._current_rates[symbol] = rate_data
                    
                    # Update history
                    if symbol not in self._history:
                        self._history[symbol] = FundingHistory(symbol=symbol)
                    self._history[symbol].rates.append(rate_data)
                    
                    # Keep last 24 periods (8 days at 8h intervals)
                    if len(self._history[symbol].rates) > 24:
                        self._history[symbol].rates = self._history[symbol].rates[-24:]
                        
            except Exception as e:
                log.error(f"Failed to fetch funding rate for {symbol}: {e}")
        
        self._last_update = datetime.now()
    
    def _fetch_funding_rate(self, symbol: str) -> Optional[FundingRateData]:
        """
        Fetch funding rate from exchange.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            FundingRateData or None
        """
        try:
            # Get ticker which includes funding info
            ticker = self.client.get_ticker(symbol)
            
            if not ticker:
                return None
            
            # Extract funding rate (may be in different fields depending on API)
            funding_rate = float(ticker.get('funding_rate', 0) or 0)
            predicted_rate = ticker.get('predicted_funding_rate')
            next_funding = ticker.get('next_funding_realization')
            
            # Parse next funding time
            next_funding_time = datetime.now() + timedelta(hours=8)  # Default
            if next_funding:
                # Convert from microseconds if needed
                if isinstance(next_funding, (int, float)):
                    next_funding_time = datetime.fromtimestamp(next_funding / 1_000_000)
            
            return FundingRateData(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_rate_8h=funding_rate,
                predicted_rate=float(predicted_rate) if predicted_rate else None,
                next_funding_time=next_funding_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            log.error(f"Error parsing funding data for {symbol}: {e}")
            return None
    
    def find_arbitrage_opportunities(self) -> List[FundingRateData]:
        """
        Find symbols with favorable funding rates for arbitrage.
        
        Returns:
            List of symbols with funding rates above threshold
        """
        opportunities = []
        
        rates = self.get_all_rates()
        for symbol, rate_data in rates.items():
            if abs(rate_data.funding_rate) >= self.funding_threshold:
                opportunities.append(rate_data)
                log.info(f"Funding opportunity: {symbol} = {rate_data.funding_rate:.4%} "
                        f"(annualized: {rate_data.annualized_rate:.2%})")
        
        # Sort by absolute funding rate (highest first)
        opportunities.sort(key=lambda x: abs(x.funding_rate), reverse=True)
        
        return opportunities
    
    def calculate_potential_income(self, symbol: str, 
                                   position_size: float,
                                   days: int = 30) -> Dict[str, float]:
        """
        Calculate potential funding income over a period.
        
        Args:
            symbol: Trading symbol
            position_size: Position size in USD
            days: Number of days to project
            
        Returns:
            Dictionary with income projections
        """
        rate_data = self.get_funding_rate(symbol)
        
        if not rate_data:
            return {'daily': 0.0, 'monthly': 0.0, 'annual': 0.0}
        
        daily_income = rate_data.estimate_daily_income(position_size)
        
        return {
            'daily': daily_income,
            'monthly': daily_income * 30,
            'annual': daily_income * 365,
            'funding_rate': rate_data.funding_rate,
            'annualized_rate': rate_data.annualized_rate
        }
    
    def get_best_opportunity(self) -> Optional[FundingRateData]:
        """
        Get the single best arbitrage opportunity.
        
        Returns:
            FundingRateData for best opportunity or None
        """
        opportunities = self.find_arbitrage_opportunities()
        return opportunities[0] if opportunities else None
    
    def record_funding_received(self, amount: float, symbol: str) -> None:
        """
        Record funding payment received.
        
        Args:
            amount: Funding amount in USD
            symbol: Symbol the funding was for
        """
        self.total_funding_earned += amount
        log.info(f"Funding received: ${amount:.4f} for {symbol} "
                f"(Total: ${self.total_funding_earned:.2f})")
    
    def get_history(self, symbol: str) -> Optional[FundingHistory]:
        """
        Get funding history for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            FundingHistory or None
        """
        return self._history.get(symbol)
    
    def get_status(self) -> Dict:
        """
        Get current status of funding monitor.
        
        Returns:
            Dictionary with status information
        """
        return {
            'symbols_monitored': len(self.symbols),
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'total_funding_earned': self.total_funding_earned,
            'current_opportunities': len(self.find_arbitrage_opportunities()),
            'rates': {
                symbol: {
                    'rate': data.funding_rate,
                    'direction': data.direction.value,
                    'annualized': data.annualized_rate
                }
                for symbol, data in self._current_rates.items()
            }
        }
