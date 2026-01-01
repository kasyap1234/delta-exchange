"""
Correlation Calculator Module.
Calculates rolling correlation between trading pairs for hedge ratio optimization.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

from src.delta_client import DeltaExchangeClient, Candle
from utils.logger import log


@dataclass
class CorrelationResult:
    """Result of correlation calculation between two assets."""
    symbol_a: str
    symbol_b: str
    correlation: float  # -1.0 to 1.0
    period_hours: int
    sample_count: int
    timestamp: datetime
    is_reliable: bool  # True if enough samples
    
    @property
    def is_strong_positive(self) -> bool:
        """Check if correlation is strong positive (>0.7)."""
        return self.correlation >= 0.7
    
    @property
    def is_weak(self) -> bool:
        """Check if correlation is weak (<0.5)."""
        return abs(self.correlation) < 0.5
    
    def get_hedge_ratio(self, base_ratio: float = 0.3) -> float:
        """
        Calculate optimal hedge ratio based on correlation.
        
        FIXED: When correlation is sufficient (>0.6), use the full base ratio.
        Only reduce or skip hedge when correlation is too weak.
        
        Args:
            base_ratio: Base hedge ratio to use when correlation is sufficient
            
        Returns:
            Hedge ratio (0.0 if correlation too weak, else base_ratio)
        """
        if not self.is_reliable:
            return 0.0
        
        if self.correlation < 0.5:
            return 0.0  # Too weak to hedge
        
        if self.correlation < 0.6:
            # Borderline - use reduced hedge
            return base_ratio * 0.5
        
        # Correlation >= 0.6: Use full base ratio
        # (Previously: return base_ratio * self.correlation caused under-hedging)
        return base_ratio


class CorrelationCalculator:
    """
    Calculates and tracks correlation between trading pairs.
    
    Used for determining optimal hedge ratios in the correlated
    pair hedging strategy.
    """
    
    # Default hedge pairs
    DEFAULT_PAIRS = {
        'BTCUSD': 'ETHUSD',
        'ETHUSD': 'BTCUSD',
        'SOLUSD': 'ETHUSD',
    }
    
    def __init__(self, client: DeltaExchangeClient,
                 min_correlation: float = 0.6,
                 lookback_hours: int = 24):
        """
        Initialize correlation calculator.
        
        Args:
            client: Delta Exchange API client
            min_correlation: Minimum correlation to enable hedging
            lookback_hours: Hours of data to use for correlation
        """
        self.client = client
        self.min_correlation = min_correlation
        self.lookback_hours = lookback_hours
        
        # Cache for correlations
        self._cache: Dict[str, CorrelationResult] = {}
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
    
    def get_hedge_pair(self, symbol: str) -> Optional[str]:
        """
        Get the hedge pair for a symbol.
        
        Args:
            symbol: Primary trading symbol
            
        Returns:
            Hedge symbol or None if not configured
        """
        return self.DEFAULT_PAIRS.get(symbol)
    
    def calculate_correlation(self, symbol_a: str, symbol_b: str,
                              resolution: str = '15m') -> CorrelationResult:
        """
        Calculate correlation between two symbols with strict time alignment.
        Uses Pearson correlation on 15m returns (96 samples/24h).
        """
        cache_key = f"{symbol_a}_{symbol_b}"
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached.timestamp < self._cache_ttl:
                return cached
        
        try:
            # Fetch candles (use 15m for better sample size: 24h * 4 = 96 samples)
            candles_a = self.client.get_candles(symbol=symbol_a, resolution=resolution)
            candles_b = self.client.get_candles(symbol=symbol_b, resolution=resolution)
            
            # --- STRICT TIME ALIGNMENT ---
            # Dict mapping timestamp -> close_price
            prices_a = {c.timestamp: c.close for c in candles_a}
            prices_b = {c.timestamp: c.close for c in candles_b}
            
            # Find common timestamps
            common_times = sorted(list(set(prices_a.keys()) & set(prices_b.keys())))
            
            if len(common_times) < 20: # Minimum 20 samples (5 hours of 15m data)
                log.warning(f"Insufficient aligned data for {symbol_a}/{symbol_b}: {len(common_times)} samples")
                return self._empty_result(symbol_a, symbol_b)

            # Create aligned price arrays
            aligned_a = np.array([prices_a[t] for t in common_times])
            aligned_b = np.array([prices_b[t] for t in common_times])
            
            # Calculate returns on aligned prices
            returns_a = np.diff(aligned_a) / aligned_a[:-1]
            returns_b = np.diff(aligned_b) / aligned_b[:-1]
            
            # Recalculate length after diff
            min_len = len(returns_a)
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(returns_a, returns_b)[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            result = CorrelationResult(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                correlation=float(correlation),
                period_hours=self.lookback_hours,
                sample_count=min_len,
                timestamp=datetime.now(),
                is_reliable=min_len >= 50 # Require ~12h of overlap
            )
            
            # Cache result
            self._cache[cache_key] = result
            log.debug(f"Correlation {symbol_a}/{symbol_b}: {correlation:.3f} (samples: {min_len})")
            return result
            
        except Exception as e:
            log.error(f"Correlation calculation failed: {e}")
            return self._empty_result(symbol_a, symbol_b)

    def _empty_result(self, symbol_a: str, symbol_b: str) -> CorrelationResult:
        """Helper to return empty result."""
        return CorrelationResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=0.0,
            period_hours=self.lookback_hours,
            sample_count=0,
            timestamp=datetime.now(),
            is_reliable=False
        )

    def _calculate_returns(self, candles: List[Candle]) -> np.ndarray:
        """Deprecated: Use internal alignment logic in calculate_correlation."""
        pass
    
    def get_all_correlations(self, symbols: List[str]) -> Dict[str, CorrelationResult]:
        """
        Calculate correlations for all configured pairs.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary of correlation results keyed by pair string
        """
        results = {}
        
        for symbol in symbols:
            hedge_symbol = self.get_hedge_pair(symbol)
            if hedge_symbol and hedge_symbol in symbols:
                result = self.calculate_correlation(symbol, hedge_symbol)
                results[f"{symbol}_{hedge_symbol}"] = result
        
        return results
    
    def should_hedge(self, symbol: str, symbol_b: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if hedging should be used for a symbol.
        
        Args:
            symbol: Primary trading symbol
            symbol_b: Optional hedge symbol (uses default if not provided)
            
        Returns:
            Tuple of (should_hedge, hedge_ratio)
        """
        hedge_symbol = symbol_b or self.get_hedge_pair(symbol)
        
        if not hedge_symbol:
            return False, 0.0
        
        correlation = self.calculate_correlation(symbol, hedge_symbol)
        
        if not correlation.is_reliable:
            log.warning(f"Correlation not reliable for {symbol}/{hedge_symbol}")
            return False, 0.0
        
        if correlation.correlation < self.min_correlation:
            log.info(f"Correlation too low for hedging: {correlation.correlation:.3f}")
            return False, 0.0
        
        hedge_ratio = correlation.get_hedge_ratio()
        return True, hedge_ratio
    
    def clear_cache(self) -> None:
        """Clear the correlation cache."""
        self._cache.clear()
