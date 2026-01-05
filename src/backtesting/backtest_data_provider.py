"""
Backtest Data Provider Module.
Provides historical data to mock client during backtesting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.backtesting.data_fetcher import HistoricalData, OHLCVBar
from src.delta_client import Candle
from utils.logger import log


@dataclass
class BacktestCandle:
    """Candle format compatible with DeltaExchangeClient."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class BacktestDataProvider:
    """
    Provides historical data to mock client during backtesting.
    
    Manages multiple symbols and timeframes, tracking current position
    in the historical data for realistic simulation.
    """
    
    def __init__(self, data_dict: Dict[str, HistoricalData]):
        """
        Initialize with historical data.
        
        Args:
            data_dict: Dictionary mapping symbol -> HistoricalData
        """
        self.data = data_dict
        self.current_idx = 0
        self.warmup_bars = 50  # Minimum bars needed for indicators
        
        # Pre-calculate total bars (use minimum to ensure sync)
        if not data_dict:
            self.total_bars = 0
            return
            
        # Find minimum length to prevent out of bounds
        min_bars = min(len(d.bars) for d in data_dict.values())
        self.total_bars = min_bars
        
        log.info(f"BacktestDataProvider initialized with {len(data_dict)} symbols, "
                 f"{self.total_bars} bars each (sync length)")
    
    def set_current_bar(self, idx: int) -> None:
        """Set the current bar index for all data access."""
        self.current_idx = min(idx, self.total_bars - 1)
    
    def get_current_bar(self, symbol: str) -> Optional[OHLCVBar]:
        """Get the current bar for a symbol."""
        if symbol not in self.data:
            return None
        return self.data[symbol].bars[self.current_idx]
    
    def get_current_price(self, symbol: str) -> float:
        """Get the close price at current bar."""
        bar = self.get_current_bar(symbol)
        return bar.close if bar else 0.0
    
    def get_current_ohlcv(self, symbol: str) -> Dict[str, float]:
        """Get OHLCV data for current bar."""
        bar = self.get_current_bar(symbol)
        if not bar:
            return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}
        return {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
    
    def get_candles_up_to(self, symbol: str, count: Optional[int] = None) -> List[Candle]:
        """
        Return candles from start up to current bar.
        
        Args:
            symbol: Trading symbol
            count: Optional limit on number of candles (most recent)
            
        Returns:
            List of Candle objects
        """
        if symbol not in self.data:
            log.warning(f"Symbol {symbol} not in backtest data")
            return []
        
        bars = self.data[symbol].bars[:self.current_idx + 1]
        
        if count and len(bars) > count:
            bars = bars[-count:]
        
        # Convert OHLCVBar to Candle format
        candles = []
        for bar in bars:
            candle = Candle(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            )
            candles.append(candle)
        
        return candles
    
    def get_higher_tf_candles(self, symbol: str, resolution: str = '4h') -> List[Candle]:
        """
        Get higher timeframe candles by aggregating base data.
        
        For backtesting, we aggregate 15m bars into 4h bars.
        4h = 16 x 15m bars
        """
        base_candles = self.get_candles_up_to(symbol)
        
        if resolution == '4h':
            bars_per_candle = 16  # 16 x 15m = 4h
        elif resolution == '1h':
            bars_per_candle = 4
        elif resolution == '1d':
            bars_per_candle = 96
        else:
            bars_per_candle = 16
        
        aggregated = []
        for i in range(0, len(base_candles) - bars_per_candle + 1, bars_per_candle):
            chunk = base_candles[i:i + bars_per_candle]
            if not chunk:
                continue
            
            agg_candle = Candle(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk)
            )
            aggregated.append(agg_candle)
        
        return aggregated
    
    def get_price_arrays(self, symbol: str) -> Dict[str, np.ndarray]:
        """Get numpy arrays for technical analysis."""
        candles = self.get_candles_up_to(symbol)
        
        return {
            'open': np.array([c.open for c in candles]),
            'high': np.array([c.high for c in candles]),
            'low': np.array([c.low for c in candles]),
            'close': np.array([c.close for c in candles]),
            'volume': np.array([c.volume for c in candles])
        }
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self.data.keys())
    
    def can_continue(self) -> bool:
        """Check if there are more bars to process."""
        return self.current_idx < self.total_bars - 1
    
    def get_current_datetime(self, symbol: str) -> str:
        """Get datetime string for current bar."""
        bar = self.get_current_bar(symbol)
        return bar.datetime if bar else ""
    
    def get_progress(self) -> float:
        """Get backtest progress as percentage."""
        if self.total_bars == 0:
            return 100.0
        return (self.current_idx / self.total_bars) * 100
