"""
Multi-Timeframe Analyzer Module.
Provides higher timeframe trend analysis for MTF strategy.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np

from src.delta_client import DeltaExchangeClient
from src.technical_analysis import TechnicalAnalyzer
from config.settings import settings
from utils.logger import log


@dataclass
class MTFAnalysisResult:
    """Result of multi-timeframe analysis."""
    symbol: str
    higher_tf: str
    entry_tf: str
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0.0 to 1.0
    ema_aligned: bool
    price_above_ema: bool
    recommendation: str  # 'long', 'short', 'hold'
    confidence: float


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to determine trend direction and entry timing.
    
    Uses higher timeframe (e.g., 4H) for trend direction and
    lower timeframe (e.g., 15m) for entry signals.
    """
    
    def __init__(self, client: DeltaExchangeClient, 
                 higher_tf: str = "4h", 
                 entry_tf: str = "15m"):
        """
        Initialize the MTF analyzer.
        
        Args:
            client: Delta Exchange API client
            higher_tf: Higher timeframe for trend (default: 4h)
            entry_tf: Entry timeframe for signals (default: 15m)
        """
        self.client = client
        self.higher_tf = higher_tf
        self.entry_tf = entry_tf
        self.analyzer = TechnicalAnalyzer(settings.trading)
        
        # EMA periods for trend detection
        self.ema_short = getattr(settings.mtf, 'trend_ema_short', 50)
        self.ema_long = getattr(settings.mtf, 'trend_ema_long', 200)
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis on a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'SOLUSD')
            
        Returns:
            Dictionary with trend and recommendation data
        """
        try:
            # Fetch higher timeframe candles (uses candle_count from settings by default)
            higher_candles = self.client.get_candles(
                symbol=symbol,
                resolution=self.higher_tf
            )
            
            if not higher_candles or len(higher_candles) < 50:
                log.warning(f"Insufficient {self.higher_tf} data for {symbol}")
                return self._neutral_dict(symbol)
            
            # Extract price data
            closes = np.array([c.close for c in higher_candles])
            highs = np.array([c.high for c in higher_candles])
            lows = np.array([c.low for c in higher_candles])
            
            # Calculate EMAs
            ema_short = self._calculate_ema(closes, self.ema_short)
            ema_long = self._calculate_ema(closes, self.ema_long)
            
            current_price = closes[-1]
            current_ema_short = ema_short[-1]
            current_ema_long = ema_long[-1]
            
            # Calculate ATR for position sizing
            atr = self.analyzer.calculate_atr(highs, lows, closes)
            
            # Determine trend direction
            ema_aligned = current_ema_short > current_ema_long
            price_above_ema = current_price > current_ema_short
            
            if ema_aligned and price_above_ema:
                higher_tf_trend = "bullish"
                entry_signal = "buy"
                should_trade = True
                trend_strength = min(1.0, (current_ema_short - current_ema_long) / current_ema_long * 100)
            elif not ema_aligned and not price_above_ema:
                higher_tf_trend = "bearish"
                entry_signal = "sell"
                should_trade = True
                trend_strength = min(1.0, (current_ema_long - current_ema_short) / current_ema_long * 100)
            else:
                higher_tf_trend = "neutral"
                entry_signal = "hold"
                should_trade = False
                trend_strength = 0.0
            
            # Calculate confidence based on trend alignment
            confidence = 0.5
            if ema_aligned == price_above_ema:
                confidence = 0.7
            if trend_strength > 0.5:
                confidence = 0.85
            
            return {
                'symbol': symbol,
                'should_trade': should_trade,
                'higher_tf_trend': higher_tf_trend,
                'entry_signal': entry_signal,
                'current_price': current_price,
                'atr': atr,
                'entry_confidence': confidence,
                'agreement_count': 2 if should_trade else 0,
                'trend_strength': abs(trend_strength),
                'ema_aligned': ema_aligned,
                'price_above_ema': price_above_ema,
                'indicators': []
            }
            
        except Exception as e:
            log.error(f"MTF analysis failed for {symbol}: {e}")
            return self._neutral_dict(symbol)
    
    def _neutral_dict(self, symbol: str) -> Dict[str, Any]:
        """Return a neutral result dictionary when analysis fails."""
        return {
            'symbol': symbol,
            'should_trade': False,
            'higher_tf_trend': 'neutral',
            'entry_signal': 'hold',
            'current_price': 0,
            'atr': 0,
            'entry_confidence': 0.0,
            'agreement_count': 0,
            'trend_strength': 0.0,
            'ema_aligned': False,
            'price_above_ema': False,
            'indicators': []
        }
    
    def get_trend(self, symbol: str) -> str:
        """
        Get simple trend direction for a symbol.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        result = self.analyze(symbol)
        return result.trend_direction
    
    def _neutral_result(self, symbol: str) -> MTFAnalysisResult:
        """Return a neutral result when analysis fails."""
        return MTFAnalysisResult(
            symbol=symbol,
            higher_tf=self.higher_tf,
            entry_tf=self.entry_tf,
            trend_direction="neutral",
            trend_strength=0.0,
            ema_aligned=False,
            price_above_ema=False,
            recommendation="hold",
            confidence=0.0
        )
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        
        # Initialize with SMA
        if len(data) >= period:
            ema[period - 1] = np.mean(data[:period])
            
            # Calculate EMA
            for i in range(period, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
        
        return ema
