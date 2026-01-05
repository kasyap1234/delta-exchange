"""
Strategy Core Module - Shared Trading Logic.

This module contains the core trading decision logic that is shared between:
- Live trading (TradingStrategy)
- Backtesting (BacktestEngine)

This ensures backtest results accurately predict live performance.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings
from src.technical_analysis import Signal, TechnicalAnalysisResult, TechnicalAnalyzer
from src.unified_signal_validator import UnifiedSignalValidator, ValidationResult
from src.risk_manager import RiskManager, PositionSizing
from utils.logger import log


class TradeDirection(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class SignalDecision:
    """
    Core trading signal decision - shared by live and backtest.
    
    This is the fundamental output of signal analysis, before
    execution-specific details are added.
    """
    symbol: str
    direction: Optional[TradeDirection]  # None = no trade
    signal: Signal
    confidence: float
    is_valid: bool
    validation_result: Optional[ValidationResult] = None
    reason: str = ""
    
    # Market context (for logging/analysis)
    htf_trend: str = "neutral"
    adx: float = 0.0
    rsi: Optional[float] = None


@dataclass  
class EntryParameters:
    """Parameters for trade entry - calculated by risk manager."""
    position_size: float
    stop_loss: float
    take_profit: float
    entry_price: float


class StrategyCore:
    """
    Core strategy logic shared by live trading and backtesting.
    
    This class contains the signal generation and validation logic
    that MUST be identical in both live and backtest contexts.
    
    Key Responsibilities:
    - Technical analysis signal generation
    - Signal validation (ADX, HTF trend, RSI, cooldowns)
    - Entry/exit decision making
    
    NOT Responsible For:
    - Order execution (handled by Trader/BacktestEngine)
    - Position monitoring (handled by PositionMonitor/BacktestEngine)
    - API calls (handled by callers)
    """
    
    def __init__(self):
        """Initialize strategy core components."""
        self.analyzer = TechnicalAnalyzer()
        self.signal_validator = UnifiedSignalValidator()
        self.risk_manager = RiskManager()
    
    def analyze_candles(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        symbol: str
    ) -> Optional[TechnicalAnalysisResult]:
        """
        Run technical analysis on price data.
        
        Args:
            close: Array of close prices
            high: Array of high prices
            low: Array of low prices
            symbol: Trading symbol
            
        Returns:
            TechnicalAnalysisResult or None if insufficient data
        """
        if len(close) < 50:
            log.warning(f"Insufficient data for {symbol}: {len(close)} bars (need 50)")
            return None
        
        return self.analyzer.analyze(close, high, low, symbol)
    
    def should_enter(
        self,
        ta_result: TechnicalAnalysisResult,
        htf_trend: str = "neutral",
        adx: float = 0.0,
        rsi: Optional[float] = None,
        volume_signal: str = "neutral",
        market_regime: str = "trending",
        timestamp: Optional[datetime] = None,
        symbol: str = ""
    ) -> SignalDecision:
        """
        Determine if we should enter a trade based on signals and validation.
        
        This is the CORE decision logic that must be identical in live and backtest.
        
        Args:
            ta_result: Technical analysis result
            htf_trend: Higher timeframe trend ("bullish", "bearish", "neutral")
            adx: ADX value for trend strength
            rsi: RSI value for overbought/oversold filtering
            volume_signal: Volume confirmation signal
            market_regime: Market regime ("trending", "ranging")
            timestamp: Current timestamp for cooldown tracking
            symbol: Trading symbol
            
        Returns:
            SignalDecision with trade direction and validation status
        """
        # Determine initial direction from basic signals
        direction = None
        if self.analyzer.should_enter_long(ta_result):
            direction = TradeDirection.LONG
        elif self.analyzer.should_enter_short(ta_result):
            direction = TradeDirection.SHORT
        
        if direction is None:
            return SignalDecision(
                symbol=symbol,
                direction=None,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                is_valid=False,
                reason="No strong signal",
                htf_trend=htf_trend,
                adx=adx,
                rsi=rsi
            )
        
        # Validate entry using unified validator
        is_valid, validation_result, reason = self.signal_validator.validate_entry(
            symbol=symbol,
            direction=direction.value,
            ta_result=ta_result,
            higher_tf_trend=htf_trend,
            adx=adx,
            rsi=rsi,
            volume_signal=volume_signal,
            market_regime=market_regime,
            timestamp=timestamp or datetime.now()
        )
        
        if not is_valid:
            log.debug(f"Signal REJECTED for {symbol} {direction.value}: {reason}")
        else:
            log.info(f"Signal APPROVED for {symbol} {direction.value}: {reason}")
        
        return SignalDecision(
            symbol=symbol,
            direction=direction if is_valid else None,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            is_valid=is_valid,
            validation_result=validation_result,
            reason=reason,
            htf_trend=htf_trend,
            adx=adx,
            rsi=rsi
        )
    
    def should_exit(
        self,
        ta_result: TechnicalAnalysisResult,
        current_direction: TradeDirection
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit an existing position.
        
        Args:
            ta_result: Technical analysis result
            current_direction: Current position direction
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if current_direction == TradeDirection.LONG:
            if self.analyzer.should_close_long(ta_result):
                return True, "Bearish signal reversal"
        else:
            if self.analyzer.should_close_short(ta_result):
                return True, "Bullish signal reversal"
        
        return False, ""
    
    def calculate_entry_params(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        available_balance: float
    ) -> EntryParameters:
        """
        Calculate entry parameters (size, SL, TP) using risk manager.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Expected entry price
            available_balance: Available capital
            
        Returns:
            EntryParameters with position sizing
        """
        side = "buy" if direction == TradeDirection.LONG else "sell"
        
        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            available_balance=available_balance,
        )
        
        return EntryParameters(
            position_size=sizing.size,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            entry_price=entry_price
        )
    
    def calculate_market_context(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate market context for signal validation.
        
        Args:
            high: Array of high prices
            low: Array of low prices  
            close: Array of close prices
            volume: Optional array of volumes
            
        Returns:
            Dictionary with adx, rsi, volume_signal, market_regime
        """
        adx = self.analyzer.calculate_adx(high, low, close)
        rsi = self.analyzer.calculate_rsi(close)
        
        # Volume signal
        volume_signal = "neutral"
        if volume is not None and len(volume) > 0:
            try:
                vol_result = self.analyzer.calculate_volume_signal(volume, close)
                volume_signal = vol_result.signal.value if hasattr(vol_result.signal, 'value') else "neutral"
            except Exception:
                pass
        
        # Market regime
        is_trending = adx > settings.signal_filtering.min_adx_for_trend
        market_regime = "trending" if is_trending else "ranging"
        
        return {
            "adx": adx,
            "rsi": rsi,
            "volume_signal": volume_signal,
            "market_regime": market_regime
        }
    
    def get_htf_trend_from_data(self, close: np.ndarray) -> str:
        """
        Determine higher timeframe trend from close prices.
        
        Args:
            close: Array of close prices (should be HTF data)
            
        Returns:
            "bullish", "bearish", or "neutral"
        """
        if len(close) < 30:
            return "neutral"
        
        return self.analyzer.get_trend_direction(close)
