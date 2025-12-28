"""
Trading Strategy Module.
Combines technical analysis with risk management to generate actionable trade decisions.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import numpy as np

from src.delta_client import DeltaExchangeClient, Candle, Position, OrderSide
from src.technical_analysis import TechnicalAnalyzer, TechnicalAnalysisResult, Signal
from src.risk_manager import RiskManager, PositionSizing, TradeRisk
from config.settings import settings
from utils.logger import log


class TradeAction(str, Enum):
    """Possible trading actions."""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


@dataclass
class TradeDecision:
    """Complete trade decision with all necessary information."""
    symbol: str
    action: TradeAction
    signal: Signal
    confidence: float
    entry_price: float
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    risk_assessment: Optional[TradeRisk] = None
    ta_result: Optional[TechnicalAnalysisResult] = None


class TradingStrategy:
    """
    Main trading strategy combining technical analysis and risk management.
    
    Strategy Logic:
    1. Fetch candle data for each trading pair
    2. Run technical analysis (RSI, MACD, BB, EMA)
    3. Generate combined signal
    4. Check risk limits
    5. Output trade decision
    
    Conservative approach:
    - Only trade when 3+ indicators agree
    - Use 15m candles for fewer false signals
    - Always respect stop-loss and take-profit levels
    """
    
    def __init__(self, client: DeltaExchangeClient):
        """
        Initialize trading strategy.
        
        Args:
            client: Delta Exchange API client
        """
        self.client = client
        self.analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.config = settings.trading
        
        # Track active positions for each symbol
        self.active_positions: Dict[str, dict] = {}
    
    def analyze_market(self, symbol: str) -> Optional[TechnicalAnalysisResult]:
        """
        Fetch market data and run technical analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            TechnicalAnalysisResult or None if analysis fails
        """
        try:
            # Fetch candles
            candles = self.client.get_candles(
                symbol=symbol,
                resolution=self.config.candle_interval
            )
            
            if len(candles) < 50:
                log.warning(f"Insufficient candles for {symbol}: {len(candles)}")
                return None
            
            # Convert to numpy arrays
            close = np.array([c.close for c in candles])
            high = np.array([c.high for c in candles])
            low = np.array([c.low for c in candles])
            
            # Run technical analysis
            result = self.analyzer.analyze(close, high, low, symbol)
            
            return result
            
        except Exception as e:
            log.error(f"Market analysis failed for {symbol}: {e}")
            return None
    
    def make_decision(self, symbol: str, available_balance: float,
                      current_positions: List[Position]) -> TradeDecision:
        """
        Make a trading decision for a symbol.
        
        Args:
            symbol: Trading pair symbol
            available_balance: Available capital
            current_positions: List of current open positions
            
        Returns:
            TradeDecision with action and details
        """
        # Run technical analysis
        ta_result = self.analyze_market(symbol)
        
        if ta_result is None:
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=Signal.HOLD,
                confidence=0.0,
                entry_price=0.0,
                reason="Technical analysis failed"
            )
        
        # Get current price
        try:
            ticker = self.client.get_ticker(symbol)
            current_price = float(ticker.get('mark_price', 0) or ticker.get('close', 0))
        except Exception as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=0.0,
                reason=f"Failed to get price: {e}"
            )
        
        # Check for existing position in this symbol
        existing_position = None
        for pos in current_positions:
            if pos.product_symbol == symbol and pos.size != 0:
                existing_position = pos
                break
        
        # Determine action based on signal and position
        if existing_position:
            return self._decide_with_position(
                symbol, ta_result, existing_position, current_price
            )
        else:
            return self._decide_without_position(
                symbol, ta_result, current_price, available_balance, 
                len(current_positions)
            )
    
    def _decide_with_position(self, symbol: str, 
                               ta_result: TechnicalAnalysisResult,
                               position: Position,
                               current_price: float) -> TradeDecision:
        """Make decision when we have an existing position."""
        is_long = position.size > 0
        
        # Check stop-loss and take-profit
        should_close, exit_reason = self.risk_manager.should_close_position(
            entry_price=position.entry_price,
            current_price=current_price,
            side='buy' if is_long else 'sell'
        )
        
        if should_close:
            action = TradeAction.CLOSE_LONG if is_long else TradeAction.CLOSE_SHORT
            return TradeDecision(
                symbol=symbol,
                action=action,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason=f"Exit triggered: {exit_reason.value if exit_reason else 'unknown'}",
                ta_result=ta_result
            )
        
        # Check if signal reversed
        if is_long and self.analyzer.should_close_long(ta_result):
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.CLOSE_LONG,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason="Signal reversed to bearish",
                ta_result=ta_result
            )
        elif not is_long and self.analyzer.should_close_short(ta_result):
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.CLOSE_SHORT,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason="Signal reversed to bullish",
                ta_result=ta_result
            )
        
        # Hold position
        return TradeDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            entry_price=current_price,
            reason="Maintaining position",
            ta_result=ta_result
        )
    
    def _decide_without_position(self, symbol: str,
                                  ta_result: TechnicalAnalysisResult,
                                  current_price: float,
                                  available_balance: float,
                                  current_positions_count: int) -> TradeDecision:
        """Make decision when we don't have a position."""
        # Assess risk
        risk_assessment = self.risk_manager.assess_trade_risk(
            available_balance=available_balance,
            current_positions=current_positions_count
        )
        
        if not risk_assessment.can_trade:
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason=f"Risk limit: {risk_assessment.reason}",
                risk_assessment=risk_assessment,
                ta_result=ta_result
            )
        
        # Check for buy signal
        if self.analyzer.should_enter_long(ta_result):
            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side='buy',
                entry_price=current_price,
                available_balance=available_balance
            )
            
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.OPEN_LONG,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                position_size=sizing.size,
                stop_loss=sizing.stop_loss_price,
                take_profit=sizing.take_profit_price,
                reason=f"Bullish signal ({ta_result.signal_strength}/4 indicators)",
                risk_assessment=risk_assessment,
                ta_result=ta_result
            )
        
        # Check for sell/short signal
        if self.analyzer.should_enter_short(ta_result):
            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side='sell',
                entry_price=current_price,
                available_balance=available_balance
            )
            
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.OPEN_SHORT,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                position_size=sizing.size,
                stop_loss=sizing.stop_loss_price,
                take_profit=sizing.take_profit_price,
                reason=f"Bearish signal ({ta_result.signal_strength}/4 indicators)",
                risk_assessment=risk_assessment,
                ta_result=ta_result
            )
        
        # No trade signal
        return TradeDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            entry_price=current_price,
            reason=f"No strong signal ({ta_result.signal_strength}/4 indicators)",
            risk_assessment=risk_assessment,
            ta_result=ta_result
        )
    
    def analyze_all_pairs(self, available_balance: float,
                          current_positions: List[Position]) -> List[TradeDecision]:
        """
        Analyze all configured trading pairs and return decisions.
        
        Args:
            available_balance: Available trading capital
            current_positions: Current open positions
            
        Returns:
            List of TradeDecisions for all pairs
        """
        decisions = []
        
        for symbol in self.config.trading_pairs:
            try:
                decision = self.make_decision(
                    symbol=symbol,
                    available_balance=available_balance,
                    current_positions=current_positions
                )
                decisions.append(decision)
                
                log.info(f"Decision for {symbol}: {decision.action.value} "
                         f"(Signal: {decision.signal.value}, Confidence: {decision.confidence:.0%})")
                
            except Exception as e:
                log.error(f"Failed to analyze {symbol}: {e}")
                decisions.append(TradeDecision(
                    symbol=symbol,
                    action=TradeAction.HOLD,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    entry_price=0.0,
                    reason=f"Analysis error: {e}"
                ))
        
        return decisions
    
    def get_actionable_decisions(self, decisions: List[TradeDecision]) -> List[TradeDecision]:
        """
        Filter decisions to only return actionable ones (not HOLD).
        
        Args:
            decisions: List of all trade decisions
            
        Returns:
            List of decisions that require action
        """
        return [d for d in decisions if d.action != TradeAction.HOLD]
