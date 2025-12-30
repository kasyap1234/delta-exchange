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
from src.hedging.hedge_manager import HedgeManager, HedgedPosition
from src.hedging.correlation import CorrelationCalculator
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

    def __init__(self, client: DeltaExchangeClient, dry_run: bool = False):
        """
        Initialize trading strategy.

        Args:
            client: Delta Exchange API client
            dry_run: If True, don't execute real trades (for hedging too)
        """
        self.client = client
        self.analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.config = settings.trading
        self.dry_run = dry_run

        # Initialize hedging components
        self.correlation_calc = CorrelationCalculator(client)
        self.hedge_manager = HedgeManager(
            client, self.correlation_calc, default_hedge_ratio=0.3, dry_run=dry_run
        )

        # Track active positions for each symbol
        self.active_positions: Dict[str, dict] = {}

        # Track hedged positions
        self.hedged_positions: Dict[str, str] = {}  # symbol -> hedge_position_id

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
                symbol=symbol, resolution=self.config.candle_interval
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

    def make_decision(
        self, symbol: str, available_balance: float, current_positions: List[Position]
    ) -> TradeDecision:
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
                reason="Technical analysis failed",
            )

        # Get current price
        try:
            ticker = self.client.get_ticker(symbol)
            current_price = float(ticker.get("mark_price", 0) or ticker.get("close", 0))
        except Exception as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=0.0,
                reason=f"Failed to get price: {e}",
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
                symbol,
                ta_result,
                current_price,
                available_balance,
                len(current_positions),
            )

    def _decide_with_position(
        self,
        symbol: str,
        ta_result: TechnicalAnalysisResult,
        position: Position,
        current_price: float,
    ) -> TradeDecision:
        """Make decision when we have an existing position."""
        is_long = position.size > 0

        # Check stop-loss and take-profit
        should_close, exit_reason = self.risk_manager.should_close_position(
            entry_price=position.entry_price,
            current_price=current_price,
            side="buy" if is_long else "sell",
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
                ta_result=ta_result,
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
                ta_result=ta_result,
            )
        elif not is_long and self.analyzer.should_close_short(ta_result):
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.CLOSE_SHORT,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason="Signal reversed to bullish",
                ta_result=ta_result,
            )

        # Hold position
        return TradeDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            entry_price=current_price,
            reason="Maintaining position",
            ta_result=ta_result,
        )

    def _decide_without_position(
        self,
        symbol: str,
        ta_result: TechnicalAnalysisResult,
        current_price: float,
        available_balance: float,
        current_positions_count: int,
    ) -> TradeDecision:
        """Make decision when we don't have a position."""
        # Assess risk
        risk_assessment = self.risk_manager.assess_trade_risk(
            available_balance=available_balance,
            current_positions=current_positions_count,
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
                ta_result=ta_result,
            )

        # Check higher timeframe trend to prevent counter-trend entries
        htf_trend = self._check_higher_timeframe_trend(symbol)

        # Check for buy signal
        if self.analyzer.should_enter_long(ta_result):
            # Only allow LONG if higher timeframe trend is bullish or neutral
            if htf_trend == "bearish":
                log.info(
                    f"Trend filter: Skipping LONG entry for {symbol} - HTF is bearish"
                )
                return TradeDecision(
                    symbol=symbol,
                    action=TradeAction.HOLD,
                    signal=ta_result.combined_signal,
                    confidence=ta_result.confidence,
                    entry_price=current_price,
                    reason=f"Counter-trend: HTF trend is bearish",
                    risk_assessment=risk_assessment,
                    ta_result=ta_result,
                )

            # Build performance data for Kelly Criterion (TradingStrategy doesn't track performance)
            performance_data = None  # TradingStrategy doesn't have performance tracking

            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side="buy",
                entry_price=current_price,
                available_balance=available_balance,
                performance_data=performance_data,
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
                reason=f"Bullish signal ({ta_result.signal_strength}/4 indicators) + {htf_trend} HTF trend",
                risk_assessment=risk_assessment,
                ta_result=ta_result,
            )

        # Check for sell/short signal
        if self.analyzer.should_enter_short(ta_result):
            # Only allow SHORT if higher timeframe trend is bearish or neutral
            if htf_trend == "bullish":
                log.info(
                    f"Trend filter: Skipping SHORT entry for {symbol} - HTF is bullish"
                )
                return TradeDecision(
                    symbol=symbol,
                    action=TradeAction.HOLD,
                    signal=ta_result.combined_signal,
                    confidence=ta_result.confidence,
                    entry_price=current_price,
                    reason=f"Counter-trend: HTF trend is bullish",
                    risk_assessment=risk_assessment,
                    ta_result=ta_result,
                )

            # Build performance data for Kelly Criterion (TradingStrategy doesn't track performance)
            performance_data = None  # TradingStrategy doesn't have performance tracking

            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side="sell",
                entry_price=current_price,
                available_balance=available_balance,
                performance_data=performance_data,
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
                reason=f"Bearish signal ({ta_result.signal_strength}/4 indicators) + {htf_trend} HTF trend",
                risk_assessment=risk_assessment,
                ta_result=ta_result,
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
                reason=f"Bullish signal ({ta_result.signal_strength}/4 indicators) + {htf_trend} HTF trend",
                risk_assessment=risk_assessment,
                ta_result=ta_result,
            )

        # Check for sell/short signal
        if self.analyzer.should_enter_short(ta_result):
            # Only allow SHORT if higher timeframe trend is bearish or neutral
            if htf_trend == "bullish":
                log.info(
                    f"Trend filter: Skipping SHORT entry for {symbol} - HTF is bullish"
                )
                return TradeDecision(
                    symbol=symbol,
                    action=TradeAction.HOLD,
                    signal=ta_result.combined_signal,
                    confidence=ta_result.confidence,
                    entry_price=current_price,
                    reason=f"Counter-trend: HTF trend is bullish",
                    risk_assessment=risk_assessment,
                    ta_result=ta_result,
                )

            # Build performance data for Kelly Criterion
            performance_data = None
            if hasattr(self, "performance"):
                performance_data = {
                    "total_trades": self.performance.total_trades,
                    "winning_trades": self.performance.winning_trades,
                    "total_pnl": self.performance.total_pnl,
                }

            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side="sell",
                entry_price=current_price,
                available_balance=available_balance,
                performance_data=performance_data,
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
                reason=f"Bearish signal ({ta_result.signal_strength}/4 indicators) + {htf_trend} HTF trend",
                risk_assessment=risk_assessment,
                ta_result=ta_result,
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
            ta_result=ta_result,
        )

    def _check_higher_timeframe_trend(self, symbol: str) -> str:
        """
        Check higher timeframe trend to filter counter-trend entries.

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        try:
            # Fetch 4H candles for trend analysis
            htf_candles = self.client.get_candles(symbol=symbol, resolution="4h")

            if len(htf_candles) < 200:
                return "neutral"

            # Convert to numpy array
            htf_close = np.array([c.close for c in htf_candles])

            # Use trend direction from technical analyzer
            trend = self.analyzer.get_trend_direction(
                close=htf_close, ema_short=50, ema_long=200
            )

            return trend

        except Exception as e:
            log.error(f"Failed to check HTF trend for {symbol}: {e}")
            return "neutral"  # Allow trade if trend check fails

    def analyze_all_pairs(
        self, available_balance: float, current_positions: List[Position]
    ) -> List[TradeDecision]:
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
                    current_positions=current_positions,
                )
                decisions.append(decision)

                log.info(
                    f"Decision for {symbol}: {decision.action.value} "
                    f"(Signal: {decision.signal.value}, Confidence: {decision.confidence:.0%})"
                )

            except Exception as e:
                log.error(f"Failed to analyze {symbol}: {e}")
                decisions.append(
                    TradeDecision(
                        symbol=symbol,
                        action=TradeAction.HOLD,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        entry_price=0.0,
                        reason=f"Analysis error: {e}",
                    )
                )

        return decisions

    def get_actionable_decisions(
        self, decisions: List[TradeDecision]
    ) -> List[TradeDecision]:
        """
        Filter decisions to only return actionable ones (not HOLD).

        Args:
            decisions: List of all trade decisions

        Returns:
            List of decisions that require action
        """
        return [d for d in decisions if d.action != TradeAction.HOLD]

    def evaluate_hedging_opportunities(
        self, current_positions: List[Position]
    ) -> List[HedgedPosition]:
        """
        Evaluate and create hedges for existing positions based on correlation.

        Automatically hedges unhedged positions when:
        1. Position is open (size != 0)
        2. Correlation with hedge pair is > 0.6
        3. No existing hedge for the symbol

        Uses aggressive policy: Hedge ALL open positions for maximum protection.

        Args:
            current_positions: Current open positions

        Returns:
            List of newly created hedged positions
        """
        new_hedges = []

        for position in current_positions:
            # Skip if position is closed or already hedged
            if position.size == 0 or position.product_symbol in self.hedged_positions:
                continue

            # Check if we have a hedge pair configured
            hedge_symbol = self.correlation_calc.get_hedge_pair(position.product_symbol)
            if not hedge_symbol:
                log.debug(f"No hedge pair configured for {position.product_symbol}")
                continue

            # Check correlation
            should_hedge, hedge_ratio = self.correlation_calc.should_hedge(
                position.product_symbol, hedge_symbol
            )

            if not should_hedge:
                log.info(
                    f"Correlation too low for {position.product_symbol}, skipping hedge"
                )
                continue

            # Determine position side
            primary_side = "long" if position.size > 0 else "short"

            # Create hedge position
            log.info(
                f"Creating hedge for {position.product_symbol}: "
                f"{primary_side} {abs(position.size):.4f} contracts "
                f"@ ${position.entry_price:.2f} (PnL: ${position.unrealized_pnl:.2f})"
            )

            hedged_pos = self.hedge_manager.create_hedged_position(
                primary_symbol=position.product_symbol,
                primary_size=abs(position.size),
                primary_side=primary_side,
                primary_price=position.entry_price,
                hedge_symbol=hedge_symbol,
                hedge_ratio=hedge_ratio,
            )

            if hedged_pos:
                self.hedged_positions[position.product_symbol] = hedged_pos.id
                new_hedges.append(hedged_pos)

        return new_hedges

    def get_hedge_status(self) -> Dict:
        """Get status of all hedged positions."""
        return self.hedge_manager.get_status()
