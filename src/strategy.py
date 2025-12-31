"""
Trading Strategy Module.
Combines technical analysis with risk management to generate actionable trade decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from src.delta_client import Candle, DeltaExchangeClient, OrderSide, Position
from src.hedging.correlation import CorrelationCalculator
from src.hedging.hedge_manager import HedgedPosition, HedgeManager
from src.risk_manager import PositionSizing, RiskManager, TradeRisk
from src.technical_analysis import Signal, TechnicalAnalysisResult, TechnicalAnalyzer
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
    pnl: Optional[float] = None
    entry_price_for_pnl: Optional[float] = None


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
    - Only trade when multiple indicators agree
    - Use configured candle interval (default 15m) for fewer false signals
    - Always respect stop-loss and take-profit levels
    """

    def __init__(self, client: DeltaExchangeClient, executor, dry_run: bool = False):
        """
        Initialize trading strategy.

        Args:
            client: Delta Exchange API client
            executor: TradeExecutor instance for performance tracking
            dry_run: If True, don't execute real trades
        """
        self.client = client
        self.executor = executor
        self.analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.config = settings.trading
        self.dry_run = dry_run

        # Initialize hedging components
        self.correlation_calc = CorrelationCalculator(client)
        self.hedge_manager = HedgeManager(
            client, self.correlation_calc, default_hedge_ratio=0.3, dry_run=dry_run
        )

    def analyze_market(self, symbol: str) -> Optional[TechnicalAnalysisResult]:
        """
        Fetch market data and run technical analysis for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSD')

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
            if pos.product_symbol == symbol and abs(pos.size) > 0:
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

    def _should_trade(self) -> bool:
        """
        Check if trading should be enabled based on performance.

        Returns:
            True if trading should continue, False if should be disabled
        """
        if not settings.enhanced_risk.enable_auto_disable:
            return True

        # Get performance data
        perf = self.executor.get_performance_data()

        if perf["total_trades"] < 10:
            # Need minimum 10 trades to assess
            return True

        # Calculate win rate
        win_rate = (
            perf["winning_trades"] / perf["total_trades"]
            if perf["total_trades"] > 0
            else 0
        )

        # Check if win rate is below threshold
        min_win_rate = (
            settings.enhanced_risk.min_win_rate_pct / 100.0
            if settings.enhanced_risk.min_win_rate_pct > 1
            else settings.enhanced_risk.min_win_rate_pct
        )

        if win_rate < min_win_rate:
            log.warning(f"Win rate {win_rate:.1%} below threshold {min_win_rate:.1%} - trading disabled")
            return False
        return True

    def _decide_without_position(
        self,
        symbol: str,
        ta_result: TechnicalAnalysisResult,
        current_price: float,
        available_balance: float,
        current_positions_count: int,
    ) -> TradeDecision:
        """Make decision when we have no existing position."""
        # Check risk limits
        risk = self.risk_manager.assess_trade_risk(available_balance, current_positions_count)

        if not risk.can_trade:
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason=risk.reason,
                risk_assessment=risk,
            )

        # Check if trading should be enabled based on performance
        if self.executor and not self._should_trade():
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,
                signal=ta_result.combined_signal,
                confidence=ta_result.confidence,
                entry_price=current_price,
                reason="Trading disabled due to poor performance",
                risk_assessment=risk,
            )

        # Check for strong bullish signals
        if self.analyzer.should_open_long(ta_result):
            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side="buy",
                entry_price=current_price,
                available_balance=available_balance,
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
                reason="Multiple bullish signals confirmed",
                risk_assessment=risk,
                ta_result=ta_result,
            )

        # Check for strong bearish signals
        elif self.analyzer.should_open_short(ta_result):
            sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                side="sell",
                entry_price=current_price,
                available_balance=available_balance,
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
                reason="Multiple bearish signals confirmed",
                risk_assessment=risk,
                ta_result=ta_result,
            )

        # No strong signal
        return TradeDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            signal=ta_result.combined_signal,
            confidence=ta_result.confidence,
            entry_price=current_price,
            reason="No strong signal - waiting",
            ta_result=ta_result,
        )

    def analyze_all_pairs(
        self, available_balance: float, current_positions: List[Position]
    ) -> List[TradeDecision]:
        """Analyze all configured trading pairs."""
        decisions = []
        for symbol in self.config.trading_pairs:
            decision = self.make_decision(symbol, available_balance, current_positions)
            decisions.append(decision)
        return decisions

    def get_actionable_decisions(
        self, decisions: List[TradeDecision]
    ) -> List[TradeDecision]:
        """Filter decisions to only actionable ones (non-HOLD)."""
        return [d for d in decisions if d.action != TradeAction.HOLD]

    def evaluate_hedging_opportunities(
        self, positions: List[Position]
    ) -> List[HedgedPosition]:
        """Evaluate and execute hedging for losing positions."""
        return self.hedge_manager.evaluate_and_hedge(positions)

    def get_hedge_status(self) -> Dict:
        """Get current hedging status."""
        return self.hedge_manager.get_hedge_summary()
