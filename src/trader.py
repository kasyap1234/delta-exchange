"""
Trade Executor Module.
Handles actual order execution based on trade decisions.
"""

from typing import Optional, Dict
from datetime import datetime

from src.delta_client import DeltaExchangeClient, Order, OrderSide, OrderType
from src.strategy import TradeDecision, TradeAction
from src.risk_manager import RiskManager
from config.settings import settings
from utils.logger import log


class TradeExecutor:
    """
    Executes trades based on strategy decisions.

    Responsibilities:
    - Place buy/sell orders
    - Manage open positions
    - Track trade history
    - Handle order errors
    """

    def __init__(self, client: DeltaExchangeClient, dry_run: bool = False):
        """
        Initialize trade executor.

        Args:
            client: Delta Exchange API client
            dry_run: If True, log trades but don't execute
        """
        self.client = client
        self.risk_manager = RiskManager()
        self.dry_run = dry_run

        # Track executed trades
        self.trade_history: list = []

        log.info(f"TradeExecutor initialized (dry_run={dry_run})")

    def execute_decision(self, decision: TradeDecision) -> Optional[Order]:
        """
        Execute a trading decision.

        Args:
            decision: TradeDecision from strategy

        Returns:
            Order if executed, None otherwise
        """
        if decision.action == TradeAction.HOLD:
            log.debug(f"HOLD for {decision.symbol}: {decision.reason}")
            return None

        log.info(f"TRADE: Executing {decision.action.value} for {decision.symbol}")
        log.info(
            f"  Signal: {decision.signal.value}, Confidence: {decision.confidence:.0%}"
        )
        log.info(f"  Entry Price: {decision.entry_price:.2f}")

        if decision.position_size:
            log.info(f"  Position Size: {decision.position_size:.6f}")
        if decision.stop_loss:
            log.info(f"  Stop Loss: {decision.stop_loss:.2f}")
        if decision.take_profit:
            log.info(f"  Take Profit: {decision.take_profit:.2f}")

        if self.dry_run:
            log.info(f"TRADE: [DRY RUN] Would execute {decision.action.value}")
            self._record_trade(decision, None, dry_run=True)
            return None

        try:
            order = self._execute_order(decision)
            self._record_trade(decision, order)
            return order
        except Exception as e:
            log.error(f"TRADE: Order execution failed: {e}")
            return None

    def _execute_order(self, decision: TradeDecision) -> Order:
        """Execute the actual order on the exchange."""

        if decision.action == TradeAction.OPEN_LONG:
            # Use limit order for better fill price (0.1% below current price)
            limit_price = decision.entry_price * 0.999  # 0.1% below market
            entry_order = self.client.place_order(
                symbol=decision.symbol,
                side=OrderSide.BUY,
                size=decision.position_size,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )

            # Place bracket order for SL/TP (exchange-side protection)
            if entry_order and decision.stop_loss:
                try:
                    product_id = self.client.get_product_id(decision.symbol)
                    trail_amount = (
                        decision.atr * 2
                        if hasattr(decision, "atr") and decision.atr
                        else None
                    )

                    self.client.place_bracket_order(
                        product_id=product_id,
                        stop_loss_price=decision.stop_loss,
                        take_profit_price=decision.take_profit,
                        trail_amount=trail_amount,
                    )
                    log.info(
                        f"Bracket order placed: SL={decision.stop_loss}, TP={decision.take_profit}"
                    )
                except Exception as e:
                    log.error(f"Failed to place bracket order: {e}")

            return entry_order

        elif decision.action == TradeAction.OPEN_SHORT:
            # Use limit order for better fill price (0.1% above current price)
            limit_price = decision.entry_price * 1.001  # 0.1% above market
            entry_order = self.client.place_order(
                symbol=decision.symbol,
                side=OrderSide.SELL,
                size=decision.position_size or 0,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )

            # Place bracket order for SL/TP
            if entry_order and decision.stop_loss:
                try:
                    product_id = self.client.get_product_id(decision.symbol)
                    trail_amount = (
                        decision.atr * 2
                        if hasattr(decision, "atr") and decision.atr
                        else None
                    )

                    self.client.place_bracket_order(
                        product_id=product_id,
                        stop_loss_price=decision.stop_loss,
                        take_profit_price=decision.take_profit,
                        trail_amount=trail_amount,
                    )
                    log.info(
                        f"Bracket order placed: SL={decision.stop_loss}, TP={decision.take_profit}"
                    )
                except Exception as e:
                    log.error(f"Failed to place bracket order: {e}")

            return entry_order

        elif decision.action in [TradeAction.CLOSE_LONG, TradeAction.CLOSE_SHORT]:
            return self.client.close_position(decision.symbol)

        else:
            raise ValueError(f"Unknown action: {decision.action}")

    def _record_trade(
        self, decision: TradeDecision, order: Optional[Order], dry_run: bool = False
    ) -> None:
        """Record trade in history for analysis."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": decision.symbol,
            "action": decision.action.value,
            "signal": decision.signal.value,
            "confidence": decision.confidence,
            "entry_price": decision.entry_price,
            "position_size": decision.position_size,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "reason": decision.reason,
            "order_id": order.id if order else None,
            "dry_run": dry_run,
        }

        self.trade_history.append(record)

        # Log to trade-specific log file
        log.info(f"TRADE RECORDED: {record}")

    def execute_multiple(self, decisions: list) -> list:
        """
        Execute multiple trading decisions.

        Args:
            decisions: List of TradeDecision objects

        Returns:
            List of executed orders
        """
        orders = []

        for decision in decisions:
            order = self.execute_decision(decision)
            if order:
                orders.append(order)

        return orders

    def get_trade_stats(self) -> Dict:
        """Get statistics on executed trades."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "dry_run_trades": 0,
            }

        buy_trades = sum(
            1 for t in self.trade_history if t["action"] in ["open_long", "close_short"]
        )
        sell_trades = sum(
            1 for t in self.trade_history if t["action"] in ["open_short", "close_long"]
        )
        dry_runs = sum(1 for t in self.trade_history if t["dry_run"])

        return {
            "total_trades": len(self.trade_history),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "dry_run_trades": dry_runs,
            "last_trade": self.trade_history[-1] if self.trade_history else None,
        }
