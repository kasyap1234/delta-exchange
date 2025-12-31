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
        
        # Track open positions for PnL calculation
        self.open_positions: Dict[str, dict] = {}

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
            
            if order is None:
                log.error(f"TRADE: Order placement returned None")
                return None
            
            # Log order ID for audit trail
            log.info(f"TRADE: Order {order.id} placed, waiting for fill verification...")
            
            # Wait for order fill using polling
            verified_order = self.client.wait_for_order_fill(
                order.id, timeout_seconds=15.0, poll_interval=0.5
            )
            
            if verified_order is None:
                log.warning(f"TRADE: Order {order.id} verification timed out - treating as pending")
                verified_order = order
            
            # Handle fill status
            if verified_order.state in ['cancelled', 'rejected']:
                log.error(f"TRADE: Order {order.id} was {verified_order.state} - NOT recording trade")
                return None
            
            if verified_order.unfilled_size > 0:
                filled_pct = (verified_order.size - verified_order.unfilled_size) / verified_order.size * 100
                log.warning(f"TRADE: Order {order.id} partially filled: {filled_pct:.1f}% ({verified_order.size - verified_order.unfilled_size}/{verified_order.size})")
            else:
                log.info(f"TRADE: Order {order.id} fully filled")
            
            self._record_trade(decision, verified_order)
            return verified_order
            
        except Exception as e:
            log.error(f"TRADE: Order execution failed: {e}")
            return None

    def _execute_order(self, decision: TradeDecision) -> Optional[Order]:
        """Execute the actual order on the exchange."""

        if decision.action == TradeAction.OPEN_LONG:
            # Use limit order for better fill price (0.2% below market)
            limit_price = decision.entry_price * 0.998  # 0.2% below market
            
            # Validate required fields before placing order
            if decision.position_size is None:
                log.error(f"Cannot place order: position_size is None for {decision.symbol}")
                return None
            if decision.position_size <= 0:
                log.error(f"Cannot place order: position_size is invalid ({decision.position_size}) for {decision.symbol}")
                return None
            
            entry_order: Order = self.client.place_order(
                symbol=decision.symbol,
                side=OrderSide.BUY,
                size=decision.position_size,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )

            # Track open position for PnL calculation
            if entry_order:
                self.open_positions[decision.symbol] = {
                    "entry_price": decision.entry_price,
                    "entry_time": datetime.now(),
                    "position_size": decision.position_size,
                    "side": "long"
                }

            # Place bracket order for SL/TP (exchange-side protection)
            if entry_order and decision.stop_loss:
                try:
                    product_id = self.client.get_product_id(decision.symbol)
                    
                    self.client.place_bracket_order(
                        product_id=product_id,
                        stop_loss_price=decision.stop_loss,
                        take_profit_price=decision.take_profit,
                        trail_amount=None,
                    )
                    log.info(
                        f"Bracket order placed: SL={decision.stop_loss}, TP={decision.take_profit}"
                    )
                except Exception as e:
                    log.error(f"Failed to place bracket order: {e}")

            return entry_order

        elif decision.action == TradeAction.OPEN_SHORT:
            # Use limit order for better fill price (0.2% above current price)
            limit_price = decision.entry_price * 1.002  # 0.2% above market
            entry_order = self.client.place_order(
                symbol=decision.symbol,
                side=OrderSide.SELL,
                size=decision.position_size or 0,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )

            # Track open position for PnL calculation
            if entry_order and decision.position_size:
                self.open_positions[decision.symbol] = {
                    "entry_price": decision.entry_price,
                    "entry_time": datetime.now(),
                    "position_size": decision.position_size,
                    "side": "short"
                }

            # Place bracket order for SL/TP
            if entry_order and decision.stop_loss:
                try:
                    product_id = self.client.get_product_id(decision.symbol)
                    
                    self.client.place_bracket_order(
                        product_id=product_id,
                        stop_loss_price=decision.stop_loss,
                        take_profit_price=decision.take_profit,
                        trail_amount=None,
                    )
                    log.info(
                        f"Bracket order placed: SL={decision.stop_loss}, TP={decision.take_profit}"
                    )
                except Exception as e:
                    log.error(f"Failed to place bracket order: {e}")

            return entry_order

        elif decision.action in [TradeAction.CLOSE_LONG, TradeAction.CLOSE_SHORT]:
            if decision.symbol in self.open_positions:
                current_pos = self.open_positions[decision.symbol]
                side = current_pos["side"]
                entry_price = current_pos["entry_price"]
                position_size = current_pos["position_size"]
                
                if side == "long":
                    pnl = (decision.entry_price - entry_price) * position_size
                else:
                    pnl = (entry_price - decision.entry_price) * position_size
                
                decision.pnl = pnl
                decision.entry_price_for_pnl = entry_price
                
                del self.open_positions[decision.symbol]
                
                log.info(f"Closing {decision.symbol}: PnL=${pnl:.2f}")
            
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
            "entry_price_for_pnl": decision.entry_price_for_pnl or decision.entry_price,
            "position_size": decision.position_size,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "reason": decision.reason,
            "order_id": order.id if order else None,
            "dry_run": dry_run,
            "pnl": decision.pnl,
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

    def get_performance_data(self) -> Dict:
        """
        Get performance data for Kelly Criterion.
        
        Returns:
            Dictionary with total_trades, winning_trades, total_pnl for position sizing
        """
        closed_trades = [t for t in self.trade_history if t["pnl"] is not None]
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
            }

        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] or 0 for t in closed_trades)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "total_pnl": total_pnl,
        }
