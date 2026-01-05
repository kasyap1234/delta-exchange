"""
Position Monitor Module.
Handles dynamic stop management for open positions:
- Break-even stops after 1.0R profit
- Trailing stops after 1.5R profit using ATR
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings
from src.delta_client import DeltaExchangeClient, Position
from src.risk_manager import RiskManager
from src.technical_analysis import TechnicalAnalyzer
from utils.logger import log


@dataclass
class TrackedPosition:
    """Tracking data for an open position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    initial_stop_loss: float
    current_stop_loss: float
    take_profit: float
    side: str  # 'long' or 'short'
    size: float
    
    # Stop management state
    break_even_activated: bool = False
    trailing_activated: bool = False
    highest_price: float = 0.0  # For long trailing
    lowest_price: float = float('inf')  # For short trailing


class PositionMonitor:
    """
    Monitors open positions and manages dynamic stops.
    
    Features:
    - Break-even stop: Move stop to entry + buffer after 1.0R profit
    - Trailing stop: Trail at ATR distance after 1.5R profit
    - Integrates with exchange bracket orders
    """
    
    def __init__(self, client: DeltaExchangeClient, risk_manager: RiskManager):
        """
        Initialize position monitor.
        
        Args:
            client: Delta Exchange API client for order updates
            risk_manager: Risk manager for stop calculations
        """
        self.client = client
        self.risk_manager = risk_manager
        self.analyzer = TechnicalAnalyzer()
        
        # Track positions we're monitoring
        self.tracked_positions: Dict[str, TrackedPosition] = {}
        
        # Configuration
        self.break_even_trigger_r = settings.enhanced_risk.break_even_trigger_r  # 1.0
        self.trailing_trigger_r = settings.enhanced_risk.trailing_trigger_r  # 1.5
        self.break_even_buffer_pct = settings.enhanced_risk.break_even_buffer_pct  # 0.001
        self.atr_multiplier = settings.enhanced_risk.atr_trailing_multiplier  # 2.0
    
    def start_tracking(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        side: str
    ) -> None:
        """Start tracking a new position."""
        self.tracked_positions[symbol] = TrackedPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=datetime.now(),
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            take_profit=take_profit,
            side=side,
            size=size,
            highest_price=entry_price if side == 'long' else 0.0,
            lowest_price=entry_price if side == 'short' else float('inf'),
        )
        log.info(f"Now tracking {symbol}: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
    
    def stop_tracking(self, symbol: str) -> None:
        """Stop tracking a position (closed)."""
        if symbol in self.tracked_positions:
            del self.tracked_positions[symbol]
            log.info(f"Stopped tracking {symbol}")
    
    def update_positions(self, exchange_positions: List[Position]) -> List[Tuple[str, float, str]]:
        """
        Update all tracked positions and return stop updates needed.
        
        Args:
            exchange_positions: Current positions from exchange
            
        Returns:
            List of (symbol, new_stop_price, reason) tuples for positions needing updates
        """
        stop_updates = []
        
        # Get current prices and update tracking
        for pos in exchange_positions:
            symbol = pos.product_symbol
            current_price = float(pos.mark_price or pos.entry_price)
            
            if symbol not in self.tracked_positions:
                # Not a position we're tracking (maybe opened before monitor started)
                continue
            
            tracked = self.tracked_positions[symbol]
            
            # Update high/low watermarks
            if tracked.side == 'long':
                if current_price > tracked.highest_price:
                    tracked.highest_price = current_price
            else:
                if current_price < tracked.lowest_price:
                    tracked.lowest_price = current_price
            
            # Calculate current R-multiple
            r_multiple = self.risk_manager.get_r_multiple(
                entry_price=tracked.entry_price,
                current_price=current_price,
                stop_loss=tracked.initial_stop_loss,
                side="buy" if tracked.side == "long" else "sell"
            )
            
            new_stop = None
            reason = ""
            
            # Check for break-even trigger (1.0R)
            if not tracked.break_even_activated and r_multiple >= self.break_even_trigger_r:
                new_stop = self.risk_manager.calculate_break_even_stop(
                    entry_price=tracked.entry_price,
                    side=tracked.side,
                    buffer_pct=self.break_even_buffer_pct
                )
                
                # Only update if it improves the stop
                if self._is_better_stop(tracked.side, new_stop, tracked.current_stop_loss):
                    tracked.current_stop_loss = new_stop
                    tracked.break_even_activated = True
                    reason = f"Break-even at {r_multiple:.1f}R"
                    stop_updates.append((symbol, new_stop, reason))
                    log.info(f"{symbol}: Moving stop to break-even @ {new_stop:.2f}")
            
            # Check for trailing trigger (1.5R)
            elif r_multiple >= self.trailing_trigger_r:
                # Get current ATR for trailing distance
                atr = self._get_atr(symbol)
                if atr is None:
                    continue
                
                new_stop = self.risk_manager.calculate_trailing_stop(
                    entry_price=tracked.entry_price,
                    current_price=current_price,
                    atr=atr,
                    side=tracked.side,
                    multiplier=self.atr_multiplier
                )
                
                # Only update if it improves the stop (trails in profit direction)
                if self._is_better_stop(tracked.side, new_stop, tracked.current_stop_loss):
                    tracked.current_stop_loss = new_stop
                    tracked.trailing_activated = True
                    reason = f"Trailing at {r_multiple:.1f}R"
                    stop_updates.append((symbol, new_stop, reason))
                    log.debug(f"{symbol}: Trailing stop to {new_stop:.2f} at {r_multiple:.1f}R")
        
        return stop_updates
    
    def _is_better_stop(self, side: str, new_stop: float, current_stop: float) -> bool:
        """Check if new stop is an improvement (moved in profit direction)."""
        if side == 'long':
            return new_stop > current_stop
        else:
            return new_stop < current_stop
    
    def _get_atr(self, symbol: str) -> Optional[float]:
        """Get current ATR for a symbol."""
        try:
            candles = self.client.get_candles(symbol=symbol, resolution="15m")
            if len(candles) < 20:
                return None
            
            high = np.array([c.high for c in candles])
            low = np.array([c.low for c in candles])
            close = np.array([c.close for c in candles])
            
            return self.analyzer.calculate_atr(high, low, close)
        except Exception as e:
            log.debug(f"ATR calculation failed for {symbol}: {e}")
            return None
    
    def apply_stop_updates(self, stop_updates: List[Tuple[str, float, str]]) -> None:
        """
        Apply stop updates to exchange bracket orders.
        
        Args:
            stop_updates: List of (symbol, new_stop_price, reason) tuples
        """
        for symbol, new_stop, reason in stop_updates:
            try:
                product_id = self.client.get_product_id(symbol)
                
                # Get current position to determine take profit
                tracked = self.tracked_positions.get(symbol)
                if not tracked:
                    continue
                
                # Update bracket order with new stop loss
                self.client.place_bracket_order(
                    product_id=product_id,
                    stop_loss_price=new_stop,
                    take_profit_price=tracked.take_profit,
                    trail_amount=None,
                )
                
                log.info(f"Updated bracket order for {symbol}: SL={new_stop:.2f} ({reason})")
                
            except Exception as e:
                log.error(f"Failed to update bracket order for {symbol}: {e}")
    
    def get_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            "tracked_positions": len(self.tracked_positions),
            "break_even_active": sum(1 for p in self.tracked_positions.values() if p.break_even_activated),
            "trailing_active": sum(1 for p in self.tracked_positions.values() if p.trailing_activated),
            "positions": {
                sym: {
                    "entry": pos.entry_price,
                    "current_sl": pos.current_stop_loss,
                    "break_even": pos.break_even_activated,
                    "trailing": pos.trailing_activated,
                }
                for sym, pos in self.tracked_positions.items()
            }
        }
