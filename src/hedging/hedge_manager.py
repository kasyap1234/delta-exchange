"""
Hedge Manager Module.
Manages hedge positions and ensures primary/hedge positions are kept in sync.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.delta_client import DeltaExchangeClient, Position, Order, OrderSide, OrderType
from src.hedging.correlation import CorrelationCalculator, CorrelationResult
from utils.logger import log


class HedgeStatus(str, Enum):
    """Status of a hedge position."""
    ACTIVE = "active"
    PENDING = "pending"
    CLOSED = "closed"
    REBALANCING = "rebalancing"
    ERROR = "error"


@dataclass
class HedgedPosition:
    """
    Represents a primary position with its associated hedge.
    
    The hedge is designed to reduce directional exposure while
    maintaining the ability to profit from the primary trade.
    """
    id: str
    primary_symbol: str
    primary_size: float
    primary_side: str  # 'long' or 'short'
    primary_entry_price: float
    hedge_symbol: str
    hedge_size: float
    hedge_side: str
    hedge_entry_price: float
    hedge_ratio: float  # What portion is hedged (0.0-1.0)
    correlation: float
    status: HedgeStatus
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def net_exposure(self) -> float:
        """
        Calculate net directional exposure.
        
        Returns:
            Net USD exposure (primary - hedge)
        """
        primary_value = self.primary_size * self.primary_entry_price
        hedge_value = self.hedge_size * self.hedge_entry_price
        
        if self.primary_side == 'long':
            return primary_value - hedge_value
        else:
            return hedge_value - primary_value
    
    @property
    def exposure_reduction_pct(self) -> float:
        """Calculate how much exposure is reduced by hedge."""
        return self.hedge_ratio * self.correlation
    
    def calculate_pnl(self, primary_current: float, 
                      hedge_current: float) -> Dict[str, float]:
        """
        Calculate P&L for both legs.
        
        Args:
            primary_current: Current price of primary
            hedge_current: Current price of hedge
            
        Returns:
            Dictionary with PnL breakdown
        """
        # Primary P&L
        if self.primary_side == 'long':
            primary_pnl = (primary_current - self.primary_entry_price) * self.primary_size
        else:
            primary_pnl = (self.primary_entry_price - primary_current) * self.primary_size
        
        # Hedge P&L (opposite direction)
        if self.hedge_side == 'long':
            hedge_pnl = (hedge_current - self.hedge_entry_price) * self.hedge_size
        else:
            hedge_pnl = (self.hedge_entry_price - hedge_current) * self.hedge_size
        
        return {
            'primary_pnl': primary_pnl,
            'hedge_pnl': hedge_pnl,
            'net_pnl': primary_pnl + hedge_pnl,
            'hedge_effectiveness': abs(hedge_pnl / primary_pnl) if primary_pnl != 0 else 0
        }


class HedgeManager:
    """
    Manages creation, tracking, and liquidation of hedged positions.
    
    Responsibilities:
    1. Create hedge orders when primary orders are placed
    2. Track primary/hedge pairs
    3. Close hedges when primaries are closed
    4. Rebalance hedges if correlation changes
    """
    
    def __init__(self, client: DeltaExchangeClient,
                 correlation_calculator: CorrelationCalculator,
                 default_hedge_ratio: float = 0.3,
                 dry_run: bool = False):
        """
        Initialize hedge manager.
        
        Args:
            client: Delta Exchange API client
            correlation_calculator: For calculating pair correlations
            default_hedge_ratio: Default portion to hedge (0.0-1.0)
            dry_run: If True, don't execute real orders
        """
        self.client = client
        self.correlation = correlation_calculator
        self.default_hedge_ratio = default_hedge_ratio
        self.dry_run = dry_run
        
        # Track active hedged positions
        self._positions: Dict[str, HedgedPosition] = {}
        self._position_counter = 0
    
    def create_hedged_position(self, 
                               primary_symbol: str,
                               primary_size: float,
                               primary_side: str,
                               primary_price: float,
                               hedge_symbol: Optional[str] = None,
                               hedge_ratio: Optional[float] = None) -> Optional[HedgedPosition]:
        """
        Create a new hedged position.
        
        Opens both the primary and hedge positions.
        
        Args:
            primary_symbol: Symbol for primary position
            primary_size: Size for primary position
            primary_side: 'long' or 'short'
            primary_price: Entry price for primary
            hedge_symbol: Symbol for hedge (auto-detected if None)
            hedge_ratio: Portion to hedge (uses default if None)
            
        Returns:
            HedgedPosition if successful, None otherwise
        """
        # Determine hedge symbol
        if hedge_symbol is None:
            hedge_symbol = self.correlation.get_hedge_pair(primary_symbol)
        
        if not hedge_symbol:
            log.warning(f"No hedge pair configured for {primary_symbol}")
            return None
        
        # Check correlation and get optimal hedge ratio
        should_hedge, optimal_ratio = self.correlation.should_hedge(
            primary_symbol, hedge_symbol
        )
        
        if not should_hedge:
            log.warning(f"Correlation too low for hedging {primary_symbol}/{hedge_symbol}")
            return None
        
        # Use provided ratio or calculate from correlation
        effective_ratio = hedge_ratio if hedge_ratio is not None else optimal_ratio
        
        # Get hedge price
        try:
            hedge_ticker = self.client.get_ticker(hedge_symbol)
            hedge_price = float(hedge_ticker.get('mark_price', 0))
        except Exception as e:
            log.error(f"Failed to get hedge price: {e}")
            return None
        
        # Calculate hedge size based on USD value
        primary_value = primary_size * primary_price
        hedge_value = primary_value * effective_ratio
        hedge_size = hedge_value / hedge_price
        
        # Hedge is opposite direction
        hedge_side = 'short' if primary_side == 'long' else 'long'
        
        # Create position ID
        self._position_counter += 1
        position_id = f"hedge_{self._position_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Execute orders
        if not self.dry_run:
            try:
                # 1. Place Primary Order
                primary_order_side = OrderSide.BUY if primary_side == 'long' else OrderSide.SELL
                self.client.place_order(
                    symbol=primary_symbol,
                    side=primary_order_side,
                    size=primary_size,
                    order_type=OrderType.MARKET
                )
                log.info(f"Primary order placed: {primary_side.upper()} {primary_size:.6f} {primary_symbol}")

                # 2. Place Hedge Order
                hedge_order_side = OrderSide.SELL if hedge_side == 'short' else OrderSide.BUY
                self.client.place_order(
                    symbol=hedge_symbol,
                    side=hedge_order_side,
                    size=hedge_size,
                    order_type=OrderType.MARKET
                )
                log.info(f"Hedge order placed: {hedge_side.upper()} {hedge_size:.6f} {hedge_symbol}")
            except Exception as e:
                log.error(f"Failed to place hedged position orders: {e}")
                return None
        else:
            log.info(f"[DRY RUN] Would place primary: {primary_side.upper()} {primary_size:.6f} {primary_symbol}")
            log.info(f"[DRY RUN] Would place hedge: {hedge_side.upper()} {hedge_size:.6f} {hedge_symbol}")
        
        # Get current correlation
        corr_result = self.correlation.calculate_correlation(primary_symbol, hedge_symbol)
        
        # Create hedged position record
        hedged_pos = HedgedPosition(
            id=position_id,
            primary_symbol=primary_symbol,
            primary_size=primary_size,
            primary_side=primary_side,
            primary_entry_price=primary_price,
            hedge_symbol=hedge_symbol,
            hedge_size=hedge_size,
            hedge_side=hedge_side,
            hedge_entry_price=hedge_price,
            hedge_ratio=effective_ratio,
            correlation=corr_result.correlation,
            status=HedgeStatus.ACTIVE
        )
        
        self._positions[position_id] = hedged_pos
        
        log.info(f"Created hedged position {position_id}: "
                f"{primary_side.upper()} {primary_symbol} / "
                f"{hedge_side.upper()} {hedge_symbol} "
                f"(ratio: {effective_ratio:.0%}, correlation: {corr_result.correlation:.2f})")
        
        return hedged_pos
    
    def close_hedged_position(self, position_id: str) -> Tuple[bool, Dict[str, float]]:
        """
        Close both primary and hedge positions.
        
        Args:
            position_id: ID of the hedged position
            
        Returns:
            Tuple of (success, pnl_breakdown)
        """
        if position_id not in self._positions:
            log.error(f"Position not found: {position_id}")
            return False, {}
        
        position = self._positions[position_id]
        
        try:
            # Get current prices
            primary_ticker = self.client.get_ticker(position.primary_symbol)
            hedge_ticker = self.client.get_ticker(position.hedge_symbol)
            
            primary_current = float(primary_ticker.get('mark_price', 0))
            hedge_current = float(hedge_ticker.get('mark_price', 0))
            
            # Calculate P&L
            pnl = position.calculate_pnl(primary_current, hedge_current)
            
            if not self.dry_run:
                # Close primary
                self.client.close_position(position.primary_symbol)
                # Close hedge
                self.client.close_position(position.hedge_symbol)
                log.info(f"Closed hedged position {position_id}")
            else:
                log.info(f"[DRY RUN] Would close hedged position {position_id}")
            
            # Update status
            position.status = HedgeStatus.CLOSED
            
            log.info(f"Position {position_id} P&L: Primary=${pnl['primary_pnl']:.2f}, "
                    f"Hedge=${pnl['hedge_pnl']:.2f}, Net=${pnl['net_pnl']:.2f}")
            
            return True, pnl
            
        except Exception as e:
            log.error(f"Failed to close hedged position: {e}")
            position.status = HedgeStatus.ERROR
            return False, {}
    
    def close_hedge_only(self, position_id: str) -> bool:
        """
        Close only the hedge, leaving primary open.
        
        Useful when removing protection after significant profit.
        
        Args:
            position_id: ID of the hedged position
            
        Returns:
            True if successful
        """
        if position_id not in self._positions:
            return False
        
        position = self._positions[position_id]
        
        try:
            if not self.dry_run:
                self.client.close_position(position.hedge_symbol)
            
            position.hedge_size = 0
            position.hedge_ratio = 0
            position.status = HedgeStatus.ACTIVE
            
            log.info(f"Removed hedge from position {position_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to close hedge: {e}")
            return False
    
    def rebalance_hedge(self, position_id: str) -> bool:
        """
        Rebalance hedge based on current correlation.
        
        If correlation has decreased, reduce hedge size.
        If correlation has increased, increase hedge size.
        
        Args:
            position_id: ID of the hedged position
            
        Returns:
            True if rebalanced successfully
        """
        if position_id not in self._positions:
            return False
        
        position = self._positions[position_id]
        
        # Check current correlation
        should_hedge, new_ratio = self.correlation.should_hedge(
            position.primary_symbol,
            position.hedge_symbol
        )
        
        if not should_hedge:
            log.warning(f"Correlation too low, closing hedge for {position_id}")
            return self.close_hedge_only(position_id)
        
        # Check if rebalance needed (>10% difference)
        ratio_diff = abs(new_ratio - position.hedge_ratio)
        if ratio_diff < 0.1:
            log.debug(f"No rebalance needed for {position_id}")
            return True
        
        position.status = HedgeStatus.REBALANCING
        
        # Calculate new hedge size
        primary_value = position.primary_size * position.primary_entry_price
        new_hedge_value = primary_value * new_ratio
        
        try:
            hedge_ticker = self.client.get_ticker(position.hedge_symbol)
            hedge_price = float(hedge_ticker.get('mark_price', 0))
        except Exception as e:
            log.error(f"Failed to get hedge price for rebalance: {e}")
            return False
        
        new_hedge_size = new_hedge_value / hedge_price
        size_diff = new_hedge_size - position.hedge_size
        
        if not self.dry_run:
            try:
                if size_diff > 0:
                    # Need to add more hedge
                    order_side = OrderSide.SELL if position.hedge_side == 'short' else OrderSide.BUY
                    self.client.place_order(
                        symbol=position.hedge_symbol,
                        side=order_side,
                        size=abs(size_diff),
                        order_type=OrderType.MARKET
                    )
                else:
                    # Need to reduce hedge
                    order_side = OrderSide.BUY if position.hedge_side == 'short' else OrderSide.SELL
                    self.client.place_order(
                        symbol=position.hedge_symbol,
                        side=order_side,
                        size=abs(size_diff),
                        order_type=OrderType.MARKET
                    )
            except Exception as e:
                log.error(f"Failed to rebalance hedge: {e}")
                position.status = HedgeStatus.ERROR
                return False
        
        position.hedge_size = new_hedge_size
        position.hedge_ratio = new_ratio
        position.status = HedgeStatus.ACTIVE
        
        log.info(f"Rebalanced hedge for {position_id}: "
                f"ratio {position.hedge_ratio:.0%} -> {new_ratio:.0%}")
        
        return True
    
    def get_position(self, position_id: str) -> Optional[HedgedPosition]:
        """Get a specific hedged position."""
        return self._positions.get(position_id)
    
    def get_all_positions(self) -> List[HedgedPosition]:
        """Get all hedged positions."""
        return list(self._positions.values())
    
    def get_active_positions(self) -> List[HedgedPosition]:
        """Get all active hedged positions."""
        return [p for p in self._positions.values() if p.status == HedgeStatus.ACTIVE]
    
    def get_total_exposure(self) -> float:
        """
        Calculate total net exposure across all hedged positions.
        
        Returns:
            Total USD exposure
        """
        return sum(p.net_exposure for p in self.get_active_positions())
    
    def get_status(self) -> Dict:
        """Get current status of hedge manager."""
        active = self.get_active_positions()
        return {
            'total_positions': len(self._positions),
            'active_positions': len(active),
            'total_exposure': self.get_total_exposure(),
            'dry_run': self.dry_run,
            'positions': [
                {
                    'id': p.id,
                    'primary': f"{p.primary_side} {p.primary_symbol}",
                    'hedge': f"{p.hedge_side} {p.hedge_symbol}",
                    'ratio': p.hedge_ratio,
                    'status': p.status.value
                }
                for p in active
            ]
        }
