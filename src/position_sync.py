"""
Position Sync Manager Module.
Provides a single source of truth for positions by fetching from the exchange.

This replaces the fragmented internal position tracking that was causing
state drift and unreliable trading decisions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.delta_client import DeltaExchangeClient, Position
from utils.logger import log


@dataclass
class PositionSnapshot:
    """Snapshot of positions from a single sync."""
    positions: Dict[str, Position]
    timestamp: datetime
    sync_successful: bool
    error: Optional[str] = None


class PositionSyncManager:
    """
    Centralized position manager that uses the exchange as the single source of truth.
    
    Key principles:
    1. NEVER cache positions across trading cycles
    2. ALWAYS fetch fresh data from exchange before decisions
    3. Provide consistent API for all strategies
    """
    
    def __init__(self, client: DeltaExchangeClient):
        """
        Initialize position sync manager.
        
        Args:
            client: Delta Exchange API client
        """
        self.client = client
        self._last_snapshot: Optional[PositionSnapshot] = None
    
    def sync_positions(self) -> PositionSnapshot:
        """
        Fetch current positions from exchange.
        
        This is the ONLY way to get position data. All strategies
        should call this at the start of each cycle.
        
        Returns:
            PositionSnapshot with current positions
        """
        try:
            positions = self.client.get_positions()
            
            # Filter to only positions with non-zero size
            active_positions = [p for p in positions if p.size != 0]
            
            snapshot = PositionSnapshot(
                positions={p.product_symbol: p for p in active_positions},
                timestamp=datetime.now(),
                sync_successful=True
            )
            
            self._last_snapshot = snapshot
            
            if active_positions:
                log.info(f"Position sync: {len(active_positions)} active positions")
                for p in active_positions:
                    side = "LONG" if p.size > 0 else "SHORT"
                    log.info(f"  {p.product_symbol}: {side} {abs(p.size):.6f} @ ${p.entry_price:.2f} "
                            f"(PnL: ${p.unrealized_pnl:.2f})")
            else:
                log.debug("Position sync: No active positions")
            
            return snapshot
            
        except Exception as e:
            log.error(f"Position sync failed: {e}")
            snapshot = PositionSnapshot(
                positions={},
                timestamp=datetime.now(),
                sync_successful=False,
                error=str(e)
            )
            self._last_snapshot = snapshot
            return snapshot
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol from last sync.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            
        Returns:
            Position if exists and has non-zero size, None otherwise
        """
        if self._last_snapshot is None:
            log.warning("get_position called before sync - syncing now")
            self.sync_positions()
        
        if self._last_snapshot and self._last_snapshot.sync_successful:
            return self._last_snapshot.positions.get(symbol)
        return None
    
    def get_all_positions(self) -> List[Position]:
        """
        Get all active positions from last sync.
        
        Returns:
            List of active positions
        """
        if self._last_snapshot is None:
            log.warning("get_all_positions called before sync - syncing now")
            self.sync_positions()
        
        if self._last_snapshot and self._last_snapshot.sync_successful:
            return list(self._last_snapshot.positions.values())
        return []
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if we have an active position in this symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists with non-zero size
        """
        pos = self.get_position(symbol)
        return pos is not None and pos.size != 0
    
    def get_position_side(self, symbol: str) -> Optional[str]:
        """
        Get the side (long/short) of a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            'long', 'short', or None if no position
        """
        pos = self.get_position(symbol)
        if pos is None or pos.size == 0:
            return None
        return 'long' if pos.size > 0 else 'short'
    
    def get_total_exposure(self) -> float:
        """
        Calculate total USD exposure across all positions.
        
        Returns:
            Total absolute exposure in USD
        """
        total = 0.0
        for pos in self.get_all_positions():
            exposure = abs(pos.size) * pos.entry_price
            total += exposure
        return total
    
    def get_position_count(self) -> int:
        """
        Get count of active positions.
        
        Returns:
            Number of positions with non-zero size
        """
        return len(self.get_all_positions())
    
    def get_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L across all positions.
        
        Returns:
            Total unrealized P&L in USD
        """
        return sum(p.unrealized_pnl for p in self.get_all_positions())
    
    def was_sync_successful(self) -> bool:
        """Check if last sync was successful."""
        return self._last_snapshot is not None and self._last_snapshot.sync_successful
    
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync."""
        if self._last_snapshot and self._last_snapshot.sync_successful:
            return self._last_snapshot.timestamp
        return None
    
    def get_status(self) -> Dict:
        """Get current status for logging/monitoring."""
        positions = self.get_all_positions()
        return {
            'position_count': len(positions),
            'total_exposure': self.get_total_exposure(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'last_sync': self._last_snapshot.timestamp.isoformat() if self._last_snapshot else None,
            'sync_successful': self.was_sync_successful(),
            'positions': [
                {
                    'symbol': p.product_symbol,
                    'side': 'long' if p.size > 0 else 'short',
                    'size': abs(p.size),
                    'entry_price': p.entry_price,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in positions
            ]
        }
