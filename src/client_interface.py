"""
Trading Client Protocol Interface.
Defines the interface that both DeltaExchangeClient and BacktestDeltaClient implement.
This enables strategies to work with either live or backtest clients.
"""

from typing import Protocol, List, Optional, Dict, Any, runtime_checkable
from dataclasses import dataclass


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Alias for compatibility
    @property
    def datetime(self) -> int:
        return self.timestamp


@dataclass 
class Position:
    """Represents a trading position."""
    product_id: int
    product_symbol: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Order:
    """Represents an order."""
    id: int
    product_id: int
    product_symbol: str
    side: str
    order_type: str
    size: float
    unfilled_size: float
    limit_price: Optional[float]
    state: str
    created_at: str


@runtime_checkable
class TradingClient(Protocol):
    """
    Protocol defining the interface for trading clients.
    
    Both DeltaExchangeClient (live) and BacktestDeltaClient (backtest)
    must implement these methods to be interchangeable with strategies.
    """
    
    def get_candles(
        self, 
        symbol: str, 
        resolution: str = "15m",
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Candle]:
        """Get historical OHLC candle data."""
        ...
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        ...
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current open positions."""
        ...
    
    def get_wallet_balance(self) -> Dict[str, Any]:
        """Get wallet balance information."""
        ...
    
    def place_order(
        self,
        product_symbol: str,
        side: str,
        size: float,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place a trading order."""
        ...
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order."""
        ...
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close an open position."""
        ...
