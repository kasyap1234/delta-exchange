"""
Delta Exchange API Client.
Handles all authenticated and public API requests to Delta Exchange.
"""

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import requests

from config.settings import settings
from utils.logger import log


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit_order"
    MARKET = "market_order"


class OrderState(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents an order on Delta Exchange."""
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
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'Order':
        return cls(
            id=data.get('id', 0),
            product_id=data.get('product_id', 0),
            product_symbol=data.get('product_symbol', ''),
            side=data.get('side', ''),
            order_type=data.get('order_type', ''),
            size=float(data.get('size', 0)),
            unfilled_size=float(data.get('unfilled_size', 0)),
            limit_price=float(data.get('limit_price')) if data.get('limit_price') else None,
            state=data.get('state', ''),
            created_at=data.get('created_at', '')
        )


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
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'Position':
        return cls(
            product_id=data.get('product_id', 0),
            product_symbol=data.get('product_symbol', ''),
            size=float(data.get('size', 0)),
            entry_price=float(data.get('entry_price', 0)),
            mark_price=float(data.get('mark_price', 0)),
            unrealized_pnl=float(data.get('unrealized_pnl', 0)),
            realized_pnl=float(data.get('realized_pnl', 0))
        )


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_api_response(cls, data) -> 'Candle':
        """
        Parse candle from API response.
        Handles both dict format and array format.
        """
        if isinstance(data, dict):
            # Dict format: {'time': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
            return cls(
                timestamp=int(data.get('time', 0)),
                open=float(data.get('open', 0)),
                high=float(data.get('high', 0)),
                low=float(data.get('low', 0)),
                close=float(data.get('close', 0)),
                volume=float(data.get('volume', 0))
            )
        else:
            # Array format: [timestamp, open, high, low, close, volume]
            return cls(
                timestamp=int(data[0]),
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5]) if len(data) > 5 else 0
            )


class DeltaExchangeClient:
    """
    Client for interacting with Delta Exchange REST API.
    Handles HMAC-SHA256 authentication for private endpoints.
    
    Supports hybrid mode: uses production API for market data while
    trading on testnet (since testnet lacks historical data).
    """
    
    # Production URLs for market data
    PRODUCTION_URLS = {
        'india': 'https://api.india.delta.exchange',
        'global': 'https://api.delta.exchange'
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 base_url: Optional[str] = None, use_hybrid_mode: bool = True):
        """
        Initialize Delta Exchange client.
        
        Args:
            api_key: API key (defaults to settings)
            api_secret: API secret (defaults to settings)
            base_url: Base URL (defaults to settings based on environment)
            use_hybrid_mode: If True and on testnet, fetch market data from production
        """
        self.api_key = api_key or settings.delta.api_key
        self.api_secret = api_secret or settings.delta.api_secret
        self.base_url = base_url or settings.delta.base_url
        
        # Hybrid mode: use production for market data when on testnet
        self.use_hybrid_mode = use_hybrid_mode
        self.is_testnet = 'testnet' in self.base_url.lower()
        
        # Set market data URL (production for candles/tickers)
        if self.is_testnet and use_hybrid_mode:
            region = getattr(settings.delta, 'region', 'india')
            self.market_data_url = self.PRODUCTION_URLS.get(region, self.PRODUCTION_URLS['india'])
            log.info(f"Hybrid mode: Market data from {self.market_data_url}, Trading on testnet")
        else:
            self.market_data_url = self.base_url
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'delta-trading-bot/1.0'
        })
        
        # Cache for product IDs
        self._products_cache: Dict[str, int] = {}
        
        log.info(f"Delta Exchange client initialized for {self.base_url}")
    
    def _generate_signature(self, method: str, timestamp: str, path: str, 
                            query_string: str = '', payload: str = '') -> str:
        """
        Generate HMAC-SHA256 signature for authenticated requests.
        
        The signature is created by hashing:
        method + timestamp + path + query_string + payload
        """
        message = method + timestamp + path + query_string + payload
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_auth_headers(self, method: str, path: str, 
                          query_string: str = '', payload: str = '') -> Dict[str, str]:
        """Generate authentication headers for private endpoints."""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(method, timestamp, path, query_string, payload)
        
        return {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature
        }
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                 data: Optional[Dict] = None, authenticated: bool = True) -> Dict[str, Any]:
        """
        Make an HTTP request to Delta Exchange API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path (e.g., '/v2/orders')
            params: Query parameters
            data: Request body data
            authenticated: Whether to add auth headers
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If API request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Prepare query string
        query_string = ''
        if params:
            query_string = '?' + '&'.join(f"{k}={v}" for k, v in params.items())
        
        # Prepare payload
        payload = ''
        if data:
            payload = json.dumps(data)
        
        # Build headers
        headers = {}
        if authenticated:
            headers = self._get_auth_headers(method, endpoint, query_string, payload)
        
        try:
            log.debug(f"API Request: {method} {url}{query_string}")
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=payload if data else None,
                headers=headers,
                timeout=(5, 30)
            )
            
            response_data = response.json()
            
            if not response.ok:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                log.error(f"API Error: {response.status_code} - {error_msg}")
                raise Exception(f"Delta API Error: {error_msg}")
            
            if not response_data.get('success', False):
                error = response_data.get('error', {})
                raise Exception(f"Delta API Error: {error}")
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed: {e}")
            raise
    
    # ==================== PUBLIC ENDPOINTS ====================
    
    def get_products(self) -> List[Dict]:
        """
        Get list of all available trading products.
        
        Returns:
            List of product dictionaries
        """
        response = self._request('GET', '/v2/products', authenticated=False)
        products = response.get('result', [])
        
        # Update cache
        for product in products:
            self._products_cache[product['symbol']] = product['id']
        
        return products
    
    def get_product_id(self, symbol: str) -> int:
        """
        Get product ID for a symbol.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSDT')
            
        Returns:
            Product ID
        """
        if not self._products_cache:
            self.get_products()
        
        if symbol not in self._products_cache:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        return self._products_cache[symbol]
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker data for a symbol.
        Uses production API in hybrid mode for accurate market data.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSD')
            
        Returns:
            Ticker data with mark_price, spot_price, etc.
        """
        # Use market_data_url for tickers (production in hybrid mode)
        url = f"{self.market_data_url}/v2/tickers/{symbol}"
        try:
            response = self.session.get(url, timeout=(5, 30))
            data = response.json()
            if data.get('success'):
                return data.get('result', {})
            return {}
        except Exception as e:
            log.error(f"Failed to get ticker for {symbol}: {e}")
            return {}
    
    def get_candles(self, symbol: str, resolution: str = '15m', 
                    start: Optional[int] = None, end: Optional[int] = None) -> List[Candle]:
        """
        Get historical OHLC candle data.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSDT')
            resolution: Candle resolution (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start: Start timestamp (Unix seconds)
            end: End timestamp (Unix seconds)
            
        Returns:
            List of Candle objects
        """
        if end is None:
            end = int(time.time())
        if start is None:
            # Default to ~100 candles worth of data
            interval_seconds = self._resolution_to_seconds(resolution)
            start = end - (interval_seconds * settings.trading.candle_count)
        
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'start': start,
            'end': end
        }
        
        # Use market_data_url for candles (production in hybrid mode)
        url = f"{self.market_data_url}/v2/history/candles"
        try:
            response = self.session.get(url, params=params, timeout=(5, 30))
            data = response.json()
            candle_data = data.get('result', [])
            return [Candle.from_api_response(c) for c in candle_data]
        except Exception as e:
            log.error(f"Failed to get candles for {symbol}: {e}")
            return []
    
    def _resolution_to_seconds(self, resolution: str) -> int:
        """Convert resolution string to seconds."""
        multipliers = {
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        unit = resolution[-1]
        value = int(resolution[:-1])
        
        return value * multipliers.get(unit, 60)
    
    # ==================== PRIVATE ENDPOINTS ====================
    
    def get_wallet_balance(self) -> Dict:
        """
        Get wallet/account balance.
        
        Returns:
            Balance data including available balance and margin
        """
        response = self._request('GET', '/v2/wallet/balances')
        return response.get('result', {})
    
    def get_positions(self, symbols: Optional[List[str]] = None) -> List[Position]:
        """
        Get all open positions.
        
        Args:
            symbols: Optional list of symbols to filter (e.g., ['BTC', 'ETH'])
        
        Returns:
            List of Position objects
        """
        all_positions = []
        
        # Use trading pairs from settings if not specified
        if symbols is None:
            # Extract base assets from trading pairs
            symbols = list(set(
                pair.replace('USDT', '').replace('USD', '')
                for pair in settings.trading.trading_pairs
            ))
        
        for symbol in symbols:
            try:
                response = self._request('GET', '/v2/positions', 
                                        params={'underlying_asset_symbol': symbol})
                positions_data = response.get('result', [])
                
                for p in positions_data:
                    if float(p.get('size', 0)) != 0:
                        all_positions.append(Position.from_api_response(p))
            except Exception as e:
                log.debug(f"No positions for {symbol}: {e}")
        
        return all_positions
    
    def get_open_orders(self, product_id: Optional[int] = None) -> List[Order]:
        """
        Get all open orders, optionally filtered by product.
        
        Args:
            product_id: Optional product ID to filter by
            
        Returns:
            List of Order objects
        """
        params = {'state': 'open'}
        if product_id:
            params['product_id'] = product_id
        
        response = self._request('GET', '/v2/orders', params=params)
        orders_data = response.get('result', [])
        
        return [Order.from_api_response(o) for o in orders_data]
    
    def place_order(self, symbol: str, side: OrderSide, size: float, 
                    order_type: OrderType = OrderType.MARKET,
                    limit_price: Optional[float] = None,
                    reduce_only: bool = False,
                    client_order_id: Optional[str] = None) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Product symbol
            side: Buy or Sell
            size: Order size (quantity)
            order_type: Market or Limit
            limit_price: Price for limit orders
            reduce_only: Whether order should only reduce position
            client_order_id: Optional custom order ID
            
        Returns:
            Created Order object
        """
        product_id = self.get_product_id(symbol)
        
        order_data = {
            'product_id': product_id,
            'side': side.value,
            'order_type': order_type.value,
            'size': size
        }
        
        if order_type == OrderType.LIMIT and limit_price:
            order_data['limit_price'] = str(limit_price)
        
        if reduce_only:
            order_data['reduce_only'] = True
        
        if client_order_id:
            order_data['client_order_id'] = client_order_id
        
        log.info(f"TRADE: Placing {side.value} order for {size} {symbol}")
        
        response = self._request('POST', '/v2/orders', data=order_data)
        order = Order.from_api_response(response.get('result', {}))
        
        log.info(f"TRADE: Order placed successfully - ID: {order.id}")
        return order
    
    def cancel_order(self, order_id: int, product_id: int) -> Order:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            product_id: Product ID of the order
            
        Returns:
            Cancelled Order object
        """
        data = {
            'id': order_id,
            'product_id': product_id
        }
        
        log.info(f"TRADE: Cancelling order {order_id}")
        
        response = self._request('DELETE', '/v2/orders', data=data)
        return Order.from_api_response(response.get('result', {}))
    
    def cancel_all_orders(self, product_id: Optional[int] = None) -> List[Order]:
        """
        Cancel all open orders, optionally for a specific product.
        
        Args:
            product_id: Optional product ID to filter by
            
        Returns:
            List of cancelled Order objects
        """
        data = {}
        if product_id:
            data['product_id'] = product_id
        
        log.info(f"TRADE: Cancelling all orders" + (f" for product {product_id}" if product_id else ""))
        
        response = self._request('DELETE', '/v2/orders/all', data=data)
        orders_data = response.get('result', [])
        
        return [Order.from_api_response(o) for o in orders_data]
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close an open position by placing an opposite market order.
        
        Args:
            symbol: Product symbol
            
        Returns:
            Closing Order object or None if no position
        """
        positions = self.get_positions()
        
        for position in positions:
            if position.product_symbol == symbol and position.size != 0:
                # Determine opposite side
                close_side = OrderSide.SELL if position.size > 0 else OrderSide.BUY
                close_size = abs(position.size)
                
                log.info(f"TRADE: Closing position for {symbol} - Size: {close_size}")
                
                return self.place_order(
                    symbol=symbol,
                    side=close_side,
                    size=close_size,
                    order_type=OrderType.MARKET,
                    reduce_only=True
                )
        
        log.warning(f"No open position found for {symbol}")
        return None
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection is successful
        """
        try:
            # Test public endpoint
            products = self.get_products()
            log.info(f"Public API OK - {len(products)} products available")
            
            # Test private endpoint
            balance = self.get_wallet_balance()
            log.info(f"Private API OK - Wallet balance retrieved")
            
            return True
        except Exception as e:
            log.error(f"Connection test failed: {e}")
            return False
