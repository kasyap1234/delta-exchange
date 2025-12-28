"""
Tests for Delta Exchange API Client.
Tests use mocking to avoid actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.delta_client import (
    DeltaExchangeClient, Order, Position, Candle,
    OrderSide, OrderType, OrderState
)


class TestDeltaExchangeClient:
    """Test cases for DeltaExchangeClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a client instance with test credentials."""
        return DeltaExchangeClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            base_url="https://testnet-api.delta.exchange"
        )
    
    def test_client_initialization(self, client):
        """Test client initializes with correct settings."""
        assert client.api_key == "test_api_key"
        assert client.api_secret == "test_api_secret"
        assert client.base_url == "https://testnet-api.delta.exchange"
    
    def test_signature_generation(self, client):
        """Test HMAC-SHA256 signature generation."""
        signature = client._generate_signature(
            method="GET",
            timestamp="1234567890",
            path="/v2/orders",
            query_string="?product_id=1",
            payload=""
        )
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest
    
    def test_auth_headers_generation(self, client):
        """Test authentication headers are correctly formed."""
        headers = client._get_auth_headers(
            method="GET",
            path="/v2/orders",
            query_string="?product_id=1"
        )
        
        assert 'api-key' in headers
        assert headers['api-key'] == "test_api_key"
        assert 'timestamp' in headers
        assert 'signature' in headers
        assert len(headers['signature']) == 64
    
    @patch('requests.Session.request')
    def test_get_products(self, mock_request, client):
        """Test fetching products from API."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'success': True,
            'result': [
                {'id': 1, 'symbol': 'BTCUSDT', 'description': 'BTC Perpetual'},
                {'id': 2, 'symbol': 'ETHUSDT', 'description': 'ETH Perpetual'},
            ]
        }
        mock_request.return_value = mock_response
        
        products = client.get_products()
        
        assert len(products) == 2
        assert products[0]['symbol'] == 'BTCUSDT'
        assert client._products_cache['BTCUSDT'] == 1
    
    @patch('requests.Session.request')
    def test_get_ticker(self, mock_request, client):
        """Test fetching ticker data."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'success': True,
            'result': {
                'symbol': 'BTCUSDT',
                'mark_price': '45000.50',
                'close': 45000,
                'high': 46000,
                'low': 44000
            }
        }
        mock_request.return_value = mock_response
        
        ticker = client.get_ticker('BTCUSDT')
        
        assert ticker['symbol'] == 'BTCUSDT'
        assert ticker['mark_price'] == '45000.50'
    
    @patch('requests.Session.request')
    def test_get_candles(self, mock_request, client):
        """Test fetching historical candles."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'success': True,
            'result': [
                [1609459200, 45000, 46000, 44000, 45500, 1000],
                [1609459500, 45500, 46500, 45000, 46000, 1200],
            ]
        }
        mock_request.return_value = mock_response
        
        candles = client.get_candles('BTCUSDT', resolution='15m')
        
        assert len(candles) == 2
        assert isinstance(candles[0], Candle)
        assert candles[0].open == 45000
        assert candles[0].close == 45500
    
    @patch('requests.Session.request')
    def test_place_order(self, mock_request, client):
        """Test placing an order."""
        # First mock for get_products (to get product_id)
        products_response = Mock()
        products_response.ok = True
        products_response.json.return_value = {
            'success': True,
            'result': [{'id': 1, 'symbol': 'BTCUSDT'}]
        }
        
        # Second mock for place_order
        order_response = Mock()
        order_response.ok = True
        order_response.json.return_value = {
            'success': True,
            'result': {
                'id': 12345,
                'product_id': 1,
                'product_symbol': 'BTCUSDT',
                'side': 'buy',
                'order_type': 'market_order',
                'size': 0.1,
                'unfilled_size': 0,
                'limit_price': None,
                'state': 'pending',
                'created_at': '2024-01-01T00:00:00Z'
            }
        }
        
        mock_request.side_effect = [products_response, order_response]
        
        order = client.place_order(
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            size=0.1,
            order_type=OrderType.MARKET
        )
        
        assert isinstance(order, Order)
        assert order.id == 12345
        assert order.side == 'buy'
    
    @patch('requests.Session.request')
    def test_get_positions(self, mock_request, client):
        """Test fetching open positions."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'success': True,
            'result': [
                {
                    'product_id': 1,
                    'product_symbol': 'BTCUSDT',
                    'size': 0.5,
                    'entry_price': 45000,
                    'mark_price': 46000,
                    'unrealized_pnl': 500,
                    'realized_pnl': 0
                }
            ]
        }
        mock_request.return_value = mock_response
        
        positions = client.get_positions()
        
        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].product_symbol == 'BTCUSDT'
        assert positions[0].size == 0.5
    
    @patch('requests.Session.request')
    def test_cancel_order(self, mock_request, client):
        """Test cancelling an order."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'success': True,
            'result': {
                'id': 12345,
                'product_id': 1,
                'product_symbol': 'BTCUSDT',
                'side': 'buy',
                'order_type': 'limit_order',
                'size': 0.1,
                'unfilled_size': 0.1,
                'limit_price': 44000,
                'state': 'cancelled',
                'created_at': '2024-01-01T00:00:00Z'
            }
        }
        mock_request.return_value = mock_response
        
        order = client.cancel_order(order_id=12345, product_id=1)
        
        assert order.state == 'cancelled'
    
    @patch('requests.Session.request')
    def test_api_error_handling(self, mock_request, client):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {
            'success': False,
            'error': {
                'code': 'invalid_request',
                'message': 'Invalid order parameters'
            }
        }
        mock_request.return_value = mock_response
        
        with pytest.raises(Exception) as exc_info:
            client._request('POST', '/v2/orders', data={'invalid': 'data'})
        
        assert 'Invalid order parameters' in str(exc_info.value)
    
    def test_resolution_to_seconds(self, client):
        """Test resolution string conversion."""
        assert client._resolution_to_seconds('1m') == 60
        assert client._resolution_to_seconds('5m') == 300
        assert client._resolution_to_seconds('15m') == 900
        assert client._resolution_to_seconds('1h') == 3600
        assert client._resolution_to_seconds('4h') == 14400
        assert client._resolution_to_seconds('1d') == 86400


class TestDataClasses:
    """Test data classes."""
    
    def test_order_from_api_response(self):
        """Test Order parsing from API response."""
        data = {
            'id': 12345,
            'product_id': 1,
            'product_symbol': 'BTCUSDT',
            'side': 'buy',
            'order_type': 'limit_order',
            'size': '0.1',
            'unfilled_size': '0.05',
            'limit_price': '45000',
            'state': 'open',
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        order = Order.from_api_response(data)
        
        assert order.id == 12345
        assert order.size == 0.1
        assert order.unfilled_size == 0.05
        assert order.limit_price == 45000
    
    def test_position_from_api_response(self):
        """Test Position parsing from API response."""
        data = {
            'product_id': 1,
            'product_symbol': 'BTCUSDT',
            'size': '0.5',
            'entry_price': '45000',
            'mark_price': '46000',
            'unrealized_pnl': '500',
            'realized_pnl': '0'
        }
        
        position = Position.from_api_response(data)
        
        assert position.size == 0.5
        assert position.entry_price == 45000
        assert position.unrealized_pnl == 500
    
    def test_candle_from_api_response(self):
        """Test Candle parsing from API response."""
        data = [1609459200, 45000, 46000, 44000, 45500, 1000]
        
        candle = Candle.from_api_response(data)
        
        assert candle.timestamp == 1609459200
        assert candle.open == 45000
        assert candle.high == 46000
        assert candle.low == 44000
        assert candle.close == 45500
        assert candle.volume == 1000


class TestEnums:
    """Test enum values."""
    
    def test_order_side_values(self):
        """Test OrderSide enum."""
        assert OrderSide.BUY.value == 'buy'
        assert OrderSide.SELL.value == 'sell'
    
    def test_order_type_values(self):
        """Test OrderType enum."""
        assert OrderType.LIMIT.value == 'limit_order'
        assert OrderType.MARKET.value == 'market_order'
