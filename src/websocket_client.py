"""
Delta Exchange WebSocket Client.
Handles real-time data streaming for positions, orders, and market data.
"""

import asyncio
import json
import time
import ssl
from typing import Callable, Dict, List, Optional, Any
import websockets
from websockets.exceptions import ConnectionClosed

from utils.logger import log
from config.settings import settings

class DeltaWebSocketClient:
    """
    Real-time data streaming from Delta Exchange.
    
    Features:
    - Automatic reconnection
    - Authentication
    - Channel subscriptions (positions, orders, funding, orderbook)
    - Callback routing
    """
    
    ENDPOINTS = {
        'india': 'wss://socket.india.delta.exchange',
        'global': 'wss://socket.delta.exchange'
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 region: str = 'india'):
        # Use settings if keys not provided
        self.api_key = api_key or settings.delta.api_key
        self.api_secret = api_secret or settings.delta.api_secret
        
        # Use region from settings if defaults
        if not api_key:
            region = settings.delta.region
            
        self.ws_url = self.ENDPOINTS.get(region, self.ENDPOINTS['india'])
        self.callbacks: Dict[str, List[Callable]] = {}
        self.ws = None
        self.running = False
        self.loop = None
        
        # Connection state
        self.authenticated = False
        self.subscribed_channels = set()
        
        log.info(f"DeltaWebSocketClient initialized (Region: {region})")
        
    async def connect(self):
        """Establish WebSocket connection with auto-reconnect."""
        self.running = True
        
        while self.running:
            try:
                log.info(f"Connecting to WebSocket: {self.ws_url}")
                
                # Create SSL context (sometimes needed for secure WS)
                ssl_context = ssl.create_default_context()
                
                async with websockets.connect(self.ws_url, ssl=ssl_context) as ws:
                    self.ws = ws
                    log.info("WebSocket connected!")
                    
                    # Authenticate if keys present
                    if self.api_key:
                        await self._authenticate()
                    
                    # Resubscribe to channels
                    await self._resubscribe()
                    
                    # Listen loop
                    await self._listen()
                    
            except (ConnectionClosed, Exception) as e:
                log.error(f"WebSocket connection error: {e}")
                self.authenticated = False
                if self.running:
                    log.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def _authenticate(self):
        """Authenticate with the WebSocket server."""
        if not self.ws or not self.api_key:
            return
            
        import hmac
        import hashlib
        
        timestamp = str(int(time.time()))
        method = "GET"
        path = "/live"
        
        # Signature: HMAC-SHA256(secret, method + timestamp + path + query_string + body)
        # For WS auth, payload matches REST signature format usually
        msg = f"{method}{timestamp}{path}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            msg.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        auth_payload = {
            "type": "auth",
            "payload": {
                "api-key": self.api_key,
                "signature": signature,
                "timestamp": timestamp
            }
        }
        
        await self.ws.send(json.dumps(auth_payload))
        log.info("Sent authentication request")

    async def _listen(self):
        """Main message loop."""
        async for message in self.ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except Exception as e:
                log.error(f"Error processing WS message: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Route messages to registered callbacks."""
        msg_type = data.get('type')
        
        if msg_type == 'auth':
            if data.get('success', False) or data.get('status') == 'success':
                self.authenticated = True
                log.info("WebSocket Authenticated Successfully")
            else:
                log.error(f"WebSocket Authentication Failed: {data}")
            return
            
        # Route to callbacks based on type/channel
        callbacks = self.callbacks.get(msg_type, [])
        for cb in callbacks:
            try:
                 if asyncio.iscoroutinefunction(cb):
                     await cb(data)
                 else:
                     cb(data)
            except Exception as e:
                log.error(f"Error in callback {cb.__name__}: {e}")

    async def subscribe(self, channel: str, symbols: Optional[List[str]] = None, 
                       callback: Optional[Callable] = None):
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name (e.g. 'v2/ticker', 'orders', 'positions')
            symbols: List of symbols (optional depending on channel)
            callback: Function to handle updates
        """
        if callback:
             if channel not in self.callbacks:
                 self.callbacks[channel] = []
             self.callbacks[channel].append(callback)
        
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {
                        "name": channel,
                        "symbols": symbols or []
                    }
                ]
            }
        }
        
        # Store for resubscription
        self.subscribed_channels.add(json.dumps(payload))
        
        if self.ws:
            await self.ws.send(json.dumps(payload))
            log.info(f"Subscribed to {channel} for {symbols}")

    async def _resubscribe(self):
        """Resubscribe to all channels after reconnection."""
        for payload_json in self.subscribed_channels:
            if self.ws:
                 await self.ws.send(payload_json)
                 
    # --- Convenience Methods ---

    async def subscribe_positions(self, callback: Callable):
        """Subscribe to real-time position updates (requires auth)."""
        await self.subscribe('positions', callback=callback)

    async def subscribe_orders(self, callback: Callable):
        """Subscribe to real-time order updates (requires auth)."""
        await self.subscribe('orders', callback=callback)
        
    async def subscribe_funding_rate(self, symbols: List[str], callback: Callable):
        """Subscribe to funding rate updates."""
         # Note: Funding rate might come via 'mark_price' or specific funding channel
         # Per docs, 'mark_price' often carries funding info, or 'v2/ticker'
         # We'll use 'v2/ticker' as it's reliable for mark price + funding
        await self.subscribe('v2/ticker', symbols=symbols, callback=callback)
        
    def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        log.info("Stopping WebSocket client...")
