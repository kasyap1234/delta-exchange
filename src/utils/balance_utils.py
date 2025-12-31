"""
Balance utility functions for retrieving account balances.
"""

from typing import Dict, List, Union, Optional
from src.delta_client import DeltaExchangeClient


def get_usd_balance(client: DeltaExchangeClient) -> float:
    """
    Get available USD balance from wallet.
    
    Delta Exchange uses USD (not USDT) for perpetual contract balances.
    
    Args:
        client: Delta Exchange API client
        
    Returns:
        Available USD balance as float, or 0.0 if not found
    """
    try:
        balance_data = client.get_wallet_balance()
        
        if not balance_data:
            return 0.0
        
        # Balance API returns a list of balances
        if isinstance(balance_data, list):
            for wallet in balance_data:
                asset = wallet.get("asset_symbol", "") or wallet.get(
                    "asset", {}
                ).get("symbol", "")
                if asset == "USD":
                    return float(
                        wallet.get("available_balance", 0)
                        or wallet.get("balance", 0)
                        or 0
                    )
            # If no USD found, return 0 (don't sum all balances as fallback)
            return 0.0
        elif isinstance(balance_data, dict):
            # Handle dict format if API returns it
            asset = balance_data.get("asset_symbol", "") or balance_data.get(
                "asset", {}
            ).get("symbol", "")
            if asset == "USD":
                return float(
                    balance_data.get("available_balance", 0)
                    or balance_data.get("balance", 0)
                    or 0
                )
        
        return 0.0
    except Exception as e:
        from utils.logger import log
        log.error(f"Failed to get USD balance: {e}")
        return 0.0
