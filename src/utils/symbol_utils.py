"""
Symbol utility functions for normalizing and validating trading symbols.
"""

from typing import List


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a trading symbol to USD format for perpetual contracts.
    
    Converts BTCUSDT -> BTCUSD, ETHUSDT -> ETHUSD, etc.
    Leaves BTCUSD unchanged.
    
    Args:
        symbol: Trading symbol (may be in USDT or USD format)
        
    Returns:
        Normalized symbol in USD format
    """
    if symbol.endswith('USDT') and not symbol.startswith('USDT'):
        # Convert BTCUSDT -> BTCUSD
        return symbol.replace('USDT', 'USD')
    return symbol


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate that a symbol uses USD format for perpetuals, not USDT.
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        True if symbol is valid (uses USD or is spot format), False otherwise
    """
    # Allow spot symbols (BTC/USDT format)
    if '/' in symbol:
        return True
    
    # Reject USDT format for perpetuals
    if symbol.endswith('USDT') and not symbol.startswith('USDT'):
        return False
    
    # Accept USD format
    if symbol.endswith('USD'):
        return True
    
    return True  # Allow other formats (base assets, etc.)


def normalize_symbols(symbols: List[str]) -> List[str]:
    """
    Normalize a list of trading symbols.
    
    Args:
        symbols: List of trading symbols
        
    Returns:
        List of normalized symbols in USD format
    """
    return [normalize_symbol(s) for s in symbols]
