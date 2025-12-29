
import pytest
from src.risk_manager import RiskManager
from config.settings import settings

def test_calculate_risk_based_size():
    risk_mgr = RiskManager()
    # Account: $10,000, Risk: 2% ($200)
    # Entry: 50,000, SL: 49,000 (Dist: 1,000)
    # Expected Size: 200 / 1000 = 0.2
    size = risk_mgr.calculate_risk_based_size(
        entry_price=50000,
        stop_loss_price=49000,
        account_balance=10000,
        risk_pct=0.02
    )
    assert size == 0.2

def test_get_kelly_fraction():
    risk_mgr = RiskManager()
    # Win rate: 50% (0.5), W/L: 2:1 (2.0)
    # Kelly: (0.5 * 2 - 0.5) / 2 = 0.5 / 2 = 0.25
    # Safe Kelly (0.5 fraction): 0.125
    # Capped at max_risk_per_trade (0.02)
    fraction = risk_mgr.get_kelly_fraction(win_rate=0.5, win_loss_ratio=2.0)
    assert fraction == 0.02 

def test_adjust_for_volatility():
    risk_mgr = RiskManager()
    base_size = 1.0
    # ATR 100, Avg ATR 100 -> No change
    size = risk_mgr.adjust_for_volatility(base_size, 100, 100)
    assert size == 1.0
    
    # ATR 300, Avg ATR 100 (Ratio 3.0)
    # Threshold 1.5. Reduction: 1.5 / 3.0 = 0.5
    size = risk_mgr.adjust_for_volatility(base_size, 300, 100)
    assert size == 0.5
