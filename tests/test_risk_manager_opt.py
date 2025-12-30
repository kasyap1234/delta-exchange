
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock loguru and utils.logger
mock_log = MagicMock()
sys.modules['loguru'] = MagicMock()
# Create a dummy module for utils.logger
import types
logger_mod = types.ModuleType('logger')
logger_mod.log = mock_log
if 'utils' not in sys.modules:
    sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils.logger'] = logger_mod
sys.modules['utils'].logger = logger_mod

from src.risk_manager import RiskManager
from config.settings import settings

class TestRiskManagerOptimization(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()
        # Ensure we use test configuration
        self.risk_manager.config.max_capital_per_trade = 0.25
        self.risk_manager.config.leverage = 10
        self.risk_manager.config.stop_loss_pct = 0.05
        self.risk_manager.config.take_profit_pct = 0.10

    def test_leverage_application(self):
        """Verify that position size incorporates 10x leverage."""
        balance = 1000.0
        entry_price = 100.0
        
        # Expected: 1000 * 0.25 (margin) * 10 (leverage) = 2500 position value
        # 2500 / 100 = 25.0 units
        
        sizing = self.risk_manager.calculate_position_size(
            symbol="TEST",
            side="buy",
            entry_price=entry_price,
            available_balance=balance
        )
        
        # Check if the calculation matches expected leveraged size
        # Margin = 1000 * 0.25 = 250
        # Value = 250 * 10 = 2500
        # Size = 2500 / 100 = 25.0
        self.assertEqual(sizing.size, 25.0, f"Expected size 25.0, got {sizing.size}")
        print(f"Leverage Test: Balance={balance}, Leverage=10x, Size={sizing.size} (Correct)")

    def test_atr_stops(self):
        """Verify that ATR-based stops are calculated correctly."""
        balance = 1000.0
        entry_price = 100.0
        atr = 2.0
        
        # SL = entry - (atr * multiplier) = 100 - (2 * 3.0) = 94.0
        # TP = entry + (atr * tp_multiplier) = 100 + (2 * 6.0) = 112.0
        # Multipliers are from enhanced_risk in settings.py template (defaults to 3.0 and 6.0)
        
        sizing = self.risk_manager.calculate_position_size(
            symbol="TEST",
            side="buy",
            entry_price=entry_price,
            available_balance=balance,
            atr=atr
        )
        
        self.assertEqual(sizing.stop_loss_price, 94.0)
        self.assertEqual(sizing.take_profit_price, 112.0)
        print(f"ATR Stop Test: Price=100, ATR=2, SL={sizing.stop_loss_price}, TP={sizing.take_profit_price} (Correct)")

if __name__ == '__main__':
    unittest.main()
