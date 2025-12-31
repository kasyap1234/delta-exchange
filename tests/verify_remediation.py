
import sys
import os
from datetime import datetime, date
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.utils.persistence_manager import PersistenceManager
from src.strategies.strategy_manager import StrategyManager, DailyStats
from src.strategies.base_strategy import StrategyType, SignalDirection, StrategySignal
from src.strategies.funding_arbitrage import FundingArbitrageStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy, MTFPosition

def test_persistence():
    print("Testing PersistenceManager...")
    pm = PersistenceManager("data/test_state.json")
    test_state = {"test": 123, "time": datetime.now().isoformat()}
    
    assert pm.save_state(test_state) == True
    loaded_state = pm.load_state()
    assert loaded_state["test"] == 123
    print("PersistenceManager test passed!")

def test_arb_logic():
    print("Testing FundingArbitrage logic...")
    mock_client = MagicMock()
    mock_client.get_ticker.return_value = {'mark_price': 50000}
    mock_client.get_spot_balance.return_value = 100000
    
    strategy = FundingArbitrageStrategy(mock_client, dry_run=False)
    
    # Test spot symbol mapping (perpetual uses USD, spot uses USDT)
    assert strategy._get_spot_symbol("BTCUSD") == "BTC/USDT"
    
    # Test entry
    arb_pos = strategy.enter_arbitrage("BTCUSD", 0.1, 0.0001)
    assert arb_pos is not None
    assert arb_pos.long_symbol == "BTC/USDT"
    assert arb_pos.short_symbol == "BTCUSD"
    
    # Verify two orders were placed
    assert mock_client.place_order.call_count == 2
    print("FundingArbitrage logic test passed!")

def test_profit_ladder():
    print("Testing MTF Profit Ladder...")
    mock_client = MagicMock()
    strategy = MultiTimeframeStrategy(mock_client, dry_run=True)
    
    # Create a position
    pos = MTFPosition(
        symbol="BTCUSD",
        side='long',
        entry_price=50000,
        size=0.1,
        entry_time=datetime.now(),
        stop_loss=49000,
        take_profit=54000,
        atr_at_entry=500,
        highest_since_entry=50000,
        lowest_since_entry=50000,
        current_trailing_stop=0,
        partial_exits=0
    )
    strategy._mtf_positions["BTCUSD"] = pos
    
    # Test 1R level (Risk = 1000, Target = 51000)
    signal = strategy._check_profit_ladder(pos, 51000)
    assert signal is not None
    assert signal.direction == SignalDirection.CLOSE_PARTIAL
    assert signal.metadata['r_multiple'] == 1.0
    
    # Apply partial exit (normally done by StrategyManager)
    strategy.apply_partial_exit("BTCUSD", signal.position_size)
    assert pos.partial_exits == 1
    assert pos.size < 0.1
    
    # Test 2R level (Target = 52000)
    signal2 = strategy._check_profit_ladder(pos, 52000)
    assert signal2 is not None
    assert signal2.metadata['r_multiple'] == 2.0
    
    print("MTF Profit Ladder test passed!")

if __name__ == "__main__":
    try:
        test_persistence()
        test_arb_logic()
        test_profit_ladder()
        print("\nAll verification tests PASSED!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
