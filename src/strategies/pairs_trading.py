"""
Pairs Trading Strategy Module.
Implements Statistical Arbitrage using Cointegration/Z-Score Mean Reversion.

Strategy Logic:
1. Monitoring: Tracks spread ratio (Price A / Price B)
2. Z-Score Calculation: (Current Spread - Rolling Mean) / Rolling Std Dev
3. Entry:
   - Z-Score > 2.0: Short Spread (Short A, Long B)
   - Z-Score < -2.0: Long Spread (Long A, Short B)
4. Exit:
   - Z-Score reverts to 0 (Mean Reversion)
   - Stop Loss: Correlation breakdown or Extreme Divergence (> 5.0 Sigma)
5. Capital Management:
   - DCA/Grid: Add to position if Z-Score expands to 3.0, 4.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from src.strategies.base_strategy import BaseStrategy, StrategySignal, SignalDirection
from src.client_interface import TradingClient
from src.unified_signal_validator import UnifiedSignalValidator
from config.settings import settings
from utils.logger import log

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading / Statistical Arbitrage Strategy.
    """
    
    def __init__(self, client: TradingClient):
        # Type is property, set internal var BEFORE super init
        self._strategy_type = "pairs_trading"
        super().__init__(client)
        
        self.validator = UnifiedSignalValidator()
        self.lookback_period = 200  # Rolling window for Z-Score
        self.entry_threshold = 2.0  # Sigma
        self.dca_thresholds = [3.0, 4.0] # Add at these sigmas
        self.exit_threshold = 0.5   # Mean reversion target
        self.stop_loss_threshold = 6.0 # Safety stop
        
        # Define Pairs (TODO: Move to settings)
        self.pairs = [('ETHUSD', 'SOLUSD')] 
        self.hedge_ratios: Dict[str, float] = {}   # Calculated Hedge Ratios
        
        # State
        self.spread_history: Dict[str, List[float]] = {
            f"{p[0]}/{p[1]}": [] for p in self.pairs
        }

    @property
    def strategy_type(self) -> str:
        return self._strategy_type

    @property
    def name(self) -> str:
        return "Pairs Trading Strategy"

    def should_trade(self) -> bool:
        return self.is_active
        
    def analyze(self, available_capital: float, current_positions: List[Dict]) -> List[StrategySignal]:
        """Analysis loop for pairs trading."""
        if not self.is_active or available_capital <= 0:
            return []
            
        self.update_positions(current_positions)
        signals = []
        
        # Capital split per pair
        capital_per_pair = available_capital / max(1, len(self.pairs))
        
        for asset_a, asset_b in self.pairs:
            pair_key = f"{asset_a}/{asset_b}"
            
            # Fetch Data
            try:
                # Note: get_candles in backtest doesn't support limit, so we slice manually
                candles_a = self.client.get_candles(asset_a, resolution='1h')
                candles_b = self.client.get_candles(asset_b, resolution='1h')
                
                # Manual slice to lookback period + buffer
                if len(candles_a) > self.lookback_period + 10:
                    candles_a = candles_a[-(self.lookback_period + 10):]
                if len(candles_b) > self.lookback_period + 10:
                    candles_b = candles_b[-(self.lookback_period + 10):]
                
                if len(candles_a) != len(candles_b):
                    # Simple sync attempt (trim to shortest)
                    min_len = min(len(candles_a), len(candles_b))
                    candles_a = candles_a[-min_len:]
                    candles_b = candles_b[-min_len:]
                
                # Extract Prices
                prices_a = np.array([c.close for c in candles_a])
                prices_b = np.array([c.close for c in candles_b])
                
                if len(prices_a) < self.lookback_period:
                    continue
                    
                # 1. Calculate Spread & Z-Score
                spread_series = prices_a / prices_b
                rolling_mean = pd.Series(spread_series).rolling(window=self.lookback_period).mean().values
                rolling_std = pd.Series(spread_series).rolling(window=self.lookback_period).std().values
                
                current_spread = spread_series[-1]
                mean_val = rolling_mean[-1]
                std_val = rolling_std[-1]
                
                if std_val == 0: continue
                
                z_score = (current_spread - mean_val) / std_val
                
                # Log Status
                log.info(f"PAIR {pair_key}: Z-Score={z_score:.2f} (Spread={current_spread:.4f}, Mean={mean_val:.4f})")
                
                # 2. Check Exits (Mean Reversion)
                exit_signals = self._check_exits(pair_key, z_score, asset_a, asset_b)
                if exit_signals:
                    signals.extend(exit_signals)
                    continue # Don't enter if exiting
                
                # 3. Check Entries (Divergence)
                entry_signals = self._check_entries(
                    pair_key, z_score, asset_a, asset_b, prices_a[-1], prices_b[-1], capital_per_pair
                )
                if entry_signals:
                    signals.extend(entry_signals)
                    
            except Exception as e:
                log.error(f"Error analyzing pair {pair_key}: {e}")
                
        return signals

    def _check_entries(self, pair_key: str, z_score: float, 
                       asset_a: str, asset_b: str, 
                       price_a: float, price_b: float, 
                       capital: float) -> List[StrategySignal]:
        """Generate entry signals for the pair."""
        signals = []
        
        # Determine Direction
        # If Z-Score > 2.0 (Spread A/B is High) -> Short Spread: Short A, Long B
        # If Z-Score < -2.0 (Spread A/B is Low) -> Long Spread: Long A, Short B
        
        direction_a = None
        direction_b = None
        is_dca = False
        
        # Current Position check
        pos_a = self.get_position(asset_a)
        # pos_b = self.get_position(asset_b) # Assuming we always have both or neither
        
        current_size_a = abs(float(pos_a.size)) if pos_a else 0
        
        # --- ENTRY LOGIC ---
        
        # Determine DCA Layer based on current size relative to base size
        # Assuming base size defined (e.g., from config or capital split)
        # Here we'll use simple doubling logic
        
        if z_score > self.entry_threshold:
            # Spread is too HIGH -> Short A, Long B
            direction_a = SignalDirection.SHORT
            direction_b = SignalDirection.LONG
            
            # Check for DCA
            if current_size_a > 0:
                # We are already short A. Check if we should add.
                # Only add if Z-Score crossed next threshold (e.g. 3.0, 4.0)
                # AND we haven't already added for this threshold.
                # Simplistic tracking: check if z_score > max_threshold_so_far
                
                # Using thresholds list
                for threshold in self.dca_thresholds:
                    if z_score > threshold:
                        # Check if we should add for this threshold
                        # Heuristic: if current size corresponds to layers below this
                        # Base size approx: sizes usually 1 unit. 
                        # This logic needs precise state tracking, but for now:
                        # Prevent add if Z-score is just hovering
                        is_dca = True
                        break
            
            # Prevent infinite adds: Limit to max 3 entries (Base + 2 DCA)
            if is_dca and current_size_a > (capital / price_a) * 0.4: # Max 40% alloc
                 is_dca = False
                 return []
                 
        elif z_score < -self.entry_threshold:
            # Spread is too LOW -> Long A, Short B
            direction_a = SignalDirection.LONG
            direction_b = SignalDirection.SHORT
            
             # Check for DCA
            if current_size_a > 0:
                 for threshold in self.dca_thresholds:
                    if abs(z_score) > threshold:
                        is_dca = True
                        break
            
            if is_dca and current_size_a > (capital / price_a) * 0.4:
                 is_dca = False
                 return []

        else:
            return [] # No Signal relative to threshold
            
        # Prevent duplicate entries for same threshold (simple cooldown check or position check)
        if current_size_a > 0 and not is_dca:
            return []

        # --- Filter / Validator ---
        # We skip validator for pairs trading DCA often, but can use it for initial entry
        # For now, simplistic implementation
        
        if abs(z_score) > self.stop_loss_threshold:
            log.warning(f"PAIR {pair_key}: Z-Score {z_score:.2f} exceeds STOP threshold! Closing all instead.")
            # Should trigger close, return empty here
            return []
            
        # Size Calculation (Neutral Weighting)
        # Allocate 50% of capital to each leg
        leg_capital = capital / 2
        
        size_a = leg_capital / price_a
        size_b = leg_capital / price_b
        
        # Create Signals
        sig_a = StrategySignal(
            strategy_type=self.strategy_type,
            symbol=asset_a,
            direction=direction_a,
            confidence=1.0, # Statistical Arb assumes high confidence on signal
            entry_price=price_a,
            position_size=size_a,
            stop_loss=None, # Managed by strategy
            take_profit=None,
            reason=f"Pairs Z-Score {z_score:.2f}",
            metadata={'pair': pair_key, 'z_score': z_score, 'role': 'leg_a'}
        )
        
        sig_b = StrategySignal(
            strategy_type=self.strategy_type,
            symbol=asset_b,
            direction=direction_b,
            confidence=1.0,
            entry_price=price_b,
            position_size=size_b,
            stop_loss=None,
            take_profit=None,
            reason=f"Pairs Z-Score {z_score:.2f}",
            metadata={'pair': pair_key, 'z_score': z_score, 'role': 'leg_b'}
        )
        
        signals.append(sig_a)
        signals.append(sig_b)
        
        return signals

    def _check_exits(self, pair_key: str, z_score: float, asset_a: str, asset_b: str) -> List[StrategySignal]:
        """Check for mean reversion exits."""
        signals = []
        pos_a = self.get_position(asset_a)
        pos_b = self.get_position(asset_b)
        
        if not pos_a and not pos_b:
            return []
            
        # Exit Condition: Z-Score close to 0 (Mean Reversion)
        should_close = False
        reason = ""
        
        if abs(z_score) < self.exit_threshold:
            should_close = True
            reason = f"Mean Reversion (Z={z_score:.2f})"
            
        # Stop Loss Condition
        if abs(z_score) > self.stop_loss_threshold:
            should_close = True
            reason = f"Stop Loss (Z={z_score:.2f} > {self.stop_loss_threshold})"
            
        if should_close:
            # Flatten both
            if pos_a:
                signals.append(self._create_exit_signal(asset_a, pos_a, reason))
            if pos_b:
                signals.append(self._create_exit_signal(asset_b, pos_b, reason))
                
        return signals

    def _create_exit_signal(self, symbol: str, position: Dict, reason: str) -> StrategySignal:
        """Helper to create full exit signal."""
        size = float(position.size)
        direction = SignalDirection.SHORT if size > 0 else SignalDirection.LONG # Close = opposite
        return StrategySignal(
            strategy_type=self.strategy_type,
            symbol=symbol,
            direction=direction, # Direction to CLOSE
            confidence=1.0,
            entry_price=0, # Market
            position_size=abs(size),
            reason=reason,
            metadata={'is_exit': True}
        )

