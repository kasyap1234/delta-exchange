"""
Tier 2: Correlated Pair Hedging Strategy.
Trades with partial hedge using correlated assets to reduce risk.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from src.strategies.base_strategy import (
    BaseStrategy, StrategyType, StrategySignal, SignalDirection
)
from src.delta_client import DeltaExchangeClient, Position
from src.hedging.correlation import CorrelationCalculator
from src.hedging.hedge_manager import HedgeManager, HedgedPosition
from src.technical_analysis import TechnicalAnalyzer, TechnicalAnalysisResult, Signal
from config.settings import settings
from utils.logger import log
import numpy as np


class CorrelatedHedgingStrategy(BaseStrategy):
    """
    Correlated Pair Hedging Strategy.
    
    This strategy:
    1. Generates signals using technical analysis
    2. Opens primary position based on signal
    3. Opens partial hedge in correlated asset (opposite direction)
    4. Net exposure reduced by 30-50%
    
    Example:
    - Signal: LONG BTC
    - Action: LONG $1000 BTC + SHORT $300 ETH
    - Net exposure: $700 (30% hedged)
    - If BTC falls, ETH likely falls too → hedge profits offset losses
    
    Risk: Medium (30-50% reduced from pure directional)
    Return: 20-40% APY
    """
    
    # Default hedge ratio (30% of primary position)
    DEFAULT_HEDGE_RATIO = 0.3
    
    # Minimum correlation required to hedge
    MIN_CORRELATION = 0.65
    
    def __init__(self, client: DeltaExchangeClient,
                 capital_allocation: float = 0.4,
                 hedge_ratio: float = 0.3,
                 position_sync=None,
                 dry_run: bool = False):
        """
        Initialize correlated hedging strategy.
        
        Args:
            client: Delta Exchange API client
            capital_allocation: Portion of capital to allocate (default 40%)
            hedge_ratio: Portion to hedge (default 30%)
            position_sync: PositionSyncManager for exchange position verification
            dry_run: If True, don't execute real trades
        """
        super().__init__(client, capital_allocation, dry_run)
        
        self.hedge_ratio = hedge_ratio
        self.position_sync = position_sync
        
        # Initialize components
        self.analyzer = TechnicalAnalyzer()
        self.correlation = CorrelationCalculator(client, min_correlation=self.MIN_CORRELATION)
        self.hedge_manager = HedgeManager(
            client, 
            self.correlation,
            position_sync=position_sync,
            default_hedge_ratio=hedge_ratio,
            dry_run=dry_run
        )
        
        # Trading pairs configuration
        self.trading_pairs = getattr(settings.trading, 'trading_pairs', 
                                     ['BTCUSD', 'ETHUSD', 'SOLUSD'])
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.CORRELATED_HEDGING
    
    @property
    def name(self) -> str:
        return "Correlated Pair Hedging"
    
    def should_trade(self) -> bool:
        """
        Check if strategy should be active.
        
        Checks correlation between primary pairs to ensure hedging is viable.
        """
        # Check at least one pair has good correlation
        for symbol in self.trading_pairs:
            hedge_symbol = self.correlation.get_hedge_pair(symbol)
            if hedge_symbol:
                should_hedge, ratio = self.correlation.should_hedge(symbol, hedge_symbol)
                if should_hedge:
                    return True
        return False
    
    def analyze(self, available_capital: float,
                current_positions: List[Position]) -> List[StrategySignal]:
        """
        Analyze markets and generate hedged trading signals.
        
        Args:
            available_capital: Capital available for this strategy
            current_positions: Current open positions
            
        Returns:
            List of StrategySignal with hedge information
        """
        signals = []
        
        if not self.is_active:
            return signals
        
        # Update position tracking
        self.update_positions(current_positions)
        
        # Check for exit signals first
        exit_signals = self._check_exit_signals()
        signals.extend(exit_signals)
        
        # Then check for entry signals
        # Limit to max positions
        active_hedges = len(self.hedge_manager.get_active_positions())
        max_positions = getattr(settings.trading, 'max_open_positions', 3)
        
        if active_hedges < max_positions:
            entry_signals = self._check_entry_signals(available_capital)
            signals.extend(entry_signals)
        
        return signals
    
    def _check_entry_signals(self, available_capital: float) -> List[StrategySignal]:
        """Generate entry signals with hedging."""
        signals = []
        
        # Capital per trade (divide by max positions)
        max_positions = getattr(settings.trading, 'max_open_positions', 3)
        capital_per_trade = available_capital / max_positions
        
        # Track symbols we're planning to trade this cycle to prevent conflicts
        symbols_in_use = set()
        
        # First, mark all symbols that already have positions
        for symbol in self.trading_pairs:
            if self.has_position(symbol):
                symbols_in_use.add(symbol)
        
        # Also check exchange positions via position_sync if available
        if self.position_sync:
            for pos in self.position_sync.get_all_positions():
                symbols_in_use.add(pos.product_symbol)
        
        for symbol in self.trading_pairs:
            # Skip if already have position
            if symbol in symbols_in_use:
                continue
            
            # Analyze market
            ta_result = self._analyze_market(symbol)
            if ta_result is None:
                continue
            
            # Get hedge info
            hedge_symbol = self.correlation.get_hedge_pair(symbol)
            should_hedge, hedge_ratio = self.correlation.should_hedge(symbol, hedge_symbol)
            
            # CRITICAL: Skip if hedge symbol already has a position (would cause netting)
            if should_hedge and hedge_symbol in symbols_in_use:
                log.warning(f"[CORR_HEDGE] {symbol}: Skipping trade - hedge symbol {hedge_symbol} "
                           f"already has a position (would cause netting)")
                continue
            
            # Check for strong signals
            signal = self._create_entry_signal(
                symbol, ta_result, capital_per_trade,
                hedge_symbol if should_hedge else None,
                hedge_ratio if should_hedge else 0
            )
            
            if signal:
                signals.append(signal)
                # Mark both primary and hedge symbols as in use for this cycle
                symbols_in_use.add(symbol)
                if hedge_symbol and should_hedge:
                    symbols_in_use.add(hedge_symbol)
        
        return signals
    
    def _analyze_market(self, symbol: str) -> Optional[TechnicalAnalysisResult]:
        """Fetch candles and run technical analysis."""
        try:
            candles = self.client.get_candles(
                symbol=symbol,
                resolution=getattr(settings.trading, 'candle_interval', '15m')
            )
            
            log.info(f"[CORR_HEDGE] {symbol}: Received {len(candles)} candles")
            
            if len(candles) < 50:
                log.info(f"[CORR_HEDGE] {symbol}: Insufficient candles ({len(candles)} < 50)")
                return None
            
            close = np.array([c.close for c in candles])
            high = np.array([c.high for c in candles])
            low = np.array([c.low for c in candles])
            
            return self.analyzer.analyze(close, high, low, symbol)
            
        except Exception as e:
            log.error(f"Market analysis failed for {symbol}: {e}")
            return None
    
    def _create_entry_signal(self, symbol: str, 
                             ta_result: TechnicalAnalysisResult,
                             capital: float,
                             hedge_symbol: Optional[str],
                             hedge_ratio: float) -> Optional[StrategySignal]:
        """Create an entry signal with hedge if applicable."""
        
        log.info(f"[CORR_HEDGE] _create_entry_signal called for {symbol}")
        
        # DEBUG: Log indicator values
        log.info(f"[HEDGE] {symbol} Analysis:")
        log.info(f"  Combined Signal: {ta_result.combined_signal.value}")
        log.info(f"  Confidence: {ta_result.confidence:.2f}")
        log.info(f"  Signal Strength: {ta_result.signal_strength}/4 indicators aligned")
        for ind in ta_result.indicators:
            log.info(f"    - {ind.name}: {ind.signal.value} (value={ind.value:.2f})")
        
        # ===== CRITICAL: Check higher timeframe trend alignment =====
        # Fetch 4h candles to determine the dominant trend
        try:
            higher_tf_candles = self.client.get_candles(
                symbol=symbol, 
                resolution='4h'
            )
            if higher_tf_candles and len(higher_tf_candles) >= 20:
                # Use standardized trend direction (EMA 50/200 cross)
                # higher_tf_candles are 4h candles
                closes = np.array([c.close for c in higher_tf_candles])
                higher_tf_trend = self.analyzer.get_trend_direction(closes)
                
                log.info(f"  4H Trend Check: {higher_tf_trend.upper()}")
            else:
                higher_tf_trend = "unknown"
                log.warning(f"  4H Trend Check: Insufficient candles ({len(higher_tf_candles) if higher_tf_candles else 0})")
        except Exception as e:
            log.warning(f"Could not fetch higher TF for {symbol}: {e}")
            higher_tf_trend = "unknown"
        
        # Determine direction based on TA
        if ta_result.combined_signal in [Signal.BUY, Signal.STRONG_BUY]:
            direction = SignalDirection.LONG
            hedge_direction = SignalDirection.SHORT
            signal_trend = "bullish"
            log.info(f"  → Entry Signal: LONG (BUY)")
        elif ta_result.combined_signal in [Signal.SELL, Signal.STRONG_SELL]:
            direction = SignalDirection.SHORT
            hedge_direction = SignalDirection.LONG
            signal_trend = "bearish"
            log.info(f"  → Entry Signal: SHORT (SELL)")
        else:
            log.info(f"  → NO SIGNAL: Combined signal is {ta_result.combined_signal.value}")
            return None
        
        # ===== CRITICAL: Reject counter-trend trades =====
        if higher_tf_trend not in ["unknown", "neutral"] and higher_tf_trend != signal_trend:
            log.info(f"  → REJECTED: Signal ({signal_trend}) conflicts with 4h trend ({higher_tf_trend})")
            return None
        
        log.info(f"  → APPROVED: Signal aligned with 4h trend")
        
        # Get current price
        try:
            ticker = self.client.get_ticker(symbol)
            current_price = float(ticker.get('mark_price', 0))
            
            if current_price <= 0:
                return None
            
            # Calculate position size with leverage
            leverage = getattr(settings.trading, 'leverage', 10)
            position_size = (capital * leverage) / current_price
            
            # Calculate hedge size
            hedge_size = None
            hedge_price = None
            if hedge_symbol and hedge_ratio > 0:
                try:
                    hedge_ticker = self.client.get_ticker(hedge_symbol)
                    hedge_price = float(hedge_ticker.get('mark_price', 0))
                    hedge_value = capital * hedge_ratio * leverage  # Apply leverage to hedge too
                    hedge_size = hedge_value / hedge_price
                except Exception as e:
                    log.warning(f"Could not get hedge price: {e}")
            
            # Calculate ATR-based stops
            candles = self.client.get_candles(symbol=symbol)
            high = np.array([c.high for c in candles])
            low = np.array([c.low for c in candles])
            close = np.array([c.close for c in candles])
            
            atr = self.analyzer.calculate_atr(high, low, close)
            
            if direction == SignalDirection.LONG:
                stop_loss = self.analyzer.calculate_atr_stop(
                    current_price, atr, 'long', multiplier=2.0
                )
                take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 R:R
            else:
                stop_loss = self.analyzer.calculate_atr_stop(
                    current_price, atr, 'short', multiplier=2.0
                )
                take_profit = current_price - (stop_loss - current_price) * 2
            
            correlation_value = 0.0
            if hedge_symbol:
                corr_result = self.correlation.calculate_correlation(symbol, hedge_symbol)
                correlation_value = corr_result.correlation
            
            return StrategySignal(
                strategy_type=self.strategy_type,
                symbol=symbol,
                direction=direction,
                confidence=ta_result.confidence,
                entry_price=current_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                hedge_symbol=hedge_symbol,
                hedge_size=hedge_size,
                hedge_direction=hedge_direction,
                reason=f"{ta_result.combined_signal.value} signal "
                       f"({ta_result.signal_strength}/4 indicators) "
                       f"+ {hedge_ratio:.0%} hedge on {hedge_symbol}",
                metadata={
                    'atr': atr,
                    'hedge_ratio': hedge_ratio,
                    'correlation': correlation_value,
                    'indicators': [
                        {'name': ind.name, 'signal': ind.signal.value}
                        for ind in ta_result.indicators
                    ]
                }
            )
            
        except Exception as e:
            log.error(f"Error creating entry signal for {symbol}: {e}")
            return None
    
    def _check_exit_signals(self) -> List[StrategySignal]:
        """Check for exit signals on existing positions."""
        signals = []
        
        for hedged_pos in self.hedge_manager.get_active_positions():
            # Check primary position
            symbol = hedged_pos.primary_symbol
            
            ta_result = self._analyze_market(symbol)
            if ta_result is None:
                continue
            
            should_exit = False
            exit_reason = ""
            
            # Check for signal reversal
            if hedged_pos.primary_side == 'long':
                if ta_result.combined_signal in [Signal.SELL, Signal.STRONG_SELL]:
                    should_exit = True
                    exit_reason = "Signal reversed to bearish"
            else:
                if ta_result.combined_signal in [Signal.BUY, Signal.STRONG_BUY]:
                    should_exit = True
                    exit_reason = "Signal reversed to bullish"
            
            # Check stop-loss / take-profit
            try:
                ticker = self.client.get_ticker(symbol)
                current_price = float(ticker.get('mark_price', 0))
                
                # Simple SL/TP check
                entry = hedged_pos.primary_entry_price
                sl_pct = getattr(settings.trading, 'stop_loss_pct', 0.05)
                tp_pct = getattr(settings.trading, 'take_profit_pct', 0.10)
                
                # Calculate thresholds
                if hedged_pos.primary_side == 'long':
                    sl_price = entry * (1 - sl_pct)
                    tp_price = entry * (1 + tp_pct)
                else:
                    sl_price = entry * (1 + sl_pct)
                    tp_price = entry * (1 - tp_pct)
                
                # Log position status every check
                log.info(f"[EXIT CHECK] {symbol} {hedged_pos.primary_side.upper()}: "
                        f"Entry=${entry:.2f}, Current=${current_price:.2f}, "
                        f"SL=${sl_price:.2f}, TP=${tp_price:.2f}")
                
                if hedged_pos.primary_side == 'long':
                    if current_price <= entry * (1 - sl_pct):
                        should_exit = True
                        exit_reason = "Stop-loss hit"
                    elif current_price >= entry * (1 + tp_pct):
                        should_exit = True
                        exit_reason = "Take-profit hit"
                else:
                    if current_price >= entry * (1 + sl_pct):
                        should_exit = True
                        exit_reason = "Stop-loss hit"
                    elif current_price <= entry * (1 - tp_pct):
                        should_exit = True
                        exit_reason = "Take-profit hit"
                
                if should_exit:
                    log.info(f"[EXIT TRIGGERED] {symbol}: {exit_reason}")
                    direction = SignalDirection.CLOSE_LONG if hedged_pos.primary_side == 'long' \
                               else SignalDirection.CLOSE_SHORT
                    
                    signals.append(StrategySignal(
                        strategy_type=self.strategy_type,
                        symbol=symbol,
                        direction=direction,
                        confidence=0.9,
                        entry_price=current_price,
                        position_size=hedged_pos.primary_size,
                        reason=exit_reason,
                        metadata={
                            'hedge_position_id': hedged_pos.id,
                            'close_hedge': True
                        }
                    ))
                    
            except Exception as e:
                log.error(f"Error checking exit for {symbol}: {e}")
        
        return signals
    
    def execute_entry(self, signal: StrategySignal) -> Optional[HedgedPosition]:
        """
        Execute an entry signal with hedge.
        
        Args:
            signal: StrategySignal with entry details
            
        Returns:
            HedgedPosition if successful
        """
        primary_side = 'long' if signal.direction == SignalDirection.LONG else 'short'
        
        return self.hedge_manager.create_hedged_position(
            primary_symbol=signal.symbol,
            primary_size=signal.position_size,
            primary_side=primary_side,
            primary_price=signal.entry_price,
            hedge_symbol=signal.hedge_symbol,
            hedge_ratio=signal.metadata.get('hedge_ratio', self.hedge_ratio)
        )
    
    def execute_exit(self, signal: StrategySignal) -> bool:
        """
        Execute an exit signal, closing both primary and hedge.
        
        Args:
            signal: StrategySignal with exit details
            
        Returns:
            True if successful
        """
        hedge_id = signal.metadata.get('hedge_position_id')
        
        if hedge_id:
            success, pnl = self.hedge_manager.close_hedged_position(hedge_id)
            
            if success:
                is_win = pnl.get('net_pnl', 0) > 0
                self.record_trade(pnl.get('net_pnl', 0), is_win)
            
            return success
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        
        active_positions = self.hedge_manager.get_active_positions()
        
        base_status.update({
            'hedged_positions': len(active_positions),
            'total_exposure': self.hedge_manager.get_total_exposure(),
            'hedge_manager': self.hedge_manager.get_status(),
            'correlations': {
                f"{s}/{self.correlation.get_hedge_pair(s)}": 
                self.correlation.calculate_correlation(
                    s, self.correlation.get_hedge_pair(s)
                ).correlation
                for s in self.trading_pairs 
                if self.correlation.get_hedge_pair(s)
            }
        })
        
        return base_status
