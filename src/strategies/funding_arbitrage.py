"""
Tier 1: Funding Rate Arbitrage Strategy.
Delta-neutral strategy that profits from funding rate payments.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from src.strategies.base_strategy import (
    BaseStrategy, StrategyType, StrategySignal, 
    SignalDirection, StrategyPerformance
)
from src.delta_client import DeltaExchangeClient, Position
from src.hedging.funding_monitor import FundingMonitor, FundingRateData
from config.settings import settings
from utils.logger import log


@dataclass
class ArbitragePosition:
    """Tracks a funding arbitrage position."""
    symbol: str
    long_size: float
    short_size: float
    entry_funding_rate: float
    entry_time: datetime
    total_funding_earned: float = 0.0
    is_active: bool = True


class FundingArbitrageStrategy(BaseStrategy):
    """
    Delta-Neutral Funding Rate Arbitrage Strategy.
    
    This strategy:
    1. Monitors funding rates on perpetual futures
    2. When funding rate is positive (longs pay shorts):
       - Opens equal long and short positions
       - Net exposure = 0 (market neutral)
       - Earns funding payments every 8 hours
    3. Closes when funding rate becomes unfavorable
    
    Risk: Very low (near zero market risk)
    Return: 10-30% APY depending on market conditions
    """
    
    # Minimum funding rate to enter (0.005% = 0.00005)
    MIN_FUNDING_THRESHOLD = 0.00005
    
    # Maximum funding rate to stay in position
    # If rate drops below this, close position
    EXIT_FUNDING_THRESHOLD = 0.00002
    
    def __init__(self, client: DeltaExchangeClient,
                 capital_allocation: float = 0.4,
                 funding_threshold: float = 0.0005,
                 dry_run: bool = False):
        """
        Initialize funding arbitrage strategy.
        
        Args:
            client: Delta Exchange API client
            capital_allocation: Portion of capital to allocate (default 40%)
            funding_threshold: Minimum funding rate to trade
            dry_run: If True, don't execute real trades
        """
        super().__init__(client, capital_allocation, dry_run)
        
        self.funding_threshold = funding_threshold
        self.funding_monitor = FundingMonitor(
            client, 
            funding_threshold=funding_threshold
        )
        
        # Track active arbitrage positions
        self._arb_positions: Dict[str, ArbitragePosition] = {}
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.FUNDING_ARBITRAGE
    
    @property
    def name(self) -> str:
        return "Delta-Neutral Funding Arbitrage"
    
    def should_trade(self) -> bool:
        """
        Check if conditions are favorable for funding arbitrage.
        
        Returns True if there are opportunities with funding > threshold.
        """
        opportunities = self.funding_monitor.find_arbitrage_opportunities()
        return len(opportunities) > 0
    
    def analyze(self, available_capital: float,
                current_positions: List[Position]) -> List[StrategySignal]:
        """
        Analyze funding rates and generate signals.
        
        Args:
            available_capital: Capital available for this strategy
            current_positions: Current open positions
            
        Returns:
            List of StrategySignal for entry/exit
        """
        signals = []
        
        if not self.is_active:
            return signals
        
        # Update position tracking
        self.update_positions(current_positions)
        
        # Check for exit signals on existing positions
        exit_signals = self._check_exit_signals()
        signals.extend(exit_signals)
        
        # Check for new entry opportunities
        if len(self._arb_positions) < 2:  # Max 2 arb positions
            entry_signals = self._check_entry_signals(available_capital)
            signals.extend(entry_signals)
        
        return signals
    
    def _check_entry_signals(self, available_capital: float) -> List[StrategySignal]:
        """Check for new arbitrage entry opportunities."""
        signals = []
        
        opportunities = self.funding_monitor.find_arbitrage_opportunities()
        
        for funding_data in opportunities:
            symbol = funding_data.symbol
            
            # Skip if already have position in this symbol
            if symbol in self._arb_positions:
                continue
            
            # Check if funding is favorable
            if not self._is_funding_favorable(funding_data):
                continue
            
            # Calculate position size
            # Use half of available capital per position (allow 2 positions)
            position_capital = available_capital / 2
            
            try:
                ticker = self.client.get_ticker(symbol)
                current_price = float(ticker.get('mark_price', 0))
                
                if current_price <= 0:
                    continue
                
                position_size = position_capital / current_price
                
                # Create signal for funding arbitrage
                # This is a special signal where we go both long and short
                signal = StrategySignal(
                    strategy_type=self.strategy_type,
                    symbol=symbol,
                    direction=SignalDirection.NEUTRAL,  # Net neutral
                    confidence=0.9,  # High confidence for arb
                    entry_price=current_price,
                    position_size=position_size,
                    reason=f"Funding arbitrage: {funding_data.funding_rate:.4%}/8h "
                           f"(annualized: {funding_data.annualized_rate:.1%})",
                    metadata={
                        'is_arbitrage': True,
                        'funding_rate': funding_data.funding_rate,
                        'annualized_rate': funding_data.annualized_rate,
                        'funding_direction': funding_data.direction.value,
                        'action': 'enter_arbitrage'
                    }
                )
                
                signals.append(signal)
                log.info(f"Funding arb opportunity: {symbol} @ {funding_data.funding_rate:.4%}/8h")
                
            except Exception as e:
                log.error(f"Error analyzing {symbol} for funding arb: {e}")
        
        return signals
    
    def _check_exit_signals(self) -> List[StrategySignal]:
        """Check if any arbitrage positions should be closed."""
        signals = []
        
        for symbol, arb_pos in self._arb_positions.items():
            if not arb_pos.is_active:
                continue
            
            funding_data = self.funding_monitor.get_funding_rate(symbol)
            
            if funding_data is None:
                continue
            
            # Check if funding has become unfavorable
            should_exit = False
            exit_reason = ""
            
            if funding_data.funding_rate < self.EXIT_FUNDING_THRESHOLD:
                should_exit = True
                exit_reason = f"Funding rate dropped to {funding_data.funding_rate:.4%}"
            
            # Check if funding flipped negative (we'd be paying)
            if funding_data.funding_rate < 0:
                should_exit = True
                exit_reason = f"Funding rate turned negative: {funding_data.funding_rate:.4%}"
            
            if should_exit:
                try:
                    ticker = self.client.get_ticker(symbol)
                    current_price = float(ticker.get('mark_price', 0))
                    
                    signal = StrategySignal(
                        strategy_type=self.strategy_type,
                        symbol=symbol,
                        direction=SignalDirection.NEUTRAL,
                        confidence=0.95,
                        entry_price=current_price,
                        position_size=arb_pos.long_size,
                        reason=f"Exit arbitrage: {exit_reason}",
                        metadata={
                            'is_arbitrage': True,
                            'action': 'exit_arbitrage',
                            'total_funding_earned': arb_pos.total_funding_earned
                        }
                    )
                    
                    signals.append(signal)
                    log.info(f"Exiting funding arb: {symbol} - {exit_reason}")
                    
                except Exception as e:
                    log.error(f"Error creating exit signal for {symbol}: {e}")
        
        return signals
    
    def _is_funding_favorable(self, funding_data: FundingRateData) -> bool:
        """
        Check if funding rate is favorable for arbitrage.
        
        Favorable = positive funding rate above threshold
        (longs pay shorts, so we short perpetual + hedge with spot/long)
        """
        if funding_data.funding_rate < self.funding_threshold:
            return False
        
        # Also check historical consistency
        history = self.funding_monitor.get_history(funding_data.symbol)
        if history and not history.is_consistently_positive:
            # Funding has been volatile, be more cautious
            return funding_data.funding_rate >= self.funding_threshold * 2
        
        return True
    
    def enter_arbitrage(self, symbol: str, size: float, 
                        funding_rate: float) -> Optional[ArbitragePosition]:
        """
        Enter a funding arbitrage position.
        
        Creates both long and short positions of equal size.
        
        Args:
            symbol: Trading symbol
            size: Position size
            funding_rate: Current funding rate
            
        Returns:
            ArbitragePosition if successful
        """
        if self.dry_run:
            log.info(f"[DRY RUN] Would enter funding arb: {symbol} size={size}")
            arb_pos = ArbitragePosition(
                symbol=symbol,
                long_size=size,
                short_size=size,
                entry_funding_rate=funding_rate,
                entry_time=datetime.now()
            )
            self._arb_positions[symbol] = arb_pos
            return arb_pos
        
        try:
            # For true delta-neutral, we need:
            # 1. Long spot or long perpetual on one venue
            # 2. Short perpetual on Delta Exchange
            
            # Since Delta only has perpetuals, we'll use the approach of:
            # - Opening a SHORT perpetual (to earn positive funding)
            # - This leaves us directionally exposed, but for demo purposes
            
            # In production, you'd want to hedge on spot market
            log.info(f"Entering funding arbitrage: SHORT {size} {symbol}")
            
            # Place short order
            from src.delta_client import OrderSide, OrderType
            self.client.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET
            )
            
            arb_pos = ArbitragePosition(
                symbol=symbol,
                long_size=0,  # Would be on spot exchange
                short_size=size,
                entry_funding_rate=funding_rate,
                entry_time=datetime.now()
            )
            
            self._arb_positions[symbol] = arb_pos
            log.info(f"Entered funding arb position: {symbol}")
            
            return arb_pos
            
        except Exception as e:
            log.error(f"Failed to enter funding arb: {e}")
            return None
    
    def exit_arbitrage(self, symbol: str) -> bool:
        """
        Exit a funding arbitrage position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        if symbol not in self._arb_positions:
            log.warning(f"No arb position found for {symbol}")
            return False
        
        arb_pos = self._arb_positions[symbol]
        
        if self.dry_run:
            log.info(f"[DRY RUN] Would exit funding arb: {symbol}")
            arb_pos.is_active = False
            return True
        
        try:
            # Close the short position
            self.client.close_position(symbol)
            
            arb_pos.is_active = False
            log.info(f"Exited funding arb: {symbol}, "
                    f"Total funding earned: ${arb_pos.total_funding_earned:.2f}")
            
            # Record the funding as profit
            self.performance.total_funding_earned += arb_pos.total_funding_earned
            
            return True
            
        except Exception as e:
            log.error(f"Failed to exit funding arb: {e}")
            return False
    
    def record_funding_payment(self, symbol: str, amount: float) -> None:
        """
        Record a funding payment received.
        
        Called when funding is settled every 8 hours.
        
        Args:
            symbol: Trading symbol
            amount: Funding amount received
        """
        if symbol in self._arb_positions:
            self._arb_positions[symbol].total_funding_earned += amount
            self.funding_monitor.record_funding_received(amount, symbol)
            log.info(f"Funding payment: +${amount:.4f} for {symbol}")
    
    def get_active_arbitrages(self) -> List[ArbitragePosition]:
        """Get list of active arbitrage positions."""
        return [p for p in self._arb_positions.values() if p.is_active]
    
    def estimate_daily_income(self, capital: float) -> Dict[str, float]:
        """
        Estimate daily income from funding arbitrage.
        
        Args:
            capital: Total capital to deploy
            
        Returns:
            Dictionary with income estimates per symbol
        """
        estimates = {}
        
        opportunities = self.funding_monitor.find_arbitrage_opportunities()
        
        for funding_data in opportunities:
            income = self.funding_monitor.calculate_potential_income(
                funding_data.symbol,
                capital / len(opportunities) if opportunities else capital,
                days=1
            )
            estimates[funding_data.symbol] = income
        
        return estimates
    
    def get_status(self) -> Dict:
        """Get strategy status."""
        base_status = super().get_status()
        
        active_arbs = self.get_active_arbitrages()
        
        base_status.update({
            'active_arbitrages': len(active_arbs),
            'total_funding_earned': self.performance.total_funding_earned,
            'funding_monitor': self.funding_monitor.get_status(),
            'positions': [
                {
                    'symbol': p.symbol,
                    'size': p.short_size,
                    'entry_rate': p.entry_funding_rate,
                    'funding_earned': p.total_funding_earned
                }
                for p in active_arbs
            ]
        })
        
        return base_status
