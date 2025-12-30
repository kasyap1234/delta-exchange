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
from src.hedging.hedge_manager import HedgeManager
from src.hedging.correlation import CorrelationCalculator
from config.settings import settings
from utils.logger import log


@dataclass
class ArbitragePosition:
    """Tracks a funding arbitrage position."""
    long_symbol: str
    short_symbol: str
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
    
    # Minimum funding rate to enter (0.01% per 8h = 0.0001)
    MIN_FUNDING_THRESHOLD = 0.0001
    
    # Minimum correlation for hedging pair
    MIN_CORRELATION = 0.8
    
    # Maximum funding rate to stay in position
    # If rate drops below this, close position
    EXIT_FUNDING_THRESHOLD = 0.00003
    
    def __init__(self, client: DeltaExchangeClient,
                 capital_allocation: float = 0.4,
                 funding_threshold: float = 0.0001,
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
        
        # Hedge manager for correlated hedging
        self.correlation_calc = CorrelationCalculator(client)
        self.hedge_manager = HedgeManager(client, self.correlation_calc, dry_run=dry_run)
        
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
    
    def should_enter_funding_arb(self, funding_rate: float, 
                                 min_rate: float = 0.0001, 
                                 correlation: float = 0.0,
                                 min_correlation: float = 0.8) -> tuple[bool, str]:
        """
        Evaluate funding arb opportunity based on rate and correlation.
        Returns: (should_enter, reason)
        """
        if funding_rate < min_rate:
            return False, f"Funding {funding_rate:.4%} < threshold {min_rate:.4%}"
        
        if correlation < min_correlation:
            return False, f"Correlation {correlation:.2f} < {min_correlation}"
        
        # Annualized return check
        annualized = funding_rate * 3 * 365  # 3x daily, 365 days
        if annualized < 0.10:  # 10% minimum annualized
            return False, f"Annualized {annualized:.1%} too low"
            
        return True, f"Opportunity: {annualized:.1%} annualized"

    def _is_funding_favorable(self, funding_data: FundingRateData) -> bool:
        """
        Check if funding rate is favorable for arbitrage.
        
        Criteria:
        1. Funding rate > MIN_FUNDING_THRESHOLD (0.01% per 8h)
        2. Annualized return > 10%
        3. Hedge correlation > MIN_CORRELATION (0.8)
        """
        symbol = funding_data.symbol
        rate = funding_data.funding_rate
        
        # 3. Use unified evaluation logic
        hedge_pair = self.correlation_calc.get_hedge_pair(symbol)
        correlation = 0.0
        if hedge_pair:
            corr_result = self.correlation_calc.calculate_correlation(symbol, hedge_pair)
            correlation = corr_result.correlation if corr_result.is_reliable else 0.0
            
        should_enter, reason = self.should_enter_funding_arb(
            rate, self.MIN_FUNDING_THRESHOLD, correlation, self.MIN_CORRELATION
        )
        
        if not should_enter:
            log.debug(f"Funding {symbol} rejected: {reason}")
            return False
        
        # 4. Historical consistency check
        history = self.funding_monitor.get_history(symbol)
        if history and not history.is_consistently_positive:
            log.debug(f"Funding {symbol}: Inconsistent historical funding")
            # If inconsistent, require a higher buffer
            if rate < self.MIN_FUNDING_THRESHOLD * 2:
                return False
        
        annual_rate = funding_data.annualized_rate
        log.info(f"Funding {symbol} PASSED: {rate:.4%} ({annual_rate:.1%} AR), Corr={correlation:.2f}")
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
            spot_symbol = self._get_spot_symbol(symbol)
            log.info(f"[DRY RUN] Would enter funding arb: {symbol} size={size}")
            arb_pos = ArbitragePosition(
                long_symbol=spot_symbol,  # Long spot
                short_symbol=symbol,       # Short perpetual
                long_size=size,
                short_size=size,
                entry_funding_rate=funding_rate,
                entry_time=datetime.now()
            )
            self._arb_positions[symbol] = arb_pos
            return arb_pos
        
        try:
            # For true delta-neutral, we need:
            # 1. Long spot (BTC/USDT)
            # 2. Short perpetual (BTCUSD or BTCUSDT)
            
            spot_symbol = self._get_spot_symbol(symbol)
            log.info(f"Entering funding arbitrage: SHORT Perp {size} {symbol} + LONG Spot {size} {spot_symbol}")
            
            if not self.dry_run:
                # 1. Check spot balance
                usdt_needed = size * self.client.get_ticker(spot_symbol).get('mark_price', 0)
                if self.client.get_spot_balance('USDT') < usdt_needed:
                    log.error(f"Insufficient USDT balance for spot hedge. Needed: {usdt_needed}")
                    return None

                from src.delta_client import OrderSide, OrderType
                # 2. Place LONG spot order
                self.client.place_order(
                    symbol=spot_symbol,
                    side=OrderSide.BUY,
                    size=size,
                    order_type=OrderType.MARKET
                )
                
                # 3. Place SHORT perp order
                self.client.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    size=size,
                    order_type=OrderType.MARKET
                )
            
            arb_pos = ArbitragePosition(
                long_symbol=spot_symbol,
                short_symbol=symbol,
                long_size=size,
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

    def _get_spot_symbol(self, perp_symbol: str) -> str:
        """Helper to get spot symbol from perp symbol."""
        # Simple mapping: BTCUSDT -> BTC/USDT, BTCUSD -> BTC/USDT
        base = perp_symbol.replace('USDT', '').replace('USD', '')
        return f"{base}/USDT"
    
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
            if not self.dry_run:
                # 1. Close long spot
                self.client.close_position(arb_pos.long_symbol)
                # 2. Close short perp
                self.client.close_position(arb_pos.short_symbol)
            
            arb_pos.is_active = False
            log.info(f"Exited funding arb: {symbol}, "
                    f"Total funding earned: ${arb_pos.total_funding_earned:.2f}")
            
            # Record the funding as profit
            self.performance.total_funding_earned += arb_pos.total_funding_earned
            
            return True
            
        except Exception as e:
            log.error(f"Failed to exit funding arb: {e}")
            return False

    def get_active_state(self) -> Dict:
        """Return active positions for persistence."""
        return {
            symbol: {
                'long_symbol': p.long_symbol,
                'short_symbol': p.short_symbol,
                'long_size': p.long_size,
                'short_size': p.short_size,
                'entry_funding_rate': p.entry_funding_rate,
                'entry_time': p.entry_time.isoformat(),
                'total_funding_earned': p.total_funding_earned
            }
            for symbol, p in self._arb_positions.items() if p.is_active
        }

    def restore_active_state(self, state: Dict) -> None:
        """Restore active positions from persisted state."""
        for symbol, data in state.items():
            try:
                self._arb_positions[symbol] = ArbitragePosition(
                    long_symbol=data['long_symbol'],
                    short_symbol=data['short_symbol'],
                    long_size=data['long_size'],
                    short_size=data['short_size'],
                    entry_funding_rate=data['entry_funding_rate'],
                    entry_time=datetime.fromisoformat(data['entry_time']),
                    total_funding_earned=data['total_funding_earned']
                )
                log.info(f"Restored funding arb position for {symbol}")
            except Exception as e:
                log.error(f"Failed to restore arb position {symbol}: {e}")
    
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
                    'symbol': p.short_symbol,  # Perpetual symbol
                    'hedge': p.long_symbol,    # Spot hedge symbol
                    'size': p.short_size,
                    'entry_rate': p.entry_funding_rate,
                    'funding_earned': p.total_funding_earned
                }
                for p in active_arbs
            ]
        })
        
        return base_status
