# STRATEGIES MODULE

## OVERVIEW
Modular trading strategy implementations (funding arbitrage, correlated hedging, multi-timeframe).

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Base interface | src/strategies/base_strategy.py | StrategySignal, StrategyType enums |
| Funding arbitrage | src/strategies/funding_arbitrage.py | Delta-neutral, funding rate arbitrage |
| Correlated hedging | src/strategies/correlated_hedging.py | Pair-based directional hedging |
| Multi-timeframe | src/strategies/multi_timeframe.py | Trend following on 4h/15m |
| Orchestrator | src/strategies/strategy_manager.py | Allocation: 40% arb, 40% hedge, 20% MTF |

## CONVENTIONS
- **All strategies inherit from** BaseStrategy (must implement `analyze()` and `generate_signal()`)
- **Signal confidence:** 0.0-1.0 float
- **Signal direction:** LONG/SHORT/NEUTRAL + CLOSE_LONG/CLOSE_SHORT
- **Arbitrage signals marked** with `metadata['is_arbitrage'] = True`
- **Position sizing via** RiskManager, never manual size calculation

## ANTI-PATTERNS
- **NEVER generate signals without confidence** below 0.5 threshold
- **NEVER skip stop_loss/take_profit** in StrategySignal
- **NEVER directly place orders** from strategies → return signals only
- **NEVER assume current positions** → query DeltaExchangeClient
