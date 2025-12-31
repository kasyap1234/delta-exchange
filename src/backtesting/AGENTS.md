# BACKTESTING MODULE

## OVERVIEW
Historical data simulation engine with realistic slippage and performance metrics.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Simulation | src/backtesting/backtest_engine.py | BacktestTrade, signal execution |
| Historical data | src/backtesting/data_fetcher.py | OHLCV, candle fetching |
| Metrics | src/backtesting/performance.py | Sharpe, max drawdown, win rate |

## CONVENTIONS
- **Realistic slippage:** 0.05% default, configurable
- **Commission:** 0.02% per trade
- **Walk-forward analysis:** Train/test split at 70/30
- **Minimum trade size:** $10 USD equivalent

## ANTI-PATTERNS
- **NEVER backtest on** < 1000 candles (insufficient data)
- **NEVER ignore transaction costs** → always include slippage + commission
- **NEVER skip realistic fill simulation** (market impact)
- **NEVER use forward bias** → only historical data available at timestamp
