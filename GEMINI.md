# PROJECT KNOWLEDGE BASE

**Generated:** 2025-12-31
**Commit:** [not available]
**Branch:** [not available]

## OVERVIEW
Delta Exchange automated trading bot in Python. Multi-strategy system with risk management, hedging, backtesting.

## STRUCTURE
```
.
├── config/           # Settings, environment configs
├── src/
│   ├── backtesting/  # Backtest engine, performance metrics
│   ├── hedging/      # Correlation analysis, hedge manager, funding monitor
│   ├── strategies/   # Funding arbitrage, correlated hedging, multi-timeframe
│   └── utils/        # Balance utils, persistence, symbol utils
├── tests/            # pytest test suite
├── utils/            # Shared logger
├── main.py           # Entry point (v1)
├── main_v2.py        # Enhanced entry point (v2)
├── run_backtest.py   # Backtest runner
└── run_simulation.py # Simulation runner
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry points | main.py, main_v2.py | v2 is newer |
| API client | src/delta_client.py | REST API wrapper |
| Strategy logic | src/strategy.py | Orchestrates strategies |
| Risk management | src/risk_manager.py | Position sizing, exposure |
| Technical indicators | src/technical_analysis.py | RSI, MACD, BB, EMA |
| Trade execution | src/trader.py | Order placement |
| Backtesting | src/backtesting/ | Simulation, metrics |
| Hedging | src/hedging/ | Correlation, hedge manager |
| Strategy modules | src/strategies/ | Base, funding arb, hedge, MTF |

## CONVENTIONS
- **Symbol format:** USD (BTCUSD, ETHUSD), NOT USDT (BTCUSDT)
- **Config hierarchy:** config/settings.py → environment vars → defaults
- **Risk controls:** 2% stop-loss, 4% take-profit (2:1 R:R), 15% max per trade
- **Timeframes:** 15m default, 4h for trend (MTF)
- **Logging:** utils/logger.py with file rotation
- **RSI thresholds:** 30 oversold, 70 overbought (balanced for tradeable signals)
- **Limit orders:** 0.2% buffer from market for better fills
- **Hedge trigger:** Only hedge positions with -2% or greater loss

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER use USDT symbols** for perpetual contracts (BTCUSDT, ETHUSDT) → use USD format
- **NEVER skip stop-loss** → always apply risk management
- **NEVER dry-run in production** → explicitly test on testnet first
- **NEVER exceed max_open_positions** → risk manager enforces hard limit
- **NEVER create duplicate hedges** → check exchange positions before hedging
- **NEVER hedge small losses** → only hedge positions with -2% or worse loss

## UNIQUE STYLES
- **Multi-tier allocation:** 40% funding arb, 40% correlated hedge, 20% MTF
- **Auto-hedging:** HedgeManager auto-protects losing positions
- **Funding arbitrage:** Delta-neutral strategy using funding rates
- **Correlation-based hedging:** Dynamic hedge ratio based on correlation
- **ATR-based stops:** Volatility-adjusted stop-loss with trailing
- **Kelly Criterion:** Adaptive position sizing based on actual trade performance

## COMMANDS
```bash
# Run (production)
python main.py

# Testnet
python main.py --testnet

# Dry run (no real trades)
python main.py --dry-run

# Single cycle
python main.py --once

# Custom interval (seconds)
python main.py --interval 60

# Backtest
python run_backtest.py

# Simulation
python run_simulation.py

# Tests
python -m pytest tests/ -v
```

## NOTES
- **Entry points:** Two versions exist (main.py vs main_v2.py) → v2 is enhanced
- **API authentication:** HMAC-SHA256 signature via delta_client.py
- **WebSocket:** websocket_client.py for real-time data (not main entry point)
- **Persistence:** src/utils/persistence_manager.py for state management
- **Funding rates:** Use 8-hour intervals, monitor via funding_monitor.py
- **Correlation:** hedging/correlation.py calculates rolling correlation for hedge ratios
