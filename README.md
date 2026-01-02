# Delta Exchange Automated Trading Bot v2.0

A sophisticated Python-based automated cryptocurrency trading bot for Delta Exchange featuring multi-strategy trading, advanced technical indicators, comprehensive signal filtering, and dynamic risk management.

## 🚀 What's New in v2.0

- **Advanced Indicators**: ADX, VWAP, Volume Profile, Market Regime Detection
- **Multi-Layer Signal Filtering**: Quality scoring, momentum confirmation, regime filtering
- **Enhanced Risk Management**: Dynamic stops, break-even triggers, profit protection
- **Market Regime Detection**: Automatically adapts to trending/ranging/volatile conditions
- **Strict Mode**: Higher quality signals with stricter indicator agreement requirements
- **Confidence-Weighted Sizing**: Position sizes adjust based on signal quality

## Features

### 📊 Technical Analysis
- **Classic Indicators**: RSI, MACD, Bollinger Bands, EMA crossover
- **Advanced Indicators**: ADX (trend strength), VWAP (institutional levels), Volume Profile
- **Multi-Timeframe Analysis**: 4H trend direction + 15m entry timing
- **Momentum Confirmation**: ROC, Williams %R, Stochastic, MFI

### 🎯 Multi-Strategy System
| Strategy | Allocation | Risk Level | Expected APY |
|----------|------------|------------|--------------|
| Funding Arbitrage | 40% | Very Low | 10-30% |
| Correlated Hedging | 40% | Medium | 20-40% |
| Multi-Timeframe Trend | 20% | Higher | 30-60% |

### 🛡️ Risk Management
- **Dynamic Stop-Loss**: ATR-based stops that adapt to volatility
- **Break-Even Triggers**: Move stop to entry after 1R profit
- **Trailing Stops**: Multiple methods (ATR, percentage, Chandelier)
- **Profit Protection**: Lock in gains at 1.5R, 2R, 3R levels
- **Position Scaling**: Automatic scale-out at profit targets
- **Drawdown Protection**: Reduce size during drawdowns, halt at 10%
- **Daily Loss Limits**: 3% daily maximum loss

### 🔍 Signal Filtering
- **Quality Score System**: 0-100 score for each signal
- **Market Regime Filter**: Avoid choppy/high-volatility markets
- **Trend Strength Filter**: ADX-based entry confirmation
- **Momentum Alignment**: Multiple momentum indicators must agree
- **Cooldown Periods**: After consecutive losses
- **Correlation Limits**: Prevent excessive correlated exposure

## Prerequisites

- Python 3.10+
- Delta Exchange account (testnet recommended for testing)
- TA-Lib (optional but recommended for performance)

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd delta-exchange
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install TA-Lib (Optional but Recommended)

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

If TA-Lib installation fails, the bot will use fallback Python implementations.

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
DELTA_ENV=testnet
DELTA_REGION=india

# Enhanced Settings (Optional)
STRICT_MODE=true
MIN_QUALITY_SCORE=55
SIGNAL_FILTER_ENABLED=true
REGIME_FILTER_ENABLED=true
```

## Usage

### Quick Start

```bash
# Dry run (recommended first)
python main.py --dry-run

# Testnet trading
python main.py --testnet

# Single cycle analysis
python main.py --once --dry-run

# Production with v2 enhanced strategy
python main_v2.py
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--testnet` | Use testnet environment |
| `--dry-run` | Log trades without executing |
| `--once` | Run one cycle and exit |
| `--interval N` | Seconds between cycles (default: 300) |
| `--paper-trade` | Paper trading with P&L simulation |

### Running Backtests

```bash
python run_backtest.py
```

### Running Simulation

```bash
python run_simulation.py
```

## Configuration

### Core Settings (`config/settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_PAIRS` | BTCUSD,ETHUSD,SOLUSD | Pairs to trade (USD format) |
| `STOP_LOSS_PCT` | 0.03 | 3% stop-loss |
| `TAKE_PROFIT_PCT` | 0.06 | 6% take-profit (2:1 R:R) |
| `MAX_OPEN_POSITIONS` | 10 | Maximum concurrent positions |
| `LEVERAGE` | 5 | Trading leverage |

### Signal Filter Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `STRICT_MODE` | false | Use stricter signal thresholds |
| `MIN_QUALITY_SCORE` | 55 | Minimum quality score (0-100) |
| `MIN_SIGNAL_CONFIDENCE` | 0.5 | Minimum confidence (0-1) |
| `MAX_DAILY_TRADES_PER_SYMBOL` | 5 | Daily trade limit per symbol |
| `CONSECUTIVE_LOSS_LIMIT` | 3 | Losses before cooldown |
| `MAX_VOLATILITY_PERCENTILE` | 85 | Max volatility to trade |

### Risk Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DAILY_LOSS_LIMIT` | 0.03 | 3% daily loss limit |
| `MAX_DRAWDOWN_LIMIT` | 0.10 | 10% max drawdown |
| `ATR_STOP_MULT` | 2.0 | ATR multiplier for stops |
| `USE_KELLY_SIZING` | true | Use Kelly Criterion sizing |
| `BREAK_EVEN_TRIGGER_R` | 1.0 | Move to BE after 1R profit |
| `TRAILING_ENABLED` | true | Enable trailing stops |

### Strategy Allocation

| Setting | Default | Description |
|---------|---------|-------------|
| `ALLOC_FUNDING_ARB` | 0.40 | 40% to funding arbitrage |
| `ALLOC_HEDGING` | 0.40 | 40% to correlated hedging |
| `ALLOC_MTF` | 0.20 | 20% to multi-timeframe |

## Trading Strategy

### Signal Generation Pipeline

```
Market Data → Technical Analysis → Advanced Indicators → Signal Filter → Risk Check → Trade Execution
```

### Entry Rules (Enhanced)

1. **Technical Analysis**: 3/4 indicators must agree (RSI, MACD, BB, EMA)
2. **Market Regime**: Must be favorable (trending or ranging, not choppy)
3. **Trend Strength**: ADX > 20 for trend trades
4. **Momentum**: Must align with trade direction
5. **Quality Score**: Must exceed 55/100
6. **Confidence**: Must exceed 50%

### Exit Rules

- **Stop-Loss**: Dynamic ATR-based or percentage-based
- **Take-Profit**: 2:1 reward-to-risk ratio
- **Break-Even**: After 1R profit
- **Trailing Stop**: 1.5x ATR from highest price
- **Time Exit**: Stagnant positions after 12 hours
- **Signal Reversal**: 3+ indicators flip direction

### Quality Score Components

| Component | Max Points | Description |
|-----------|------------|-------------|
| Signal Confidence | 20 | Based on indicator agreement |
| Trend Strength | 20 | ADX-based strength + alignment |
| Market Regime | 10 | Favorable vs unfavorable |
| Momentum | 15 | Multiple momentum indicators |
| VWAP Position | 10 | Price vs institutional level |
| No Conflicts | 25 | Penalty for conflicting signals |

## Project Structure

```
delta-exchange/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── delta_client.py          # Delta Exchange API wrapper
│   ├── technical_analysis.py    # Classic TA indicators
│   ├── advanced_indicators.py   # ADX, VWAP, Volume Profile, Regime
│   ├── signal_filter.py         # Multi-layer signal filtering
│   ├── enhanced_risk.py         # Dynamic stops, profit protection
│   ├── risk_manager.py          # Position sizing & risk
│   ├── strategy.py              # Trading strategy v1
│   ├── strategy_v2.py           # Enhanced trading strategy
│   ├── trader.py                # Trade execution
│   ├── paper_trader.py          # Paper trading simulator
│   ├── position_sync.py         # Exchange position sync
│   ├── websocket_client.py      # Real-time data
│   ├── strategies/
│   │   ├── base_strategy.py     # Strategy base class
│   │   ├── funding_arbitrage.py # Tier 1: Funding arb
│   │   ├── correlated_hedging.py # Tier 2: Hedge strategy
│   │   ├── multi_timeframe.py   # Tier 3: MTF trend
│   │   └── strategy_manager.py  # Strategy orchestrator
│   ├── hedging/
│   │   ├── correlation.py       # Correlation calculator
│   │   ├── hedge_manager.py     # Hedge management
│   │   └── funding_monitor.py   # Funding rate monitor
│   ├── backtesting/
│   │   ├── backtest_engine.py   # Backtest simulation
│   │   ├── data_fetcher.py      # Historical data
│   │   └── performance.py       # Performance metrics
│   └── utils/
│       ├── balance_utils.py     # Balance helpers
│       ├── persistence_manager.py # State persistence
│       └── symbol_utils.py      # Symbol helpers
├── utils/
│   └── logger.py                # Logging setup
├── tests/
│   ├── test_technical_analysis.py
│   ├── test_delta_client.py
│   ├── test_risk_manager.py
│   └── test_enhanced_strategy.py
├── logs/                        # Log files (created automatically)
├── data/                        # Persisted state
├── main.py                      # Entry point v1
├── main_v2.py                   # Entry point v2 (enhanced)
├── run_backtest.py              # Backtest runner
├── run_simulation.py            # Simulation runner
├── requirements.txt
├── .env.example
├── AGENTS.md                    # Project knowledge base
└── README.md
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_enhanced_strategy.py -v

# With coverage
python -m pytest tests/ -v --cov=src
```

## Logs

Logs are stored in the `logs/` directory:
- `trading.log`: General bot activity with rotation
- `trades.log`: Trade execution audit trail

## Performance Metrics

The bot tracks comprehensive performance metrics:

- **Returns**: Total return, annualized return
- **Risk Metrics**: Max drawdown, volatility, Sharpe ratio, Sortino ratio
- **Trade Stats**: Win rate, profit factor, expectancy
- **Per-Strategy**: Individual strategy performance

## Safety Features

### Automatic Protections

1. **Daily Loss Limit**: Trading halts after 3% daily loss
2. **Max Drawdown**: Trading halts after 10% drawdown
3. **Consecutive Loss Cooldown**: 30 minute cooldown after 3 losses
4. **High Volatility Filter**: Reduces size or avoids in top 15% volatility
5. **Position Limits**: Maximum 10 open positions
6. **Correlated Exposure Limits**: Prevents overexposure to correlated assets

### Anti-Patterns (Avoided)

- ❌ NEVER use USDT symbols for perpetual contracts
- ❌ NEVER skip stop-loss
- ❌ NEVER exceed max position limits
- ❌ NEVER create duplicate hedges
- ❌ NEVER hedge small losses (< 2%)

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /v2/products` | List trading pairs |
| `GET /v2/tickers/{symbol}` | Live prices |
| `GET /v2/history/candles` | Historical OHLC |
| `GET /v2/wallet/balances` | Account balance |
| `POST /v2/orders` | Place orders |
| `POST /v2/orders/bracket` | Place bracket orders (SL/TP) |
| `DELETE /v2/orders` | Cancel orders |
| `GET /v2/positions` | Open positions |

## Troubleshooting

### Common Issues

1. **API Authentication Error**
   - Check API key and secret in `.env`
   - Ensure API key has trading permissions

2. **Insufficient Candle Data**
   - Wait for markets to have enough history
   - Check if symbol is valid (USD format, not USDT)

3. **Signal Filter Rejecting All Trades**
   - Lower `MIN_QUALITY_SCORE` temporarily
   - Check market regime (may be choppy)
   - Review `STRICT_MODE` setting

4. **Position Not Opening**
   - Check available balance
   - Verify max positions not reached
   - Review daily loss limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Safety Notes

> ⚠️ **WARNING**: Automated trading carries significant financial risk.

1. **Always test on testnet first**
2. **Start with small position sizes**
3. **Monitor the bot regularly**
4. **Never invest more than you can afford to lose**
5. **Review all code before running with real funds**
6. **Understand the strategies before deploying**

## License

MIT License - Use at your own risk.

---

## Quick Reference

```bash
# Development
python main.py --dry-run --once          # Quick test
python main.py --testnet                  # Testnet trading
python main_v2.py --paper-trade           # Paper trading with P&L

# Production
python main_v2.py                         # Full multi-strategy bot

# Analysis
python run_backtest.py                    # Run backtests
python run_simulation.py                  # Run simulation

# Tests
python -m pytest tests/ -v                # Run all tests
```
