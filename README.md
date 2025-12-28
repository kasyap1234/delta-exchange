# Delta Exchange Automated Trading Bot

A Python-based automated cryptocurrency trading bot for Delta Exchange that uses multiple technical analysis indicators to generate trading signals with robust risk management.

## Features

- **Multiple Technical Indicators**: RSI, MACD, Bollinger Bands, EMA crossover
- **Conservative Trading Strategy**: Only trades when 3+ indicators agree
- **Risk Management**: 
  - 10% max capital per trade
  - 2% stop-loss
  - 4% take-profit (2:1 reward-to-risk ratio)
  - Maximum 3 open positions
- **Multi-Asset Support**: Trade BTC/USDT, ETH/USDT, SOL/USDT simultaneously
- **Background Service**: Runs continuously with configurable intervals
- **Dry Run Mode**: Test strategy without executing real trades
- **Comprehensive Logging**: Trade audit trail and detailed logs

## Prerequisites

- Python 3.10+
- Delta Exchange account (testnet recommended for testing)
- TA-Lib (optional but recommended for performance)

## Installation

### 1. Clone and Setup

```bash
cd /Users/kasyap/Documents/projects/delta-exchange
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
```

## Usage

### Run with Dry Run (Recommended First)

```bash
python main.py --dry-run
```

This logs what trades would be executed without actually placing orders.

### Run on Testnet

```bash
python main.py --testnet
```

### Run Single Analysis Cycle

```bash
python main.py --once --dry-run
```

### Run with Custom Interval

```bash
python main.py --interval 60  # Analyze every 60 seconds
```

### Production Usage

```bash
python main.py
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--testnet` | Use testnet environment |
| `--dry-run` | Log trades without executing |
| `--once` | Run one cycle and exit |
| `--interval N` | Seconds between cycles (default: 300) |

## Configuration

Edit `.env` or `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_PAIRS` | BTCUSDT,ETHUSDT,SOLUSDT | Pairs to trade |
| `MAX_CAPITAL_PER_TRADE` | 0.10 | 10% of capital per trade |
| `STOP_LOSS_PCT` | 0.02 | 2% stop-loss |
| `TAKE_PROFIT_PCT` | 0.04 | 4% take-profit |
| `CANDLE_INTERVAL` | 15m | Analysis timeframe |
| `MAX_OPEN_POSITIONS` | 3 | Maximum concurrent positions |

## Trading Strategy

The bot uses a **conservative multi-indicator confirmation** approach:

1. **RSI (14-period)**: Identifies oversold (<30) and overbought (>70) conditions
2. **MACD (12, 26, 9)**: Detects momentum changes via crossovers
3. **Bollinger Bands (20, 2)**: Identifies price extremes
4. **EMA Crossover (9, 21)**: Confirms trend direction

**Entry Rules:**
- Long: 3+ indicators show bullish signals
- Short: 3+ indicators show bearish signals

**Exit Rules:**
- Stop-loss: -2% from entry
- Take-profit: +4% from entry
- Signal reversal: 3+ indicators flip direction

## Project Structure

```
delta-exchange/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── delta_client.py      # Delta Exchange API wrapper
│   ├── technical_analysis.py # TA indicators
│   ├── risk_manager.py      # Position sizing & risk
│   ├── strategy.py          # Trading strategy
│   └── trader.py            # Trade execution
├── utils/
│   └── logger.py            # Logging setup
├── tests/
│   ├── test_technical_analysis.py
│   └── test_delta_client.py
├── logs/                     # Log files (created automatically)
├── main.py                   # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Logs

Logs are stored in the `logs/` directory:
- `trading.log`: General bot activity
- `trades.log`: Trade execution audit trail

## Safety Notes

> ⚠️ **WARNING**: Automated trading carries significant financial risk.

1. **Always test on testnet first**
2. **Start with small position sizes**
3. **Monitor the bot regularly**
4. **Never invest more than you can afford to lose**
5. **Review all code before running with real funds**

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /v2/products` | List trading pairs |
| `GET /v2/tickers/{symbol}` | Live prices |
| `GET /v2/history/candles` | Historical OHLC |
| `GET /v2/wallet/balances` | Account balance |
| `POST /v2/orders` | Place orders |
| `DELETE /v2/orders` | Cancel orders |
| `GET /v2/positions` | Open positions |

## License

MIT License - Use at your own risk.
