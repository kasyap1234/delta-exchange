"""
Configuration settings for Delta Exchange Trading Bot.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import logger for validation warnings
try:
    from utils.logger import log
except ImportError:
    # Fallback if logger not available during import
    import logging
    log = logging.getLogger(__name__)


@dataclass
class DeltaConfig:
    """Delta Exchange API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("DELTA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("DELTA_API_SECRET", ""))
    environment: str = field(default_factory=lambda: os.getenv("DELTA_ENV", "testnet"))
    region: str = field(default_factory=lambda: os.getenv("DELTA_REGION", "india"))

    @property
    def base_url(self) -> str:
        """Get the appropriate API base URL based on environment and region."""
        if self.environment == "testnet":
            if self.region == "india":
                return "https://cdn-ind.testnet.deltaex.org"
            return "https://testnet-api.delta.exchange"
        else:
            if self.region == "india":
                return "https://api.india.delta.exchange"
            return "https://api.delta.exchange"

    def validate(self) -> bool:
        """Validate that required credentials are set."""
        if not self.api_key or not self.api_secret:
            return False
        if self.api_key == "your_api_key_here":
            return False
        return True


@dataclass
class StrategyAllocationConfig:
    """Capital allocation per strategy tier."""

    # Tier 1: Funding Rate Arbitrage (low risk, passive income)
    funding_arbitrage: float = field(
        default_factory=lambda: float(os.getenv("ALLOC_FUNDING_ARB", "0.40"))
    )  # 40% default

    # Tier 2: Correlated Pair Hedging (medium risk)
    correlated_hedging: float = field(
        default_factory=lambda: float(os.getenv("ALLOC_HEDGING", "0.40"))
    )  # 40% default

    # Tier 3: Multi-Timeframe Trend Following (higher risk)
    multi_timeframe: float = field(
        default_factory=lambda: float(os.getenv("ALLOC_MTF", "0.20"))
    )  # 20% default

    def validate(self) -> bool:
        """Validate allocations sum to ~1.0."""
        total = self.funding_arbitrage + self.correlated_hedging + self.multi_timeframe
        if abs(total - 1.0) > 0.01:
            log.warning(f"Strategy allocations sum to {total:.2f}, expected 1.0")
            return False
        return True


@dataclass
class HedgingConfig:
    """Hedging strategy configuration."""

    # Default hedge ratio (portion of position hedged)
    default_hedge_ratio: float = field(
        default_factory=lambda: float(os.getenv("HEDGE_RATIO", "0.30"))
    )  # 30%

    # Minimum correlation to allow hedging
    min_correlation: float = field(
        default_factory=lambda: float(os.getenv("MIN_CORRELATION", "0.65"))
    )

    # Hedge pair mappings (should match trading pair symbols - USD not USDT)
    hedge_pairs: Dict[str, str] = field(
        default_factory=lambda: {
            "BTCUSD": "ETHUSD",
            "ETHUSD": "BTCUSD",
            "SOLUSD": "ETHUSD",
        }
    )

    # Funding arbitrage settings
    funding_threshold: float = field(
        default_factory=lambda: float(os.getenv("FUNDING_THRESHOLD", "0.0001"))
    )  # 0.01% per 8h (reduced for realistic rates)


@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe analysis configuration."""

    # Higher timeframe for trend direction
    higher_timeframe: str = field(default_factory=lambda: os.getenv("HIGHER_TF", "4h"))

    # Entry timeframe for signals
    entry_timeframe: str = field(default_factory=lambda: os.getenv("ENTRY_TF", "15m"))

    # Trend detection EMA periods
    trend_ema_short: int = 50
    trend_ema_long: int = 200


@dataclass
class EnhancedRiskConfig:
    """Enhanced risk management configuration."""

    # Daily loss limit (halt trading if exceeded)
    daily_loss_limit_pct: float = field(
        default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))
    )  # 3%

    # Maximum drawdown limit (halt trading if exceeded)
    max_drawdown_limit_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.10"))
    )  # 10%

    # ATR-based stops (balanced for better R:R ratios)
    atr_period: int = 14
    atr_stop_multiplier: float = field(
        default_factory=lambda: float(os.getenv("ATR_STOP_MULT", "2.0"))
    )  # 2x ATR (balanced for better risk-reward)
    atr_trailing_multiplier: float = field(
        default_factory=lambda: float(os.getenv("ATR_TRAIL_MULT", "1.5"))
    )  # 1.5x ATR (tighter trailing for protection)

    # Win rate threshold for automatic strategy disabling (e.g., 40% min win rate)
    min_win_rate_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_WIN_RATE_PCT", "0.40"))
    )  # 40% minimum to continue trading

    # Profitability monitoring
    enable_auto_disable: bool = field(
        default_factory=lambda: os.getenv("ENABLE_AUTO_DISABLE", "true").lower() == "true"
    )  # Auto-disable if losing

    # Volatility-adjusted sizing
    reduce_size_high_volatility: bool = field(
        default_factory=lambda: os.getenv("REDUCE_SIZE_HIGH_VOL", "true").lower()
        == "true"
    )
    atr_size_multiplier: float = 1.5  # Reduce size if ATR > 1.5x average

    # Kelly Criterion position sizing
    use_kelly_sizing: bool = field(
        default_factory=lambda: os.getenv("USE_KELLY_SIZING", "true").lower() == "true"
    )
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.5"))
    )  # Use half-Kelly for safety

    # Position sizing limits
    max_risk_per_trade: float = field(
        default_factory=lambda: float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
    )  # 2% risk per trade
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.15"))
    )  # 15% max position size

    # Trailing stop settings
    trailing_enabled: bool = field(
        default_factory=lambda: os.getenv("TRAILING_ENABLED", "true").lower() == "true"
    )

    # Profit ladder (scale out at profit targets)
    profit_ladder_enabled: bool = field(
        default_factory=lambda: os.getenv("PROFIT_LADDER_ENABLED", "false").lower() == "true"
    )


@dataclass
class TradingConfig:
    """Trading strategy configuration."""

    # Trading pairs to monitor (USD format for perpetual contracts)
    trading_pairs: List[str] = field(
        default_factory=lambda: os.getenv(
            "TRADING_PAIRS", "BTCUSD,ETHUSD,SOLUSD"
        ).split(",")
    )

    # Risk management - Wider stops for better profitability
    max_capital_per_trade: float = field(
        default_factory=lambda: float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.15"))
    )  # 15% (was 25%)
    stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "0.04"))
    )  # 4% (was 3%) - Wider to avoid premature exits
    take_profit_pct: float = field(
        default_factory=lambda: float(os.getenv("TAKE_PROFIT_PCT", "0.09"))
    )  # 9% (was 6%) - 2.25:1 R:R maintained
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    )  # Allow room for Arb(3) + Hedge(3) + MTF(3)

    # Candle settings
    candle_interval: str = field(
        default_factory=lambda: os.getenv("CANDLE_INTERVAL", "15m")
    )
    candle_count: int = 300  # Number of candles to fetch for analysis

    # Indicator settings (RSI) - Balanced thresholds for better trade frequency
    rsi_period: int = 14
    rsi_oversold: int = 40  # More balanced for trade frequency
    rsi_overbought: int = 60  # More balanced for trade frequency

    # Indicator settings (MACD)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Indicator settings (Bollinger Bands)
    bb_period: int = 20
    bb_std: float = 2.0

    # Indicator settings (EMA)
    ema_short: int = 9
    ema_long: int = 21

    # Minimum indicators that must agree for a trade (2 = 50% agreement)
    min_signal_agreement: int = 2  # Balanced: 2/4 indicators needed (50% agreement)

    # Order type: limit or market (market fills immediately, limit gets better price)
    use_market_orders: bool = field(
        default_factory=lambda: os.getenv("USE_MARKET_ORDERS", "false").lower() == "true"
    )  # Use market orders for guaranteed fills in volatile markets


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = "logs/trading.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


@dataclass
class Settings:
    """Main settings container."""

    delta: DeltaConfig = field(default_factory=DeltaConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # New enhanced configurations
    strategy_allocation: StrategyAllocationConfig = field(
        default_factory=StrategyAllocationConfig
    )
    hedging: HedgingConfig = field(default_factory=HedgingConfig)
    mtf: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)
    enhanced_risk: EnhancedRiskConfig = field(default_factory=EnhancedRiskConfig)

    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_symbols()
        self._validate_strategy_allocation()

    def _validate_symbols(self):
        """Validate that trading pairs use USD format, not USDT."""
        for pair in self.trading.trading_pairs:
            if 'USDT' in pair and not pair.startswith('USDT'):
                # Allow USDT as base asset (like USDT/USD), but reject BTCUSDT format
                if pair.endswith('USDT') or 'USDT' in pair.upper():
                    raise ValueError(
                        f"Invalid symbol format: {pair}. "
                        f"Delta Exchange perpetual contracts use USD format (e.g., BTCUSD, ETHUSD). "
                        f"USDT symbols (BTCUSDT, ETHUSDT) should not be used."
                    )
        
        # Validate hedge pairs match trading pairs
        for primary, hedge in self.hedging.hedge_pairs.items():
            if primary not in self.trading.trading_pairs:
                log.warning(
                    f"Hedge pair '{primary}' -> '{hedge}' references symbol not in trading_pairs"
                )
            if hedge not in self.trading.trading_pairs:
                log.warning(
                    f"Hedge pair '{primary}' -> '{hedge}' references hedge symbol not in trading_pairs"
                )

    def _validate_strategy_allocation(self):
        """Validate that strategy allocations sum to 1.0."""
        if not self.strategy_allocation.validate():
            raise ValueError(
                "Strategy allocation validation failed"
            )


# Global settings instance
settings = Settings()
