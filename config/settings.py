"""
Configuration settings for Delta Exchange Trading Bot.
Loads settings from environment variables with sensible defaults.

Enhanced with:
- Signal filtering configuration
- Strict mode for higher quality signals
- Market regime detection settings
- Enhanced risk management options
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
        default_factory=lambda: float(os.getenv("ATR_TRAIL_MULT", "2.0"))
    )  # 2.0x ATR (wider trailing for better profit capture)

    # Win rate threshold for automatic strategy disabling (e.g., 40% min win rate)
    min_win_rate_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_WIN_RATE_PCT", "0.40"))
    )  # 40% minimum to continue trading

    # Profitability monitoring
    enable_auto_disable: bool = field(
        default_factory=lambda: os.getenv("ENABLE_AUTO_DISABLE", "true").lower()
        == "true"
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
        default_factory=lambda: os.getenv("PROFIT_LADDER_ENABLED", "false").lower()
        == "true"
    )

    # Profit lock levels: list of (r_multiple, lock_percentage) tuples
    # At each R-multiple, lock that percentage of profit with stop
    profit_lock_levels: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (1.5, 0.25),  # At 1.5R, lock 25% of profit
            (2.0, 0.50),  # At 2R, lock 50% of profit
            (3.0, 0.75),  # At 3R, lock 75% of profit
        ]
    )

    # Break-even stop configuration
    break_even_trigger_r: float = field(
        default_factory=lambda: float(os.getenv("BREAK_EVEN_TRIGGER_R", "1.0"))
    )  # Move stop to break-even after 1R profit
    break_even_buffer_pct: float = field(
        default_factory=lambda: float(os.getenv("BREAK_EVEN_BUFFER_PCT", "0.001"))
    )  # 0.1% buffer above entry for break-even stop


@dataclass
class SignalFilterConfig:
    """Signal filtering configuration for trade quality."""

    # Enable/disable signal filtering
    enabled: bool = field(
        default_factory=lambda: os.getenv("SIGNAL_FILTER_ENABLED", "true").lower()
        == "true"
    )

    # Use strict mode (stricter RSI thresholds, higher agreement requirements)
    strict_mode: bool = field(
        default_factory=lambda: os.getenv("STRICT_MODE", "false").lower() == "true"
    )

    # Minimum quality score for entry (0-100)
    min_quality_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_QUALITY_SCORE", "60.0"))
    )

    # Minimum signal confidence (0-1)
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.5"))
    )

    # Maximum daily trades per symbol
    max_daily_trades_per_symbol: int = field(
        default_factory=lambda: int(os.getenv("MAX_DAILY_TRADES_PER_SYMBOL", "5"))
    )

    # Consecutive loss limit before cooldown
    consecutive_loss_limit: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))
    )

    # Cooldown after loss streak (minutes)
    loss_cooldown_minutes: int = field(
        default_factory=lambda: int(os.getenv("LOSS_COOLDOWN_MINUTES", "30"))
    )

    # Minimum indicator agreement (out of total indicators)
    min_indicator_agreement: int = field(
        default_factory=lambda: int(os.getenv("MIN_INDICATOR_AGREEMENT", "3"))
    )

    # Maximum volatility percentile to trade (0-100)
    max_volatility_percentile: float = field(
        default_factory=lambda: float(os.getenv("MAX_VOLATILITY_PERCENTILE", "85.0"))
    )

    # Minimum ADX for trend trades
    min_adx_for_trend: float = field(
        default_factory=lambda: float(os.getenv("MIN_ADX_FOR_TREND", "20.0"))
    )


@dataclass
class MarketRegimeConfig:
    """Market regime detection configuration."""

    # Enable market regime filtering
    enabled: bool = field(
        default_factory=lambda: os.getenv("REGIME_FILTER_ENABLED", "true").lower()
        == "true"
    )

    # ADX thresholds for trend strength
    adx_weak_threshold: float = 20.0
    adx_moderate_threshold: float = 25.0
    adx_strong_threshold: float = 40.0
    adx_exhausted_threshold: float = 70.0

    # Choppiness Index thresholds
    choppy_threshold: float = 61.8
    trending_threshold: float = 38.2

    # Volatility percentile thresholds
    low_volatility_percentile: float = 20.0
    high_volatility_percentile: float = 80.0

    # Avoid trading in these regimes
    avoid_regimes: List[str] = field(
        default_factory=lambda: ["choppy", "high_volatility"]
    )

    # Preferred regimes for trend following
    trend_regimes: List[str] = field(
        default_factory=lambda: [
            "uptrend",
            "downtrend",
            "strong_uptrend",
            "strong_downtrend",
        ]
    )

    # Preferred regimes for mean reversion
    mean_reversion_regimes: List[str] = field(
        default_factory=lambda: ["ranging", "low_volatility"]
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
        default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "0.03"))
    )  # 3% - Tighter for better risk management
    take_profit_pct: float = field(
        default_factory=lambda: float(os.getenv("TAKE_PROFIT_PCT", "0.06"))
    )  # 6% - 2:1 R:R ratio
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    )  # Allow room for Arb(3) + Hedge(3) + MTF(3)

    # Candle settings
    candle_interval: str = field(
        default_factory=lambda: os.getenv("CANDLE_INTERVAL", "15m")
    )
    candle_count: int = 300  # Number of candles to fetch for analysis

    # Indicator settings (RSI) - Stricter thresholds for higher quality signals
    rsi_period: int = 14
    rsi_oversold: int = 30  # Standard oversold threshold
    rsi_overbought: int = 70  # Standard overbought threshold

    # Strict RSI thresholds (used in strict_mode)
    rsi_oversold_strict: int = 25
    rsi_overbought_strict: int = 75

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

    # Minimum indicators that must agree for a trade (3 = 75% agreement for 4 indicators)
    min_signal_agreement: int = 3  # Stricter: 3/4 indicators needed (75% agreement)

    # Minimum confidence for trade entry (0.0 - 1.0)
    min_entry_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_ENTRY_CONFIDENCE", "0.55"))
    )

    # Leverage setting (5x is conservative, 10x is aggressive)
    leverage: int = field(
        default_factory=lambda: int(os.getenv("LEVERAGE", "5"))
    )  # 5x leverage (user preference)

    # Order type: limit or market (market fills immediately, limit gets better price)
    use_market_orders: bool = field(
        default_factory=lambda: os.getenv("USE_MARKET_ORDERS", "false").lower()
        == "true"
    )  # Use market orders for guaranteed fills in volatile markets

    # Limit order buffer (how far from market price for limit orders)
    limit_order_buffer_pct: float = field(
        default_factory=lambda: float(os.getenv("LIMIT_ORDER_BUFFER_PCT", "0.002"))
    )  # 0.2% from market price

    # Maximum hold time for positions (hours, 0 = unlimited)
    max_hold_time_hours: int = field(
        default_factory=lambda: int(os.getenv("MAX_HOLD_TIME_HOURS", "168"))
    )  # 7 days default

    # Stagnant position threshold (hours with no significant move)
    stagnant_threshold_hours: int = field(
        default_factory=lambda: int(os.getenv("STAGNANT_THRESHOLD_HOURS", "12"))
    )  # 12 hours

    # Minimum notional value for orders
    min_order_notional: float = field(
        default_factory=lambda: float(os.getenv("MIN_ORDER_NOTIONAL", "10.0"))
    )  # $10 minimum


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

    # Signal filtering and market regime detection
    signal_filter: SignalFilterConfig = field(default_factory=SignalFilterConfig)
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)

    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_symbols()
        self._validate_strategy_allocation()

    def _validate_symbols(self):
        """Validate that trading pairs use USD format, not USDT."""
        for pair in self.trading.trading_pairs:
            if "USDT" in pair and not pair.startswith("USDT"):
                # Allow USDT as base asset (like USDT/USD), but reject BTCUSDT format
                if pair.endswith("USDT") or "USDT" in pair.upper():
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
            raise ValueError("Strategy allocation validation failed")

    def _validate_risk_settings(self):
        """Validate risk management settings are sensible."""
        # Ensure stop loss is not too tight
        if self.trading.stop_loss_pct < 0.01:
            log.warning(
                f"Stop loss {self.trading.stop_loss_pct:.1%} is very tight - may cause premature exits"
            )

        # Ensure R:R ratio is at least 1.5:1
        rr_ratio = self.trading.take_profit_pct / self.trading.stop_loss_pct
        if rr_ratio < 1.5:
            log.warning(
                f"Risk:Reward ratio {rr_ratio:.2f}:1 is below recommended 1.5:1"
            )

        # Validate daily loss limit is reasonable
        if self.enhanced_risk.daily_loss_limit_pct > 0.10:
            log.warning(
                f"Daily loss limit {self.enhanced_risk.daily_loss_limit_pct:.1%} is high - consider lowering"
            )

    def get_strict_rsi_thresholds(self) -> Tuple[int, int]:
        """Get strict RSI thresholds for high-quality signal mode."""
        return (
            getattr(self.trading, "rsi_oversold_strict", 25),
            getattr(self.trading, "rsi_overbought_strict", 75),
        )

    def is_strict_mode(self) -> bool:
        """Check if strict signal filtering mode is enabled."""
        return self.signal_filter.strict_mode


# Global settings instance
settings = Settings()
