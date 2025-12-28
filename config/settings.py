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
    # Tier 1: Delta-neutral funding arbitrage (safest)
    funding_arbitrage: float = field(default_factory=lambda: 
        float(os.getenv("ALLOC_FUNDING_ARB", "0.40")))  # 40%
    
    # Tier 2: Correlated pair hedging (medium risk)
    correlated_hedging: float = field(default_factory=lambda: 
        float(os.getenv("ALLOC_CORR_HEDGE", "0.40")))  # 40%
    
    # Tier 3: Multi-timeframe trend following (higher risk)
    multi_timeframe: float = field(default_factory=lambda: 
        float(os.getenv("ALLOC_MTF", "0.20")))  # 20%
    
    def validate(self) -> bool:
        """Validate allocations sum to 1.0."""
        total = self.funding_arbitrage + self.correlated_hedging + self.multi_timeframe
        return 0.99 <= total <= 1.01  # Allow small float error


@dataclass
class HedgingConfig:
    """Hedging strategy configuration."""
    # Default hedge ratio (portion of position hedged)
    default_hedge_ratio: float = field(default_factory=lambda: 
        float(os.getenv("HEDGE_RATIO", "0.30")))  # 30%
    
    # Minimum correlation to allow hedging
    min_correlation: float = field(default_factory=lambda: 
        float(os.getenv("MIN_CORRELATION", "0.65")))
    
    # Hedge pair mappings
    hedge_pairs: Dict[str, str] = field(default_factory=lambda: {
        'BTCUSDT': 'ETHUSDT',
        'ETHUSDT': 'BTCUSDT',
        'SOLUSDT': 'ETHUSDT',
    })
    
    # Funding arbitrage settings
    funding_threshold: float = field(default_factory=lambda: 
        float(os.getenv("FUNDING_THRESHOLD", "0.0005")))  # 0.05% per 8h


@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe analysis configuration."""
    # Higher timeframe for trend direction
    higher_timeframe: str = field(default_factory=lambda: 
        os.getenv("HIGHER_TF", "4h"))
    
    # Entry timeframe for signals
    entry_timeframe: str = field(default_factory=lambda: 
        os.getenv("ENTRY_TF", "15m"))
    
    # Trend detection EMA periods
    trend_ema_short: int = 50
    trend_ema_long: int = 200


@dataclass
class EnhancedRiskConfig:
    """Enhanced risk management configuration."""
    # Daily loss limit (halt trading if exceeded)
    daily_loss_limit_pct: float = field(default_factory=lambda: 
        float(os.getenv("DAILY_LOSS_LIMIT", "0.03")))  # 3%
    
    # ATR-based stops (widened for leverage trading - reduces whipsaws)
    atr_period: int = 14
    atr_stop_multiplier: float = field(default_factory=lambda: 
        float(os.getenv("ATR_STOP_MULT", "3.0")))  # 3x ATR (was 2x - wider stops)
    atr_trailing_multiplier: float = field(default_factory=lambda: 
        float(os.getenv("ATR_TRAIL_MULT", "2.0")))  # 2x ATR (was 1.5x - more room)
    
    # Trailing stops
    trailing_enabled: bool = field(default_factory=lambda: 
        os.getenv("TRAILING_ENABLED", "true").lower() == "true")
    
    # Profit ladder (take partial profits)
    profit_ladder_enabled: bool = field(default_factory=lambda: 
        os.getenv("PROFIT_LADDER", "true").lower() == "true")
    
    # Profit ladder levels: (R-multiple, exit percentage)
    profit_levels: List[Dict] = field(default_factory=lambda: [
        {'r_multiple': 1.0, 'exit_pct': 0.25},  # 25% at 1R
        {'r_multiple': 2.0, 'exit_pct': 0.25},  # 25% at 2R
        # Remaining 50% trails
    ])


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Trading pairs to monitor
    trading_pairs: List[str] = field(default_factory=lambda: 
        os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT").split(","))
    
    # Risk management - Widened stops for leverage trading
    max_capital_per_trade: float = field(default_factory=lambda: 
        float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.25")))  # 25% (was 10%)
    stop_loss_pct: float = field(default_factory=lambda: 
        float(os.getenv("STOP_LOSS_PCT", "0.05")))  # 5% (was 2% - wider to avoid whipsaws)
    take_profit_pct: float = field(default_factory=lambda: 
        float(os.getenv("TAKE_PROFIT_PCT", "0.10")))  # 10% (was 4% - 2:1 R:R maintained)
    max_open_positions: int = field(default_factory=lambda: 
        int(os.getenv("MAX_OPEN_POSITIONS", "3")))
    
    # Leverage setting (Delta Exchange supports up to 100x)
    leverage: int = field(default_factory=lambda: 
        int(os.getenv("LEVERAGE", "10")))  # 10x leverage (increased for higher returns)
    
    # Candle settings
    candle_interval: str = field(default_factory=lambda: 
        os.getenv("CANDLE_INTERVAL", "15m"))
    candle_count: int = 300  # Number of candles to fetch for analysis
    
    # Indicator settings (RSI) - Slightly relaxed for more signals
    rsi_period: int = 14
    rsi_oversold: int = 35   # Was 30 (more signals in mildly oversold)
    rsi_overbought: int = 65  # Was 70 (more signals in mildly overbought)
    
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
    
    # Minimum indicators that must agree for a trade (3 = 75% agreement)
    min_signal_agreement: int = 2  # Reduced to 2 for more frequent trading on Testnet


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
        default_factory=StrategyAllocationConfig)
    hedging: HedgingConfig = field(default_factory=HedgingConfig)
    mtf: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)
    enhanced_risk: EnhancedRiskConfig = field(default_factory=EnhancedRiskConfig)


# Global settings instance
settings = Settings()

