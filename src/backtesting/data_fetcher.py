"""
Historical Data Fetcher Module.
Fetches and caches historical OHLCV data for backtesting.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from src.delta_client import DeltaExchangeClient, Candle
from utils.logger import log


@dataclass
class OHLCVBar:
    """Single OHLCV bar for backtesting."""

    timestamp: int
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_candle(cls, candle: Candle) -> "OHLCVBar":
        """Create from Delta Exchange Candle object."""
        dt = datetime.fromtimestamp(candle.timestamp)
        return cls(
            timestamp=candle.timestamp,
            datetime=dt.strftime("%Y-%m-%d %H:%M:%S"),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )


@dataclass
class HistoricalData:
    """Container for historical price data."""

    symbol: str
    resolution: str
    start_date: str
    end_date: str
    bars: List[OHLCVBar]

    @property
    def closes(self) -> np.ndarray:
        """Get closing prices as numpy array."""
        return np.array([bar.close for bar in self.bars])

    @property
    def highs(self) -> np.ndarray:
        """Get high prices as numpy array."""
        return np.array([bar.high for bar in self.bars])

    @property
    def lows(self) -> np.ndarray:
        """Get low prices as numpy array."""
        return np.array([bar.low for bar in self.bars])

    @property
    def opens(self) -> np.ndarray:
        """Get open prices as numpy array."""
        return np.array([bar.open for bar in self.bars])

    @property
    def volumes(self) -> np.ndarray:
        """Get volumes as numpy array."""
        return np.array([bar.volume for bar in self.bars])

    @property
    def timestamps(self) -> List[int]:
        """Get timestamps."""
        return [bar.timestamp for bar in self.bars]

    def get_bar_at(self, index: int) -> Optional[OHLCVBar]:
        """Get bar at specific index."""
        if 0 <= index < len(self.bars):
            return self.bars[index]
        return None

    def get_slice(self, start_idx: int, end_idx: int) -> "HistoricalData":
        """Get a slice of the data."""
        sliced_bars = self.bars[start_idx:end_idx]
        return HistoricalData(
            symbol=self.symbol,
            resolution=self.resolution,
            start_date=sliced_bars[0].datetime if sliced_bars else "",
            end_date=sliced_bars[-1].datetime if sliced_bars else "",
            bars=sliced_bars,
        )


class HistoricalDataFetcher:
    """
    Fetches historical OHLCV data from Delta Exchange.

    Features:
    - Fetches data in chunks (API limit handling)
    - Caches data locally for faster subsequent runs
    - Supports multiple timeframes
    - Handles rate limiting
    """

    # Resolution to seconds mapping
    RESOLUTION_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    # Max candles per API request
    MAX_CANDLES_PER_REQUEST = 500

    def __init__(
        self,
        client: Optional[DeltaExchangeClient] = None,
        cache_dir: str = "data/cache",
    ):
        """
        Initialize data fetcher.

        Args:
            client: Delta Exchange API client
            cache_dir: Directory for caching historical data
        """
        self.client = client or DeltaExchangeClient()
        self.cache_dir = cache_dir

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(
        self, symbol: str, resolution: str, start_date: str, end_date: str
    ) -> str:
        """Generate cache file path."""
        filename = f"{symbol}_{resolution}_{start_date}_{end_date}.json"
        return os.path.join(self.cache_dir, filename)

    def _load_from_cache(self, cache_path: str) -> Optional[HistoricalData]:
        """Load data from cache if exists."""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                bars = [OHLCVBar(**bar) for bar in data["bars"]]
                return HistoricalData(
                    symbol=data["symbol"],
                    resolution=data["resolution"],
                    start_date=data["start_date"],
                    end_date=data["end_date"],
                    bars=bars,
                )
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_path: str, data: HistoricalData) -> None:
        """Save data to cache."""
        try:
            cache_data = {
                "symbol": data.symbol,
                "resolution": data.resolution,
                "start_date": data.start_date,
                "end_date": data.end_date,
                "bars": [asdict(bar) for bar in data.bars],
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            log.info(f"Cached {len(data.bars)} bars to {cache_path}")
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")

    def fetch(
        self,
        symbol: str,
        resolution: str = "15m",
        days_back: int = 30,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> HistoricalData:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            resolution: Candle resolution (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            days_back: Number of days to fetch
            end_date: End date (defaults to now)
            use_cache: Whether to use cached data

        Returns:
            HistoricalData object with OHLCV bars
        """
        # Calculate date range
        end_dt = end_date or datetime.now()
        start_dt = end_dt - timedelta(days=days_back)

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        log.info(f"Fetching {symbol} {resolution} data from {start_str} to {end_str}")

        # Check cache
        cache_path = self._get_cache_path(symbol, resolution, start_str, end_str)
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                log.info(f"Loaded {len(cached.bars)} bars from cache")
                return cached

        # Fetch from API
        bars = self._fetch_range(
            symbol=symbol,
            resolution=resolution,
            start_ts=int(start_dt.timestamp()),
            end_ts=int(end_dt.timestamp()),
        )

        if not bars:
            log.warning(f"No data fetched for {symbol}")
            return HistoricalData(
                symbol=symbol,
                resolution=resolution,
                start_date=start_str,
                end_date=end_str,
                bars=[],
            )

        # Create result
        data = HistoricalData(
            symbol=symbol,
            resolution=resolution,
            start_date=start_str,
            end_date=end_str,
            bars=bars,
        )

        # Cache result
        if use_cache:
            self._save_to_cache(cache_path, data)

        log.info(f"Fetched {len(bars)} bars for {symbol}")
        return data

    def _fetch_range(
        self, symbol: str, resolution: str, start_ts: int, end_ts: int
    ) -> List[OHLCVBar]:
        """
        Fetch data in chunks handling API limits.

        Args:
            symbol: Trading pair
            resolution: Candle resolution
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)

        Returns:
            List of OHLCVBar objects
        """
        resolution_seconds = self.RESOLUTION_SECONDS.get(resolution, 900)
        all_bars: List[OHLCVBar] = []

        current_start = start_ts
        request_count = 0

        while current_start < end_ts:
            # Calculate chunk end
            chunk_duration = self.MAX_CANDLES_PER_REQUEST * resolution_seconds
            current_end = min(current_start + chunk_duration, end_ts)

            try:
                # Fetch chunk
                candles = self.client.get_candles(
                    symbol=symbol,
                    resolution=resolution,
                    start=current_start,
                    end=current_end,
                )

                # Convert to OHLCVBar
                for candle in candles:
                    bar = OHLCVBar.from_candle(candle)
                    all_bars.append(bar)

                log.debug(
                    f"Fetched {len(candles)} candles from {current_start} to {current_end}"
                )

                # Move to next chunk
                if candles:
                    # Use last candle timestamp + resolution as next start
                    current_start = candles[-1].timestamp + resolution_seconds
                else:
                    current_start = current_end

                request_count += 1

                # Rate limiting
                if request_count % 5 == 0:
                    time.sleep(0.5)  # Avoid rate limits

            except Exception as e:
                log.error(f"Failed to fetch chunk: {e}")
                current_start = current_end
                time.sleep(1)  # Wait before retry

        # Sort by timestamp and remove duplicates
        all_bars.sort(key=lambda x: x.timestamp)
        unique_bars = []
        seen_timestamps = set()
        for bar in all_bars:
            if bar.timestamp not in seen_timestamps:
                seen_timestamps.add(bar.timestamp)
                unique_bars.append(bar)

        return unique_bars

    def fetch_multiple(
        self,
        symbols: List[str],
        resolution: str = "15m",
        days_back: int = 30,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, HistoricalData]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of trading pairs
            resolution: Candle resolution
            days_back: Number of days to fetch
            end_date: End date

        Returns:
            Dictionary mapping symbol to HistoricalData
        """
        result = {}

        for symbol in symbols:
            try:
                data = self.fetch(
                    symbol=symbol,
                    resolution=resolution,
                    days_back=days_back,
                    end_date=end_date,
                )
                result[symbol] = data
                time.sleep(0.2)  # Avoid rate limits
            except Exception as e:
                log.error(f"Failed to fetch {symbol}: {e}")

        return result

    def fetch_multi_timeframe(
        self, symbol: str, resolutions: List[str] = ["15m", "4h"], days_back: int = 30
    ) -> Dict[str, HistoricalData]:
        """
        Fetch data for multiple timeframes (for multi-timeframe strategy).

        Args:
            symbol: Trading pair
            resolutions: List of resolutions to fetch
            days_back: Number of days to fetch

        Returns:
            Dictionary mapping resolution to HistoricalData
        """
        result = {}

        for resolution in resolutions:
            try:
                data = self.fetch(
                    symbol=symbol, resolution=resolution, days_back=days_back
                )
                result[resolution] = data
                time.sleep(0.2)
            except Exception as e:
                log.error(f"Failed to fetch {symbol} {resolution}: {e}")

        return result
