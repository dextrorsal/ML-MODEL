"""
Data Fetcher Module

This module handles data fetching from various exchanges and sources for the comparison system.
It includes caching mechanisms to avoid repeated API calls and supports multiple data sources.
"""

import os
import ccxt
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Import config
from comparison_config import ComparisonConfig, MarketType


class DataFetcher:
    """Data fetcher for comparison system"""

    def __init__(self, config: ComparisonConfig):
        """Initialize data fetcher with configuration"""
        self.config = config
        self.cache_dir = Path(config.data_cache_dir)

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange based on configuration"""
        try:
            # Get exchange class
            exchange_id = self.config.exchange.name
            exchange_class = getattr(ccxt, exchange_id)

            # Create exchange instance
            exchange = exchange_class(
                {
                    "apiKey": self.config.exchange.api_key,
                    "secret": self.config.exchange.api_secret,
                    "enableRateLimit": self.config.exchange.rate_limit,
                    "timeout": self.config.exchange.timeout,
                }
            )

            # Use testnet if configured
            if self.config.exchange.testnet and hasattr(exchange, "set_sandbox_mode"):
                exchange.set_sandbox_mode(True)

            print(f"Successfully initialized {exchange_id} connection")
            return exchange

        except Exception as e:
            print(f"Error initializing exchange: {str(e)}")
            print("Using a generic exchange for data fetching")
            return ccxt.binance({"enableRateLimit": True})

    def _get_cache_filename(self) -> Path:
        """Generate cache filename based on configuration"""
        market_type = self.config.market.market_type.value
        symbol = self.config.market.symbol.replace("/", "_")
        timeframe = self.config.market.timeframe
        limit = self.config.market.candle_limit

        filename = f"{self.config.exchange.name}_{market_type}_{symbol}_{timeframe}_{limit}.csv"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is valid (not too old)"""
        if not cache_file.exists():
            return False

        # Check file modification time
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        now = datetime.now()

        # Cache is valid if it's less than 1 hour old
        return (now - mtime) < timedelta(hours=1)

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid"""
        if not self.config.use_cached_data:
            return None

        cache_file = self._get_cache_filename()

        if self._is_cache_valid(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                print(f"Loaded {len(df)} candles from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading from cache: {str(e)}")

        return None

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save data to cache"""
        if not self.config.use_cached_data:
            return

        cache_file = self._get_cache_filename()

        try:
            # Reset index to include timestamp as column
            df_to_save = df.reset_index()
            df_to_save.to_csv(cache_file, index=False)
            print(f"Saved {len(df)} candles to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")

    def fetch_data(self) -> pd.DataFrame:
        """Fetch data based on configuration"""
        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        print(
            f"Fetching {self.config.market.candle_limit} {self.config.market.timeframe} candles for {self.config.market.symbol}..."
        )

        try:
            # Determine market parameters
            symbol = self.config.market.symbol
            timeframe = self.config.market.timeframe
            limit = self.config.market.candle_limit

            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            print(f"Successfully fetched data with shape: {df.shape}")

            # Save to cache
            self._save_to_cache(df)

            return df

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            print("Falling back to synthetic data generation...")

            # Generate synthetic data as fallback
            synthetic_data = self._generate_synthetic_data(limit)
            return synthetic_data

    def _generate_synthetic_data(self, length: int = 1000) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing (fallback if API fails)"""
        print(f"Generating {length} synthetic candles for testing...")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate timestamps
        start_time = datetime.now() - timedelta(days=length // 24)
        timestamps = [start_time + timedelta(hours=i) for i in range(length)]

        # Generate price data with a simulated trend and some noise
        close = np.zeros(length)
        close[0] = 100  # Starting price

        # Generate a random walk with some momentum
        for i in range(1, length):
            # Add momentum (trend-following) and mean-reversion components
            momentum = (
                0.1 * (close[i - 1] - close[max(0, i - 5)]) / close[max(0, i - 5)]
            )
            mean_reversion = -0.05 * (close[i - 1] - close[0]) / close[0]

            # Add some regime changes for realism
            if i % 200 == 0:
                regime_change = np.random.choice([-0.1, 0.1])
            else:
                regime_change = 0

            # Daily random component (with more realistic volatility)
            random_change = np.random.normal(0, 0.015)

            # Combine components with different weights
            change = momentum + mean_reversion + random_change + regime_change
            close[i] = close[i - 1] * (1 + change)

        # Generate OHLC data around the close price
        high = close * (1 + np.abs(np.random.normal(0, 0.01, length)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, length)))
        open_price = low + (high - low) * np.random.random(length)

        # Generate volume with some correlation to price volatility
        volatility = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
        volume = 1000000 + volatility * 5000000 + np.random.normal(0, 500000, length)
        volume = np.maximum(100, volume)  # Ensure positive volume

        # Create DataFrame
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        df.set_index("timestamp", inplace=True)
        print(f"Generated synthetic data with shape: {df.shape}")
        return df

    def fetch_funding_rates(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical funding rates for futures"""
        if self.config.market.market_type != MarketType.FUTURES:
            print("Funding rates only available for futures markets")
            return pd.DataFrame()

        try:
            # TODO: Implement funding rate fetching for different exchanges
            # This is a simple mock implementation
            print(f"Fetching funding rates for {self.config.market.symbol}...")

            # Generate mock funding rates
            now = datetime.now()
            timestamps = [
                now - timedelta(hours=8 * i) for i in range(3 * days)
            ]  # Most exchanges have 8h funding intervals

            # Generate realistic funding rates (typically between -0.1% and 0.1%)
            rates = np.random.normal(0, 0.0005, len(timestamps))

            df = pd.DataFrame({"timestamp": timestamps, "rate": rates})
            df.set_index("timestamp", inplace=True)

            print(f"Generated mock funding rates with shape: {df.shape}")
            return df

        except Exception as e:
            print(f"Error fetching funding rates: {str(e)}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    from comparison_config import default_config

    fetcher = DataFetcher(default_config)
    data = fetcher.fetch_data()
    print(data.head())
