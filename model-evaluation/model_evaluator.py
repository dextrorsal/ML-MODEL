"""
All-in-One Model Evaluator

This consolidated file provides all functionality for evaluating and comparing
machine learning models for trading. It combines the functionality of multiple
separate files into a single convenient module.

Features:
- Configuration management
- Data fetching and preparation
- Model implementation comparison
- Backtesting and performance metrics
- Visualization and reporting
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import traceback
import matplotlib

# Set non-interactive backend if not showing plots
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import time
import argparse
from pathlib import Path
from tabulate import tabulate
import ccxt

# Try importing torch, but make it optional
try:
    import torch
except ImportError:
    print("PyTorch not found, using numpy for tensor operations")

    # Create a placeholder for torch.is_tensor
    class torch:
        @staticmethod
        def is_tensor(x):
            return False


# Handle BaseModel if it exists in the codebase
try:
    from base_model import BaseModel
except ImportError:
    # Create a simple BaseModel class if not available
    class BaseModel:
        def __init__(self):
            pass

# Configuration Classes
# ====================


class MarketType(Enum):
    """Market type enumeration"""

    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class ExchangeConfig:
    """Exchange configuration"""

    name: str = "binance"  # Exchange name (binance, bitget, bybit, etc.)
    api_key: str = ""  # API key (optional)
    api_secret: str = ""  # API secret (optional)
    testnet: bool = True  # Use testnet when available

    # Load API keys from environment variables if available
    def __post_init__(self):
        if not self.api_key and os.environ.get(f"{self.name.upper()}_API_KEY"):
            self.api_key = os.environ.get(f"{self.name.upper()}_API_KEY")
        if not self.api_secret and os.environ.get(f"{self.name.upper()}_API_SECRET"):
            self.api_secret = os.environ.get(f"{self.name.upper()}_API_SECRET")


@dataclass
class DataConfig:
    """Data configuration"""

    symbol: str = "BTC/USDT"  # Trading pair
    timeframe: str = "1h"  # Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    start_date: str = "2023-01-01"  # Start date for historical data
    end_date: str = "2023-08-01"  # End date for historical data
    data_dir: str = "data"  # Directory for data storage
    use_cached_data: bool = True  # Use cached data if available


@dataclass
class ModelConfig:
    """Model configuration"""

    # Parameters for all models
    use_gpu: bool = torch.cuda.is_available()  # Use GPU if available
    batch_size: int = 64  # Batch size for training

    # Lorentzian model parameters
    lorentzian_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lookback_bars": 30,
            "prediction_bars": 5,
            "k_neighbors": 15,
            "distinction_weight": 2.0,
            "use_regime_filter": True,
            "use_volatility_filter": True,
        }
    )

    # Logistic regression parameters
    logistic_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lookback": 3,
            "learning_rate": 0.0009,
            "iterations": 1000,
            "use_amp": False,
            "threshold": 0.5,
        }
    )

    # Chandelier exit parameters
    chandelier_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "atr_period": 22,
            "atr_multiplier": 3.0,
            "use_close": False,
        }
    )


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_balance: float = 10000.0  # Starting balance
    risk_per_trade: float = 2.0  # Risk percentage per trade
    fee_rate: float = 0.001  # Trading fee rate
    use_stop_loss: bool = True  # Whether to use stop loss
    stop_loss_pct: float = 2.5  # Stop loss percentage
    use_take_profit: bool = True  # Whether to use take profit
    take_profit_pct: float = 5.0  # Take profit percentage
    trailing_stop: bool = True  # Whether to use trailing stop
    max_positions: int = 1  # Maximum number of concurrent positions
    allow_shorts: bool = True  # Allow short positions
    compound_returns: bool = True  # Use compounding


@dataclass
class OutputConfig:
    """Output configuration"""

    output_dir: str = "output"  # Directory for results
    show_plots: bool = True  # Show plots during execution
    save_plots: bool = True  # Save plots to file
    verbose: bool = True  # Display detailed information
    save_results: bool = True  # Save results to file


@dataclass
class EvaluationConfig:
    """Main configuration for model evaluation"""

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Comparison options
    models_to_compare: List[str] = field(
        default_factory=lambda: [
            "your_implementation",
            "standalone_implementation",
            "analysis_implementation",
            "modern_pytorch_implementation",
        ]
    )

    compare_logistic_regression: bool = False
    compare_chandelier_exit: bool = False


# Default configurations
default_config = EvaluationConfig()

# Configuration functions
# =====================


def load_config_from_file(file_path: str) -> EvaluationConfig:
    """Load configuration from JSON file"""
    import json

    try:
        with open(file_path, "r") as f:
            config_dict = json.load(f)

        # Convert dictionary to EvaluationConfig
        config = EvaluationConfig()

        # Handle nested dataclasses
        if "exchange" in config_dict:
            config.exchange = ExchangeConfig(**config_dict["exchange"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "model" in config_dict:
            model_dict = config_dict["model"]
            config.model = ModelConfig()
            if "lorentzian_params" in model_dict:
                config.model.lorentzian_params = model_dict["lorentzian_params"]
            if "logistic_params" in model_dict:
                config.model.logistic_params = model_dict["logistic_params"]
            if "chandelier_params" in model_dict:
                config.model.chandelier_params = model_dict["chandelier_params"]
        if "backtest" in config_dict:
            config.backtest = BacktestConfig(**config_dict["backtest"])
        if "output" in config_dict:
            config.output = OutputConfig(**config_dict["output"])

        # Handle special fields
        if "models_to_compare" in config_dict:
            config.models_to_compare = config_dict["models_to_compare"]
        if "compare_logistic_regression" in config_dict:
            config.compare_logistic_regression = config_dict[
                "compare_logistic_regression"
            ]
        if "compare_chandelier_exit" in config_dict:
            config.compare_chandelier_exit = config_dict["compare_chandelier_exit"]

        return config

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return default_config


def save_config_to_file(config: EvaluationConfig, file_path: str) -> bool:
    """Save configuration to JSON file"""
    import json
    from dataclasses import asdict
    import os

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Helper function to handle Enum values
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")

        # Convert to dictionary
        config_dict = asdict(config)

        with open(file_path, "w") as f:
            json.dump(config_dict, f, default=serialize, indent=4)

        return True

    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Comparison System")

    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--save_config", type=str, help="Save configuration to file")

    parser.add_argument("--symbol", type=str, help="Trading pair (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", type=str, help="Candle timeframe (e.g., 1h)")
    parser.add_argument("--start_date", type=str, help="Start date for backtesting")
    parser.add_argument("--end_date", type=str, help="End date for backtesting")

    parser.add_argument("--initial_balance", type=float, help="Starting balance")
    parser.add_argument(
        "--risk_per_trade", type=float, help="Risk percentage per trade"
    )
    parser.add_argument("--fee_rate", type=float, help="Trading fee rate")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")

    # Comparison flags
    parser.add_argument(
        "--compare_logistic",
        action="store_true",
        help="Compare logistic regression implementations",
    )
    parser.add_argument(
        "--compare_chandelier",
        action="store_true",
        help="Compare chandelier exit implementations",
    )

    return parser.parse_args()


def update_config_from_args(
    config: EvaluationConfig, args: Dict[str, Any]
) -> EvaluationConfig:
    """Update configuration with command-line arguments"""
    for key, value in args.items():
        if value is None:
            continue

        if key == "symbol":
            config.data.symbol = value
        elif key == "timeframe":
            config.data.timeframe = value
        elif key == "start_date":
            config.data.start_date = value
        elif key == "end_date":
            config.data.end_date = value
        elif key == "initial_balance":
            config.backtest.initial_balance = float(value)
        elif key == "risk_per_trade":
            config.backtest.risk_per_trade = float(value)
        elif key == "fee_rate":
            config.backtest.fee_rate = float(value)
        elif key == "output_dir":
            config.output.output_dir = value
        elif key == "no_plots":
            config.output.show_plots = not value
            config.output.save_plots = not value
        elif key == "compare_logistic":
            config.compare_logistic_regression = value
        elif key == "compare_chandelier":
            config.compare_chandelier_exit = value

    return config


# Data Fetching
# =============


class DataFetcher:
    """Fetches historical market data for backtesting"""

    def __init__(self, config: EvaluationConfig):
        """Initialize with configuration"""
        self.config = config
        self.cache_dir = Path(config.data.data_dir) / "cache"

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_filename(self) -> Path:
        """Generate cache filename based on configuration"""
        symbol = self.config.data.symbol.replace("/", "_")
        timeframe = self.config.data.timeframe
        start_date = self.config.data.start_date
        end_date = self.config.data.end_date

        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is valid (not too old)"""
        if not cache_file.exists():
            return False

        # Check file modification time
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        now = datetime.now()

        # Cache is valid if it's less than 1 day old
        return (now - mod_time) < timedelta(days=1)

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        if not self.config.data.use_cached_data:
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
                print(f"Error loading from cache: {e}")

        return None

    def _save_to_cache(self, df: pd.DataFrame) -> bool:
        """Save data to cache"""
        if not self.config.data.use_cached_data:
            return False

        cache_file = self._get_cache_filename()

        try:
            # Save with timestamp as a column
            df_to_save = df.reset_index()
            df_to_save.to_csv(cache_file, index=False)
            print(f"Saved {len(df)} candles to cache: {cache_file}")
            return True
        except Exception as e:
            print(f"Error saving to cache: {e}")
            return False

    def _initialize_exchange(self) -> Optional[ccxt.Exchange]:
        """Initialize exchange connection"""
        try:
            exchange_id = self.config.exchange.name.lower()
            exchange_class = getattr(ccxt, exchange_id)

            exchange = exchange_class(
                {
                    "apiKey": self.config.exchange.api_key,
                    "secret": self.config.exchange.api_secret,
                    "enableRateLimit": True,
                }
            )

            if self.config.exchange.testnet:
                exchange.set_sandbox_mode(True)

            return exchange

        except Exception as e:
            print(f"Error initializing exchange: {e}")
            print("Will use synthetic data instead")
            return None

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data based on configuration"""
        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        print(
            f"Fetching {self.config.data.timeframe} data for {self.config.data.symbol}..."
        )

        try:
            # Initialize exchange
            exchange = self._initialize_exchange()

            if exchange is None:
                raise Exception("Exchange initialization failed")

            # Convert date strings to timestamps
            since = int(
                datetime.strptime(self.config.data.start_date, "%Y-%m-%d").timestamp()
                * 1000
            )
            until = int(
                datetime.strptime(self.config.data.end_date, "%Y-%m-%d").timestamp()
                * 1000
            )

            # Fetch data in chunks to handle large date ranges
            all_candles = []
            current_since = since

            while current_since < until:
                print(
                    f"Fetching chunk from {datetime.fromtimestamp(current_since/1000)}"
                )
                candles = exchange.fetch_ohlcv(
                    self.config.data.symbol,
                    self.config.data.timeframe,
                    since=current_since,
                    limit=1000,
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update since for next iteration
                current_since = candles[-1][0] + 1

                # Safety check
                if len(all_candles) > 50000:
                    break

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Filter by date range
            df = df[self.config.data.start_date : self.config.data.end_date]

            print(f"Fetched {len(df)} candles successfully")

            # Save to cache
            self._save_to_cache(df)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Generating synthetic data instead...")
            return self._generate_synthetic_data(1000)  # Use 1000 candles for testing

    def _generate_synthetic_data(self, length: int = 1000) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        print(f"Generating {length} synthetic candles")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate timestamps
        start_date = datetime.strptime(self.config.data.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.data.end_date, "%Y-%m-%d")

        # Calculate date range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 30  # Default to 30 days

        # Determine candle interval in days
        interval_map = {
            "1m": 1 / 1440,
            "5m": 5 / 1440,
            "15m": 15 / 1440,
            "30m": 30 / 1440,
            "1h": 1 / 24,
            "2h": 2 / 24,
            "4h": 4 / 24,
            "6h": 6 / 24,
            "8h": 8 / 24,
            "12h": 12 / 24,
            "1d": 1,
            "3d": 3,
            "1w": 7,
            "1M": 30,
        }

        interval = interval_map.get(self.config.data.timeframe, 1 / 24)  # Default to 1h

        # Calculate number of candles
        num_candles = min(int(date_range / interval), length)

        # Generate timestamps
        timestamps = [
            start_date + timedelta(days=i * interval) for i in range(num_candles)
        ]

        # Generate price data
        close = np.zeros(num_candles)
        close[0] = 100  # Starting price

        # Random walk with trend and noise
        for i in range(1, num_candles):
            # Trend component (momentum)
            momentum = (
                0.1 * (close[i - 1] - close[max(0, i - 5)]) / close[max(0, i - 5)]
            )

            # Mean reversion
            mean_reversion = -0.05 * (close[i - 1] - close[0]) / close[0]

            # Random noise
            noise = np.random.normal(0, 0.02)

            # Add regime changes periodically
            regime = 0.05 if i % 200 < 100 else -0.03

            # Combine components
            change = momentum + mean_reversion + noise + regime
            close[i] = close[i - 1] * (1 + change)

        # Generate OHLCV data
        high = close * (1 + np.abs(np.random.normal(0, 0.01, num_candles)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, num_candles)))
        open_price = low + (high - low) * np.random.random(num_candles)

        # Generate volume
        volume = 1000000 + 500000 * np.random.random(num_candles)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=timestamps,
        )

        return df


# More sections will be added here, including model implementations,
# backtesting functionality, and visualization.

# Model Implementations
# ====================


class BaseModel:
    """Base model class for all model wrappers"""

    def __init__(self):
        """Initialize base model"""
        self.fitted = False

    def fit(self, df):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement fit")

    def predict(self, df):
        """Generate predictions"""
        raise NotImplementedError("Subclasses must implement predict")


# Import custom model implementations
try:
    from your_implementation import LorentzianANN as YourImplementation

    print("Your implementation loaded successfully")
except ImportError:
    print("Your implementation not found, skipping.")
    YourImplementation = None

try:
    from standalone_implementation import LorentzianANN as StandaloneImplementation

    print("Standalone implementation loaded successfully")
except ImportError:
    print("Standalone implementation not found, skipping.")
    StandaloneImplementation = None

try:
    from analysis_implementation import LorentzianANN as AnalysisImplementation

    print("Analysis implementation loaded successfully")
except ImportError:
    print("Analysis implementation not found, skipping.")
    AnalysisImplementation = None

try:
    from modern_pytorch_implementation import ModernLorentzian as ModernImplementation

    print("Modern implementation loaded successfully")
except ImportError:
    print("Modern implementation not found, skipping.")
    ModernImplementation = None

# Logistic Regression Implementations
try:
    from your_logistic_regression import LogisticRegression as YourLogisticRegression

    print("Your logistic regression loaded successfully")
except ImportError:
    print("Your logistic regression not found, skipping.")
    YourLogisticRegression = None

try:
    from example_logistic_regression import (
        LogisticRegression as ExampleLogisticRegression,
    )

    print("Example logistic regression loaded successfully")
except ImportError:
    print("Example logistic regression not found, skipping.")
    ExampleLogisticRegression = None

try:
    from src_logistic_regression import LogisticRegression as SrcLogisticRegression

    print("Source logistic regression loaded successfully")
except ImportError:
    print("Source logistic regression not found, skipping.")
    SrcLogisticRegression = None

# Chandelier Exit Implementations
try:
    from your_chandelier_exit import ChandelierExitIndicator as YourChandelierExit

    print("Your chandelier exit loaded successfully")
except ImportError:
    try:
        from your_chandelier_exit import ChandelierExit as YourChandelierExit

        print("Your chandelier exit loaded successfully")
    except (ImportError, AttributeError):
        print("Your chandelier exit not found, skipping.")
        YourChandelierExit = None

try:
    from example_chandelier_exit import ChandelierExit as ExampleChandelierExit

    print("Example chandelier exit loaded successfully")
except ImportError:
    print("Example chandelier exit not found, skipping.")
    ExampleChandelierExit = None

try:
    from src_chandelier_exit import ChandelierExit as SrcChandelierExit

    print("Source chandelier exit loaded successfully")
except ImportError:
    print("Source chandelier exit not found, skipping.")
    SrcChandelierExit = None


# Wrapper classes for common interface
class LorentzianModelWrapper(BaseModel):
    """Wrapper for Lorentzian model implementations"""

    def __init__(self, model_class, config: ModelConfig):
        super().__init__()

        # Initialize specific model parameters
        model_params = config.lorentzian_params

        # Set up device
        device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Create model instance with only the parameters it accepts
        # Different implementations might have different parameter sets
        common_params = {
            "lookback_bars": model_params.get("lookback_bars", 30),
            "prediction_bars": model_params.get("prediction_bars", 5),
            "k_neighbors": model_params.get("k_neighbors", 15),
        }

        # Try to initialize with minimal parameters first
        self.model = model_class(**common_params)

        self.name = model_class.__name__
        self.description = f"Lorentzian model using {model_class.__name__}"

    def fit(self, df: pd.DataFrame) -> None:
        """Fit model to data"""
        # Generate features for Lorentzian model
        features = self._generate_features(df)

        # Get prices for the Lorentzian model (needed by some implementations)
        prices = df["close"].values

        # Try to fit with different parameter combinations based on implementation
        try:
            # Method 1: fit(features, prices)
            self.model.fit(features, prices)
        except TypeError:
            try:
                # Method 2: fit(features)
                self.model.fit(features)
            except Exception as e:
                print(f"Error fitting model {self.name}: {e}")
                raise

        self.fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Generate features
        features = self._generate_features(df)

        # Generate predictions
        signals = self.model.predict(features)

        # Return dataframe with prediction column
        predictions = df.copy()
        predictions["signal"] = signals
        return predictions

    def _generate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Generate features for Lorentzian model"""
        # Basic price-based features
        features = []

        # Close price normalized
        price = df["close"].values
        returns = np.diff(price) / price[:-1]
        returns = np.append([0], returns)  # Pad with zero

        # Add returns
        features.append(returns)

        # Momentum features (price relatives)
        for period in [5, 10, 20]:
            if len(price) > period:
                momentum = price / np.roll(price, period) - 1
                momentum[:period] = 0  # Replace initial NaNs with 0
                features.append(momentum)

        # Volatility features
        for period in [10, 20]:
            if len(returns) > period:
                volatility = pd.Series(returns).rolling(window=period).std().values
                volatility[: period - 1] = volatility[period - 1]  # Fill initial NaNs
                features.append(volatility)

        # Combine features
        X = np.column_stack(features)

        return X


class LogisticRegressionWrapper:
    """Wrapper for logistic regression models"""

    def __init__(self, model_class=None, config=None):
        """Initialize with model_class and config"""
        self.name = model_class.__name__ if model_class else "LogisticRegressionWrapper"
        self.config = config

        # Extract parameters from config
        self.lookback = (
            getattr(config.model, "lookback", 20) if hasattr(config, "model") else 20
        )
        self.learning_rate = (
            getattr(config.model, "learning_rate", 0.01)
            if hasattr(config, "model")
            else 0.01
        )
        self.iterations = (
            getattr(config.model, "iterations", 1000)
            if hasattr(config, "model")
            else 1000
        )
        self.threshold = (
            getattr(config.model, "threshold", 0.5) if hasattr(config, "model") else 0.5
        )

        # Initialize the model
        try:
            if model_class.__name__ == "ExampleLogisticRegression":
                self.model = model_class(
                    lookback=self.lookback, learning_rate=self.learning_rate
                )
            elif model_class.__name__ == "SrcLogisticRegression":
                self.model = model_class(
                    lookback=self.lookback, learning_rate=self.learning_rate
                )
            elif model_class.__name__ == "YourLogisticRegression":
                self.model = model_class(
                    lookback=self.lookback, learning_rate=self.learning_rate
                )
            else:
                # Try different initialization approaches
                try:
                    # Try with config
                    self.model = model_class(config)
                except Exception:
                    try:
                        # Try with individual parameters
                        self.model = model_class(
                            lookback=self.lookback,
                            learning_rate=self.learning_rate,
                            iterations=self.iterations,
                            threshold=self.threshold,
                        )
                    except Exception:
                        # Fall back to minimal parameters
                        self.model = model_class(self.lookback)
        except Exception as e:
            print(f"Error initializing model {self.name}: {str(e)}")
            # Create a placeholder model
            self.model = None

    def fit(self, df):
        """Fit the model to the data"""
        if self.model is None:
            print("Model not initialized, cannot fit.")
            return self

        try:
            # Try different fitting approaches
            if hasattr(self.model, "fit") and callable(self.model.fit):
                # Check for standard fit interface (df, target)
                x_train = self._prepare_features(df)
                y_train = self._prepare_labels(df)

                # Check the fit signature
                import inspect

                sig = inspect.signature(self.model.fit)
                if len(sig.parameters) >= 2:
                    self.model.fit(x_train, y_train)
                else:
                    # Some models take just the DataFrame
                    self.model.fit(df)
            elif hasattr(self.model, "train") and callable(self.model.train):
                # Try train method instead
                self.model.train(df)
            else:
                print(f"Model {self.name} has no fit or train method")
        except Exception as e:
            print(f"Error fitting model {self.name}: {str(e)}")

        return self

    def predict(self, df):
        """Generate predictions from the model"""
        if self.model is None:
            print("Model not initialized, returning zeros.")
            return np.zeros(len(df))

        try:
            # Try different prediction approaches
            if hasattr(self.model, "predict") and callable(self.model.predict):
                # Standard predict interface
                x_test = self._prepare_features(df)

                # Check if the model expects DataFrame or numpy
                if (
                    hasattr(self.model, "_expects_dataframe")
                    and self.model._expects_dataframe
                ):
                    return self.model.predict(df)
                else:
                    return self.model.predict(x_test)
            elif hasattr(self.model, "get_signals") and callable(
                self.model.get_signals
            ):
                # Some models use get_signals
                return self.model.get_signals(df)
            elif hasattr(self.model, "generate_signals") and callable(
                self.model.generate_signals
            ):
                # Or generate_signals
                return self.model.generate_signals(df)
            else:
                print(f"Model {self.name} has no compatible prediction method")
                return np.zeros(len(df))
        except Exception as e:
            print(f"Error predicting with model {self.name}: {str(e)}")
            return np.zeros(len(df))

    def _prepare_features(self, df):
        """Prepare features for the model"""
        # Convert DataFrame to numpy array for features
        if "close" in df.columns:
            # Create features from OHLCV data
            features = df[["open", "high", "low", "close", "volume"]].values
            return features
        else:
            # Just use all numerical columns
            return df.select_dtypes(include=["number"]).values

    def _prepare_labels(self, df):
        """Prepare labels for the model (for training)"""
        # For training, we'll use price movement direction as a simple target
        if "close" in df.columns:
            # Use future price direction (1 for up, 0 for down)
            closes = df["close"].values
            targets = np.zeros(len(closes))
            for i in range(len(closes) - 1):
                targets[i] = 1 if closes[i + 1] > closes[i] else 0
            # Last value can't be determined, set to 0
            targets[-1] = 0
            return targets
        else:
            # If no price data, return zeros
            return np.zeros(len(df))


class ChandelierExitWrapper:
    """Wrapper for chandelier exit models"""

    def __init__(self, model_class=None, config=None):
        """Initialize with model_class and config"""
        self.name = model_class.__name__ if model_class else "ChandelierExitWrapper"
        self.config = config

        # Extract parameters from config
        if hasattr(config, "model"):
            self.atr_period = (
                getattr(config.model, "atr_period", 22)
                if hasattr(config.model, "atr_period")
                else 22
            )
            self.atr_multiplier = (
                getattr(config.model, "atr_multiplier", 3.0)
                if hasattr(config.model, "atr_multiplier")
                else 3.0
            )
            self.use_close = (
                getattr(config.model, "use_close", True)
                if hasattr(config.model, "use_close")
                else True
            )
        else:
            self.atr_period = 22
            self.atr_multiplier = 3.0
            self.use_close = True

        # Initialize the model
        try:
            if model_class.__name__ == "ExampleChandelierExit":
                self.model = model_class(
                    length=self.atr_period, mult=self.atr_multiplier
                )
            elif model_class.__name__ == "SrcChandelierExit":
                self.model = model_class(
                    length=self.atr_period, mult=self.atr_multiplier
                )
            elif model_class.__name__ == "YourChandelierExit":
                self.model = model_class(
                    length=self.atr_period, mult=self.atr_multiplier
                )
            else:
                # Try different initialization approaches
                try:
                    # Try with config
                    self.model = model_class(config)
                except Exception:
                    try:
                        # Try with individual parameters
                        self.model = model_class(
                            length=self.atr_period,
                            mult=self.atr_multiplier,
                            use_close=self.use_close,
                        )
                    except Exception:
                        # Fall back to minimal parameters
                        self.model = model_class(self.atr_period, self.atr_multiplier)
        except Exception as e:
            print(f"Error initializing model {self.name}: {str(e)}")
            # Create a placeholder model
            self.model = None

    def fit(self, df):
        """Fit the model to the data (if needed)"""
        # Many Chandelier Exit models don't need fitting
        if self.model is None:
            print("Model not initialized, cannot fit.")
            return self

        try:
            # Try different fitting approaches if available
            if hasattr(self.model, "fit") and callable(self.model.fit):
                self.model.fit(df)
            elif hasattr(self.model, "train") and callable(self.model.train):
                self.model.train(df)
            # It's fine if no fitting is needed for this type of model
        except Exception as e:
            print(f"Error fitting model {self.name}: {str(e)}")

        return self

    def predict(self, df):
        """Generate signals from the model"""
        if self.model is None:
            print("Model not initialized, calculating basic signals.")
            return self._calculate_basic_signals(df)

        try:
            # Try different prediction approaches
            if hasattr(self.model, "predict") and callable(self.model.predict):
                return self.model.predict(df)
            elif hasattr(self.model, "get_signals") and callable(
                self.model.get_signals
            ):
                return self.model.get_signals(df)
            elif hasattr(self.model, "generate_signals") and callable(
                self.model.generate_signals
            ):
                return self.model.generate_signals(df)
            elif hasattr(self.model, "calculate_signals") and callable(
                self.model.calculate_signals
            ):
                return self.model.calculate_signals(df)
            else:
                print(
                    f"Model {self.name} has no compatible prediction method, using basic calculation"
                )
                return self._calculate_basic_signals(df)
        except Exception as e:
            print(f"Error predicting with model {self.name}: {str(e)}")
            return self._calculate_basic_signals(df)

    def _calculate_basic_signals(self, df):
        """Calculate basic chandelier exit signals as a fallback"""
        # Simple implementation of chandelier exit
        import pandas as pd
        import numpy as np

        signals = np.zeros(len(df))

        # Copy dataframe to avoid modifying original
        temp_df = df.copy()

        # Calculate ATR
        if "high" in temp_df.columns and "low" in temp_df.columns:
            # Calculate True Range
            temp_df["tr0"] = abs(temp_df["high"] - temp_df["low"])
            temp_df["tr1"] = abs(temp_df["high"] - temp_df["close"].shift(1))
            temp_df["tr2"] = abs(temp_df["low"] - temp_df["close"].shift(1))
            temp_df["tr"] = temp_df[["tr0", "tr1", "tr2"]].max(axis=1)

            # Calculate ATR
            temp_df["atr"] = temp_df["tr"].rolling(window=self.atr_period).mean()

            # Calculate chandelier exit
            temp_df["highest"] = temp_df["high"].rolling(window=self.atr_period).max()
            temp_df["lowest"] = temp_df["low"].rolling(window=self.atr_period).min()

            temp_df["long_exit"] = temp_df["highest"] - (
                temp_df["atr"] * self.atr_multiplier
            )
            temp_df["short_exit"] = temp_df["lowest"] + (
                temp_df["atr"] * self.atr_multiplier
            )

            # Simple signal generation
            for i in range(self.atr_period, len(temp_df)):
                price = temp_df["close"].iloc[i]
                long_exit = temp_df["long_exit"].iloc[i]
                short_exit = temp_df["short_exit"].iloc[i]

                if price > long_exit:
                    signals[i] = 1  # Long signal
                elif price < short_exit:
                    signals[i] = -1  # Short signal

        return signals


class ModelEvaluator:
    """Model evaluation and comparison system"""

    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        self.results = {}

        # Initialize verbose mode
        self.verbose = (
            hasattr(config, "output")
            and hasattr(config.output, "verbose")
            and config.output.verbose
        )

        if self.verbose:
            print("Initializing ModelEvaluator")

        # Initialize data fetcher
        self.data_fetcher = DataFetcher(config)

    def get_data(self):
        """Get data for model evaluation"""
        if self.verbose:
            print("Fetching data for model evaluation")
        return self.data_fetcher.fetch_data()

    def generate_synthetic_data(self, length=1000):
        """Generate synthetic data for testing"""
        if self.verbose:
            print(f"Generating {length} synthetic candles")
        return self.data_fetcher._generate_synthetic_data(length)

    def backtest_model(self, model, df):
        """Run backtest on model"""
        if self.verbose:
            print(
                f"Backtesting model: {model.name if hasattr(model, 'name') else 'Unknown'}"
            )

        # Get predictions
        try:
            predictions = model.predict(df)

            # Make sure we have a numpy array
            if isinstance(predictions, pd.DataFrame):
                predictions = (
                    predictions["signal"].values
                    if "signal" in predictions.columns
                    else predictions.values
                )

            if torch.is_tensor(predictions):
                predictions = predictions.cpu().detach().numpy()

            # Flatten if needed
            if hasattr(predictions, "shape") and len(predictions.shape) > 1:
                predictions = predictions.flatten()

            # Ensure we have the right length
            if len(predictions) != len(df):
                raise ValueError(
                    f"Predictions length {len(predictions)} doesn't match data length {len(df)}"
                )

        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            predictions = np.zeros(len(df))

        # Create backtesting DataFrame
        backtest_df = df.copy()
        backtest_df["signal"] = predictions

        # Initialize account parameters
        initial_balance = (
            self.config.backtest.initial_balance
            if hasattr(self.config, "backtest")
            else 10000.0
        )
        risk_per_trade = (
            self.config.backtest.risk_per_trade / 100
            if hasattr(self.config, "backtest")
            else 0.02
        )
        fee_rate = (
            self.config.backtest.fee_rate if hasattr(self.config, "backtest") else 0.001
        )

        # Add columns for backtesting
        backtest_df["balance"] = initial_balance
        backtest_df["position"] = 0  # -1 (short), 0 (no position), 1 (long)
        backtest_df["equity"] = initial_balance

        # Trading stats
        current_balance = initial_balance
        position = 0
        entry_price = 0
        entry_balance = initial_balance
        trades = []

        # Iterate through the backtest
        for i in range(1, len(backtest_df)):
            # Get current signal and price
            prev_signal = backtest_df["signal"].iloc[i - 1]
            current_signal = backtest_df["signal"].iloc[i]
            price = backtest_df["close"].iloc[i]
            prev_price = backtest_df["close"].iloc[i - 1]

            # Update balance based on position
            if position != 0:
                # Calculate P&L
                if position == 1:  # Long
                    pnl_pct = (price / entry_price - 1) - fee_rate
                else:  # Short
                    pnl_pct = (1 - price / entry_price) - fee_rate

                # Update current balance
                current_balance = entry_balance * (
                    1 + position * pnl_pct * risk_per_trade * 10
                )

            # Check for signal changes (entry/exit)
            if current_signal != position:
                # If we have a position and signal changed, close it
                if position != 0:
                    # Record trade
                    exit_price = price
                    pnl = current_balance - entry_balance
                    pnl_pct = pnl / entry_balance * 100
                    trades.append(
                        {
                            "entry_time": backtest_df.index[i - position_duration],
                            "exit_time": backtest_df.index[i],
                            "direction": "Long" if position == 1 else "Short",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                        }
                    )

                    # Reset position
                    position = 0

                # Check for new entry
                if current_signal != 0:
                    # Enter new position
                    position = current_signal
                    entry_price = price
                    entry_balance = current_balance
                    position_duration = 0

            # Update position duration
            if position != 0:
                position_duration += 1

            # Update backtest DataFrame
            backtest_df["balance"].iloc[i] = current_balance
            backtest_df["position"].iloc[i] = position
            backtest_df["equity"].iloc[i] = current_balance

        # Calculate backtest statistics
        total_return = current_balance - initial_balance
        total_return_pct = (total_return / initial_balance) * 100

        # Handle case with no trades
        if len(trades) == 0:
            stats = {
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "avg_trade_pct": 0,
                "long_trades": 0,
                "short_trades": 0,
                "signal_change_rate": 0,
            }
            return backtest_df, stats

        # Calculate win rate
        winning_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100

        # Calculate profit factor
        gross_profit = sum([t["pnl"] for t in trades if t["pnl"] > 0])
        gross_loss = abs(sum([t["pnl"] for t in trades if t["pnl"] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate maximum drawdown
        peak = initial_balance
        drawdown = 0
        max_drawdown = 0

        for balance in backtest_df["balance"]:
            if balance > peak:
                peak = balance

            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate Sharpe ratio
        daily_returns = backtest_df["balance"].pct_change().dropna()
        sharpe_ratio = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() != 0
            else 0
        )

        # Signal activity
        signal_changes = (backtest_df["signal"].diff() != 0).sum()
        signal_change_rate = signal_changes / len(backtest_df) * 100

        # Create statistics dictionary
        stats = {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(trades),
            "avg_trade_pct": sum([t["pnl_pct"] for t in trades]) / len(trades),
            "long_trades": len([t for t in trades if t["direction"] == "Long"]),
            "short_trades": len([t for t in trades if t["direction"] == "Short"]),
            "signal_change_rate": signal_change_rate,
        }

        return backtest_df, stats

    def compare_models(self, df=None, model_list=None):
        """Compare multiple models on the same data"""
        if df is None:
            df = self.get_data()

        if model_list is None:
            # Use default model list from config
            model_list = self.config.models_to_compare

            # Add logistic regression models if enabled
            if self.config.compare_logistic_regression:
                model_list.extend(
                    [
                        "your_logistic_regression",
                        "example_logistic_regression",
                        "src_logistic_regression",
                    ]
                )

            # Add chandelier exit models if enabled
            if self.config.compare_chandelier_exit:
                model_list.extend(
                    [
                        "your_chandelier_exit",
                        "example_chandelier_exit",
                        "src_chandelier_exit",
                    ]
                )

        # Clear results for new comparison
        self.results = {}

        # Filter out models that are not available
        available_models = []
        for model_name in model_list:
            if model_name == "your_implementation" and YourImplementation is None:
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "standalone_implementation"
                and StandaloneImplementation is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "analysis_implementation"
                and AnalysisImplementation is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "modern_pytorch_implementation"
                and ModernImplementation is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "your_logistic_regression"
                and YourLogisticRegression is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "example_logistic_regression"
                and ExampleLogisticRegression is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "src_logistic_regression"
                and SrcLogisticRegression is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif model_name == "your_chandelier_exit" and YourChandelierExit is None:
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif (
                model_name == "example_chandelier_exit"
                and ExampleChandelierExit is None
            ):
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            elif model_name == "src_chandelier_exit" and SrcChandelierExit is None:
                if self.verbose:
                    print(f"Model {model_name} not available, skipping")
                continue
            else:
                available_models.append(model_name)

        if not available_models:
            print("No models available for comparison")
            return

        # Evaluate each model
        for model_name in available_models:
            try:
                if self.verbose:
                    print(f"\nEvaluating {model_name}...")
                self.evaluate_model(model_name, df)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                traceback.print_exc()
                continue

        # Compare model results
        if self.verbose:
            print("\nComparison Results:")

        comparison_table = []
        for model_name, result in self.results.items():
            stats = result.get("stats", {})
            row = [model_name]

            # Key metrics
            row.append(f"{stats.get('total_return', 0):.2f}%")
            row.append(f"{stats.get('win_rate', 0):.2f}%")
            row.append(f"{stats.get('profit_factor', 0):.2f}")
            row.append(f"{stats.get('max_drawdown', 0):.2f}%")
            row.append(f"{stats.get('sharpe_ratio', 0):.2f}")
            row.append(f"{stats.get('total_trades', 0)}")

            comparison_table.append(row)

        # Check if comparison_table has values
        if comparison_table:
            # Print comparison table
            header = [
                "Model",
                "Return (%)",
                "Win Rate (%)",
                "Profit Factor",
                "Max Drawdown (%)",
                "Sharpe Ratio",
                "Total Trades",
            ]

            # Print table using tabulate
            print("\nModel Comparison:")
            print(tabulate(comparison_table, headers=header, tablefmt="grid"))

            # Save results
            self.save_results(df)
        else:
            print("No models were successfully evaluated for comparison")

    def evaluate_model(self, model_name, df=None, clear_results=False):
        """Evaluate a specific model"""
        if self.verbose:
            print(f"\nEvaluating model: {model_name}")

        # Check if model already exists in results
        if model_name in self.results and not clear_results:
            if self.verbose:
                print(f"Model {model_name} already evaluated, returning cached results")
            return self.results[model_name]

        if df is None:
            df = self.get_data()

        # Create model if not already in results
        model = None

        # Try to create model based on name
        if model_name == "your_implementation":
            if YourImplementation:
                model = LorentzianModelWrapper(YourImplementation, self.config.model)
        elif model_name == "standalone_implementation":
            if StandaloneImplementation:
                model = LorentzianModelWrapper(
                    StandaloneImplementation, self.config.model
                )
        elif model_name == "analysis_implementation":
            if AnalysisImplementation:
                model = LorentzianModelWrapper(
                    AnalysisImplementation, self.config.model
                )
        elif model_name == "modern_pytorch_implementation":
            if ModernImplementation:
                model = LorentzianModelWrapper(ModernImplementation, self.config.model)
        elif model_name == "your_logistic_regression":
            if YourLogisticRegression:
                model = LogisticRegressionWrapper(
                    YourLogisticRegression, self.config.model
                )
        elif model_name == "example_logistic_regression":
            if ExampleLogisticRegression:
                model = LogisticRegressionWrapper(
                    ExampleLogisticRegression, self.config.model
                )
        elif model_name == "src_logistic_regression":
            if SrcLogisticRegression:
                model = LogisticRegressionWrapper(
                    SrcLogisticRegression, self.config.model
                )
        elif model_name == "your_chandelier_exit":
            if YourChandelierExit:
                model = ChandelierExitWrapper(YourChandelierExit, self.config.model)
        elif model_name == "example_chandelier_exit":
            if ExampleChandelierExit:
                model = ChandelierExitWrapper(ExampleChandelierExit, self.config.model)
        elif model_name == "src_chandelier_exit":
            if SrcChandelierExit:
                model = ChandelierExitWrapper(SrcChandelierExit, self.config.model)
        else:
            print(f"Unknown model: {model_name}")
            return None

        if model is None:
            print(f"Model {model_name} could not be created, skipping evaluation")
            return None

        # Train model
        try:
            if self.verbose:
                print(f"Training model {model_name}...")
            model.fit(df)

            # Backtest model
            if self.verbose:
                print(f"Backtesting model {model_name}...")
            backtest_df, stats = self.backtest_model(model, df)

            # Save results
            self.results[model_name] = {
                "model": model,
                "backtest_df": backtest_df,
                "stats": stats,
            }

            return self.results[model_name]
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            traceback.print_exc()
            return None

    def save_results(self, df=None):
        """Save evaluation results to files"""
        if not self.results:
            print("No results to save")
            return

        if not hasattr(self.config, "output") or not hasattr(
            self.config.output, "output_dir"
        ):
            print("No output directory specified in config")
            return

        # Create output directory
        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual model results
        for model_name, result in self.results.items():
            if "backtest_df" in result:
                csv_path = output_dir / f"{model_name}_backtest.csv"
                result["backtest_df"].to_csv(csv_path)
                print(f"Saved {model_name} backtest results to {csv_path}")

        # Save comparison table
        comparison_df = self.compare_models(df)
        if comparison_df is not None:
            csv_path = output_dir / "model_comparison.csv"
            comparison_df.to_csv(csv_path, index=False)
            print(f"Saved comparison results to {csv_path}")


def main():
    """Main function to run model evaluation"""
    parser = argparse.ArgumentParser(description="Trading Model Evaluator")

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default="config_samples/default_config.json",
        help="Path to config file",
    )

    # Model comparison flags
    parser.add_argument(
        "--compare_logistic",
        action="store_true",
        help="Compare logistic regression implementations",
    )

    parser.add_argument(
        "--compare_chandelier",
        action="store_true",
        help="Compare chandelier exit implementations",
    )

    parser.add_argument(
        "--compare_lorentzian",
        action="store_true",
        help="Compare Lorentzian implementations",
    )

    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all available model implementations",
    )

    # Optional model list
    parser.add_argument(
        "--models", type=str, help="Comma-separated list of models to compare"
    )

    # Verbose mode
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Debug mode
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output with full tracebacks"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file {config_path}: {e}")
        config_data = {}

    # Create config object using SimpleNamespace for dot access
    config = json.loads(
        json.dumps(config_data),
        object_hook=lambda d: type("Config", (), {k: v for k, v in d.items()}),
    )

    # Add command line flags to config
    config.compare_logistic_regression = args.compare_logistic
    config.compare_chandelier_exit = args.compare_chandelier
    config.compare_lorentzian_models = args.compare_lorentzian
    config.verbose = args.verbose
    config.debug = args.debug

    # If compare_all is set, set all comparison flags to True
    if args.compare_all:
        config.compare_logistic_regression = True
        config.compare_chandelier_exit = True
        config.compare_lorentzian_models = True

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Get data
    df = evaluator.get_data()
    if df is None or len(df) == 0:
        print("No data found. Generating synthetic data...")
        df = evaluator.generate_synthetic_data(1000)

    # Determine models to compare
    models_to_compare = None
    if args.models:
        models_to_compare = args.models.split(",")

    # Run model comparison
    # If any comparison flag is set or specific models are requested, run comparison
    if (
        args.compare_logistic
        or args.compare_chandelier
        or args.compare_lorentzian
        or args.compare_all
        or models_to_compare
    ):
        print("Running model comparison...")
        comparison = evaluator.compare_models(df, models_to_compare)
        if comparison is not None:
            print("\nModel Comparison Results:")
            print(comparison)

            # Save results
            evaluator.save_results(df)
        else:
            print("No comparison results available.")
    else:
        print(
            "No comparison mode selected. Use --compare_all to compare all models, "
            "or use --compare_logistic, --compare_chandelier, or --compare_lorentzian "
            "to compare specific model types."
        )
