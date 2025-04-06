"""
Comparison Configuration

This module provides configuration options for the Lorentzian model comparison system,
allowing customization of exchange, market type, trading pair, and other parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import os


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
    rate_limit: bool = True  # Enable rate limiting
    timeout: int = 30000  # Timeout in ms

    # Load API keys from environment variables if available
    def __post_init__(self):
        if not self.api_key and os.environ.get(f"{self.name.upper()}_API_KEY"):
            self.api_key = os.environ.get(f"{self.name.upper()}_API_KEY")
        if not self.api_secret and os.environ.get(f"{self.name.upper()}_API_SECRET"):
            self.api_secret = os.environ.get(f"{self.name.upper()}_API_SECRET")


@dataclass
class MarketConfig:
    """Market configuration"""

    symbol: str = "SOL/USDT"  # Trading pair
    market_type: MarketType = MarketType.SPOT  # Market type
    timeframe: str = "5m"  # Candle timeframe
    leverage: int = 1  # Leverage (for futures)
    candle_limit: int = 1000  # Number of candles to fetch


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_balance: float = 10000.0  # Starting balance
    position_size: float = 0.1  # Position size as fraction of balance
    max_positions: int = 1  # Maximum number of concurrent positions
    allow_shorts: bool = True  # Allow short positions
    maker_fee: float = 0.0002  # Maker fee (0.02%)
    taker_fee: float = 0.0005  # Taker fee (0.05%)
    slippage: float = 0.0005  # Slippage estimation (0.05%)
    funding_rate: Optional[float] = None  # Fixed funding rate (None = use actual)
    order_type: OrderType = OrderType.MARKET  # Order type
    compound_returns: bool = True  # Use compounding


@dataclass
class ComparisonConfig:
    """Main configuration for model comparison"""

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Data settings
    use_cached_data: bool = True  # Use cached data if available
    data_cache_dir: str = "data/cache"

    # Display settings
    show_plots: bool = True
    save_plots: bool = True
    plot_filename: str = "model_comparison.png"
    performance_plot_filename: str = "model_performance_comparison.png"
    detailed_report_filename: str = "detailed_comparison_report.csv"

    # Model settings
    use_gpu: bool = True  # Use GPU acceleration if available


# Default configuration
default_config = ComparisonConfig()

# Binance BTCUSDT Futures configuration example
binance_btc_futures_config = ComparisonConfig(
    exchange=ExchangeConfig(name="binance"),
    market=MarketConfig(
        symbol="BTC/USDT",
        market_type=MarketType.FUTURES,
        timeframe="1h",
        leverage=3,
        candle_limit=500,
    ),
    backtest=BacktestConfig(
        initial_balance=10000.0, position_size=0.1, allow_shorts=True
    ),
)

# Bitget SOL Futures configuration example
bitget_sol_futures_config = ComparisonConfig(
    exchange=ExchangeConfig(name="bitget"),
    market=MarketConfig(
        symbol="SOL/USDT:USDT",
        market_type=MarketType.FUTURES,
        timeframe="15m",
        leverage=5,
        candle_limit=1000,
    ),
    backtest=BacktestConfig(
        initial_balance=5000.0,
        position_size=0.2,
        allow_shorts=True,
        maker_fee=0.0001,
        taker_fee=0.0003,
    ),
)

# Add more preset configurations as needed
"""
TODO List for Advanced Comparison System:
1. Data Management
   - Implement data caching to avoid repeated API calls
   - Add support for historical data from different sources
   - Support for importing CSV/custom data

2. Exchange Integration
   - Add support for more exchanges (Binance, Bitget, Bybit, etc.)
   - Implement proper handling of futures specific parameters
   - Calculate accurate funding fees based on historical data

3. Backtesting Engine
   - Create a realistic backtesting engine with order simulation
   - Implement proper position sizing and management
   - Support for different order types and execution logic
   - Track liquidation events and margin requirements

4. Performance Metrics
   - Add comprehensive trading metrics (Sharpe, Sortino, Calmar, etc.)
   - Calculate drawdown statistics and risk metrics
   - Generate equity curves and detailed trade logs
   - Compare models with statistical significance tests

5. Visualization
   - Create interactive charts for trade analysis
   - Generate detailed performance reports
   - Visualize differences between model predictions
   
6. Configuration System
   - Create a flexible configuration system with presets
   - Support for loading configurations from files
   - Command-line interface for running comparisons

7. Optimization
   - Implement parameter optimization for models
   - Evaluate models across multiple timeframes and symbols
   - Cross-validation to prevent overfitting
"""
