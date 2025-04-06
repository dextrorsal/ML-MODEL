"""
Advanced Lorentzian Model Comparison

This script compares the performance of different Lorentzian implementations
using the advanced backtester, which provides detailed metrics including:
- Realistic fee calculation
- Funding rates for futures
- Position sizing and leverage effects
- Drawdown and risk metrics
- Advanced performance statistics (Sharpe, Sortino, Calmar)

The implementations being compared:
1. Your Implementation (src/models/strategy/lorentzian_classifier.py)
2. Standalone Version (example-files/strategies/LorentzianStrategy/lorentzian_classifier.py)
3. Analysis Version (example-files/analyze_lorentzian_ann.py)
4. Modern PyTorch Version (example-files/strategies/LorentzianStrategy/models/primary/lorentzian_classifier.py)
"""

import sys
import os
import asyncio
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the comparison modules
from comparison.your_implementation import LorentzianANN as YourImplementation
from comparison.standalone_implementation import LorentzianANN as StandaloneLorentzian
from comparison.analysis_implementation import LorentzianANN as AnalysisLorentzian
from comparison.modern_pytorch_implementation import ModernLorentzian

# Import configuration and backtester
from comparison.comparison_config import (
    ComparisonConfig,
    MarketType,
    OrderType,
    default_config,
    binance_btc_futures_config,
    bitget_sol_futures_config,
)
from comparison.advanced_backtester import AdvancedBacktester


def prepare_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Prepare features for model training and prediction

    Args:
        df: DataFrame with OHLCV data

    Returns:
        torch.Tensor with prepared features
    """
    # Calculate basic price features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Price changes over different periods
    for period in [5, 10, 20, 50]:
        df[f"price_change_{period}"] = df["close"].pct_change(period)

    # Volatility features
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()

    # Volume features
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Create feature matrix
    feature_columns = [
        "returns",
        "log_returns",
        "price_change_5",
        "price_change_10",
        "price_change_20",
        "price_change_50",
        "volatility_10",
        "volatility_20",
        "volume_change",
        "volume_ma_ratio",
    ]

    # Drop NaN values
    features_df = df[feature_columns].dropna()

    # Convert to PyTorch tensor
    features = torch.tensor(features_df.values, dtype=torch.float32)

    return features


class ModelComparison:
    """Class for comparing multiple Lorentzian implementations"""

    def __init__(self, config: ComparisonConfig):
        """Initialize comparison with configuration"""
        self.config = config
        self.backtester = AdvancedBacktester(config)
        self.results = {}
        self.models = {}

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all model implementations"""
        try:
            print("Initializing models...")

            # Common parameters for fair comparison
            lookback = 50
            prediction_bars = 4
            k_neighbors = 20

            # Your implementation
            self.models["your_implementation"] = YourImplementation(
                lookback_bars=lookback,
                prediction_bars=prediction_bars,
                k_neighbors=k_neighbors,
            )

            # Standalone implementation
            self.models["standalone"] = StandaloneLorentzian(
                lookback_bars=lookback,
                prediction_bars=prediction_bars,
                k_neighbors=k_neighbors,
            )

            # Analysis implementation
            self.models["analysis"] = AnalysisLorentzian(
                lookback_bars=lookback,
                prediction_bars=prediction_bars,
                k_neighbors=k_neighbors,
            )

            # Modern PyTorch implementation
            self.models["modern"] = ModernLorentzian()

            print("All models initialized successfully")

        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    async def fetch_data(self) -> pd.DataFrame:
        """Fetch data for comparison using the backtester"""
        print(f"Fetching data for {self.config.market.symbol}...")
        data = await self.backtester.fetch_data(
            self.config.market.timeframe, lookback_days=30
        )
        print(f"Fetched {len(data)} candles")
        return data

    def train_and_predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """Train a model and generate predictions"""
        print(f"Training and predicting with {model_name}...")
        model = self.models[model_name]

        # Prepare features
        features = prepare_features(data)
        prices = torch.tensor(data["close"].values, dtype=torch.float32)

        # Fit model
        start_time = time.time()
        model.fit(features, prices)

        # Generate predictions
        predictions = model.predict(features)
        end_time = time.time()

        print(
            f"Training and prediction completed in {end_time - start_time:.2f} seconds"
        )

        # Convert to numpy array if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        return predictions

    async def run_comparison(self):
        """Run backtest comparison for all models"""
        try:
            # Fetch data
            data = await self.fetch_data()

            # Generate predictions for each model and run backtest
            for model_name in self.models.keys():
                print(f"\n===== Testing {model_name} =====")

                # Train and predict
                predictions = self.train_and_predict(model_name, data)

                # Run backtest
                report = await self.backtester.run_backtest(predictions, data)

                # Store results
                self.results[model_name] = report
                print(f"Completed backtest for {model_name}")

            # Generate comparison report
            self.generate_comparison_report()

        except Exception as e:
            print(f"Error running comparison: {str(e)}")
            raise

    def generate_comparison_report(self):
        """Generate a detailed comparison report of all models"""
        if not self.results:
            print("No results to compare")
            return

        print("\n===== MODEL COMPARISON REPORT =====")

        # Create a comparison DataFrame
        metrics = [
            "total_return_pct",
            "annualized_return_pct",
            "total_trades",
            "win_rate_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "total_fees",
            "total_funding",
        ]

        comparison_data = {}
        for model_name, report in self.results.items():
            model_metrics = {metric: report.get(metric, 0) for metric in metrics}
            comparison_data[model_name] = model_metrics

        df = pd.DataFrame(comparison_data)

        # Print the comparison table
        print(df.T)

        # Save detailed report to CSV
        os.makedirs("results", exist_ok=True)
        df.T.to_csv(f"results/{self.config.detailed_report_filename}")
        print(
            f"Detailed report saved to results/{self.config.detailed_report_filename}"
        )

        # Generate comparative visualizations
        self.plot_comparison("total_return_pct", "Total Return (%)")
        self.plot_comparison("win_rate_pct", "Win Rate (%)")
        self.plot_comparison("sharpe_ratio", "Sharpe Ratio")
        self.plot_comparison(
            "max_drawdown_pct", "Maximum Drawdown (%)", lower_is_better=True
        )

    def plot_comparison(self, metric: str, title: str, lower_is_better: bool = False):
        """Generate bar chart comparing a specific metric across models"""
        values = []
        model_names = []

        for model_name, report in self.results.items():
            values.append(report.get(metric, 0))
            model_names.append(model_name)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, values)

        # Color best performer
        if values:
            idx = np.argmin(values) if lower_is_better else np.argmax(values)
            bars[idx].set_color("green")

        plt.title(f"Comparison of {title}")
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"results/comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Comparison chart saved to {filename}")

    @staticmethod
    def get_config_by_name(config_name: str) -> ComparisonConfig:
        """Get a configuration by name"""
        config_map = {
            "default": default_config,
            "binance_btc_futures": binance_btc_futures_config,
            "bitget_sol_futures": bitget_sol_futures_config,
        }

        if config_name in config_map:
            return config_map[config_name]

        print(f"Config '{config_name}' not found, using default config")
        return default_config


async def main():
    """Main function for running comparison with command-line options"""
    parser = argparse.ArgumentParser(
        description="Compare Lorentzian models with advanced backtesting"
    )
    parser.add_argument(
        "--config",
        default="default",
        choices=["default", "binance_btc_futures", "bitget_sol_futures"],
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Number of days of historical data to fetch",
    )
    parser.add_argument("--symbol", type=str, help="Symbol to test (e.g., SOL/USDT)")
    parser.add_argument(
        "--timeframe", type=str, help="Timeframe to test (e.g., 5m, 1h, 4h)"
    )
    parser.add_argument("--balance", type=float, help="Initial balance for backtest")
    parser.add_argument(
        "--show-plots", action="store_true", help="Show plots during backtest"
    )

    args = parser.parse_args()

    # Get config by name
    config = ModelComparison.get_config_by_name(args.config)

    # Override config values with command-line arguments
    if args.symbol:
        config.market.symbol = args.symbol
    if args.timeframe:
        config.market.timeframe = args.timeframe
    if args.balance:
        config.backtest.initial_balance = args.balance
    if args.show_plots:
        config.show_plots = True

    # Run comparison
    comparison = ModelComparison(config)
    await comparison.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())
