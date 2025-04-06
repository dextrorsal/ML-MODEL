"""
Simple Lorentzian Implementation Test

This script tests each of the Lorentzian implementations individually
with synthetic data to verify basic functionality.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add path to root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import implementation classes
from src.comparison.your_implementation import LorentzianANN as YourImplementation
from src.comparison.standalone_implementation import (
    LorentzianANN as StandaloneLorentzian,
)
from src.comparison.analysis_implementation import LorentzianANN as AnalysisLorentzian
from src.comparison.modern_pytorch_implementation import ModernLorentzian


def generate_synthetic_data(length=1000):
    """Generate synthetic OHLCV data for testing"""
    print(f"Generating {length} synthetic candles for testing...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate fake timestamps
    timestamps = pd.date_range(start="2023-01-01", periods=length, freq="1h")

    # Generate price data with a random walk
    close = np.zeros(length)
    close[0] = 100  # Starting price

    # Add trend and noise
    for i in range(1, length):
        # Add momentum and mean-reversion components
        momentum = 0.1 * (close[i - 1] - close[max(0, i - 5)]) / close[max(0, i - 5)]
        mean_reversion = -0.05 * (close[i - 1] - close[0]) / close[0]

        # Add regime changes occasionally
        if i % 200 == 0:
            regime_change = np.random.choice([-0.1, 0.1])
        else:
            regime_change = 0

        # Random component
        random_change = np.random.normal(0, 0.015)

        # Combine components
        change = momentum + mean_reversion + random_change + regime_change
        close[i] = close[i - 1] * (1 + change)

    # Generate OHLC data around the close price
    high = close * (1 + np.abs(np.random.normal(0, 0.01, length)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, length)))
    open_price = low + (high - low) * np.random.random(length)

    # Generate volume data
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

    return df


def prepare_features(df):
    """Prepare features for model training and prediction"""
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

    return features, features_df.index


def evaluate_signals(signals, prices, fee_rate=0.001):
    """Calculate performance metrics for signals"""
    # Ensure signals and prices are numpy arrays
    if isinstance(signals, torch.Tensor):
        signals = signals.cpu().numpy()
    if isinstance(prices, torch.Tensor):
        prices = prices.cpu().numpy()

    # Make sure lengths match
    min_len = min(len(signals), len(prices))
    signals = signals[:min_len]
    prices = prices[:min_len]

    # Calculate returns
    returns = np.diff(prices) / prices[:-1]

    # Shift signals by 1 to avoid look-ahead bias
    shifted_signals = np.roll(signals, 1)
    shifted_signals[0] = 0

    # Calculate position returns
    position_returns = returns * shifted_signals[:-1]

    # Add trading fees (simplistic approach)
    # Calculate signal changes (only when position changes)
    signal_changes = np.diff(np.where(shifted_signals != 0, shifted_signals, 0))

    # Make sure these arrays have the same length
    if len(position_returns) > len(signal_changes):
        position_returns = position_returns[: len(signal_changes)]
    elif len(position_returns) < len(signal_changes):
        signal_changes = signal_changes[: len(position_returns)]

    fees = np.abs(signal_changes) * fee_rate
    net_position_returns = position_returns - fees

    # Calculate metrics
    total_return = np.sum(net_position_returns) * 100  # percentage
    win_rate = (
        np.mean(net_position_returns > 0) * 100 if len(net_position_returns) > 0 else 0
    )
    profit_factor = (
        -np.sum(net_position_returns[net_position_returns > 0])
        / np.sum(net_position_returns[net_position_returns < 0])
        if np.sum(net_position_returns[net_position_returns < 0]) < 0
        else 0
    )

    # Create a summary dict
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    neutral_signals = np.sum(signals == 0)
    signal_activity = ((buy_signals + sell_signals) / len(signals)) * 100

    # Count trades (each change in position)
    trades = np.sum(np.abs(signal_changes) > 0)

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": trades,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "neutral_signals": neutral_signals,
        "signal_activity": signal_activity,
    }


def plot_signals(data_index, prices, signals, title, filename=None):
    """Plot price chart with signals"""
    if isinstance(signals, torch.Tensor):
        signals = signals.cpu().numpy()
    if isinstance(prices, torch.Tensor):
        prices = prices.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(data_index, prices, label="Price")

    # Plot buy signals
    buy_indices = data_index[signals == 1]
    buy_prices = prices[signals == 1]
    plt.scatter(buy_indices, buy_prices, color="green", marker="^", label="Buy Signal")

    # Plot sell signals
    sell_indices = data_index[signals == -1]
    sell_prices = prices[signals == -1]
    plt.scatter(sell_indices, sell_prices, color="red", marker="v", label="Sell Signal")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def test_implementation(
    model_cls, model_name, lookback=50, prediction_bars=4, k_neighbors=20
):
    """Test an implementation with synthetic data"""
    print(f"\n===== Testing {model_name} =====")

    # Generate data
    data = generate_synthetic_data(1000)

    # Prepare features
    features, feature_index = prepare_features(data)
    prices = torch.tensor(data["close"].values, dtype=torch.float32)

    # Initialize model
    try:
        if model_name == "modern":
            # The modern implementation needs the input size
            input_size = features.shape[1] if features.shape[0] > 0 else 10
            model = model_cls(input_size=input_size)
        else:
            model = model_cls(
                lookback_bars=lookback,
                prediction_bars=prediction_bars,
                k_neighbors=k_neighbors,
            )

        # Fit model
        import time

        start_time = time.time()
        model.fit(features, prices)

        # Generate predictions
        predictions = model.predict(features)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Training and prediction completed in {duration:.2f} seconds")

        # Evaluate performance
        performance = evaluate_signals(
            predictions, prices[len(prices) - len(predictions) :]
        )
        print(f"Performance metrics:")
        for metric, value in performance.items():
            print(f"  {metric}: {value:.2f}")

        # Plot signals
        os.makedirs("results", exist_ok=True)
        plot_signals(
            feature_index,
            data.loc[feature_index, "close"].values,
            predictions,
            f"{model_name} Signals",
            f"results/{model_name}_signals.png",
        )

        return {
            "model_name": model_name,
            "predictions": predictions,
            "performance": performance,
            "execution_time": duration,
        }

    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"model_name": model_name, "error": str(e)}


def run_all_tests():
    """Run tests for all implementations"""
    print("Starting Lorentzian implementation tests...")

    # Define implementations to test
    implementations = [
        (YourImplementation, "your_implementation"),
        (StandaloneLorentzian, "standalone"),
        (AnalysisLorentzian, "analysis"),
        (ModernLorentzian, "modern"),
    ]

    results = []

    # Test each implementation
    for model_cls, model_name in implementations:
        result = test_implementation(model_cls, model_name)
        results.append(result)

    # Compare results
    compare_results(results)


def compare_results(results):
    """Compare results between implementations"""
    # Create comparison DataFrame
    comparison_data = {}
    for result in results:
        if "error" in result:
            continue

        model_name = result["model_name"]
        performance = result["performance"]
        comparison_data[model_name] = {
            "total_return": performance["total_return"],
            "win_rate": performance["win_rate"],
            "profit_factor": performance["profit_factor"],
            "trades": performance["trades"],
            "signal_activity": performance["signal_activity"],
            "execution_time": result["execution_time"],
        }

    if not comparison_data:
        print("No valid results to compare")
        return

    df = pd.DataFrame(comparison_data)

    # Print comparison
    print("\n===== IMPLEMENTATION COMPARISON =====")
    print(df)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/implementation_comparison.csv")
    print("Comparison saved to results/implementation_comparison.csv")

    # Plot comparative bar charts
    metrics = ["total_return", "win_rate", "profit_factor", "signal_activity"]

    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(df.columns, df.loc[metric])
        plt.title(f"Comparison of {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.savefig("results/metrics_comparison.png")
    print("Comparison charts saved to results/metrics_comparison.png")

    # Plot execution time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df.columns, df.loc["execution_time"])
    plt.title("Execution Time Comparison")
    plt.ylabel("Seconds")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/execution_time_comparison.png")
    print("Execution time comparison saved to results/execution_time_comparison.png")


if __name__ == "__main__":
    run_all_tests()
