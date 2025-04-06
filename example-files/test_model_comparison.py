"""
Lorentzian Model Comparison Test

This script compares the performance of all three Lorentzian implementations:
1. Modern PyTorch Version (models/primary/lorentzian_classifier.py)
2. Standalone Version (lorentzian_classifier.py)
3. Analysis Version (analyze_lorentzian_ann.py)

Each model will be tested on the same dataset with the same metrics for fair comparison.
"""

import sys

sys.path.append(".")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ccxt
import time
import matplotlib.pyplot as plt
import torch.optim as optim

# Import implementations
from strategies.LorentzianStrategy.models.primary.lorentzian_classifier import (
    ModernLorentzian,
)
from strategies.LorentzianStrategy.lorentzian_classifier import (
    LorentzianANN as StandaloneLorentzian,
)
from analyze_lorentzian_ann import LorentzianANN as AnalysisLorentzian


def fetch_training_data(symbol="SOL/USDT", timeframe="5m", limit=1000):
    """Fetch recent price data for testing"""
    print(f"Fetching {limit} {timeframe} candles for {symbol}...")

    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def prepare_features(df):
    """Prepare basic features for testing"""
    # Calculate basic features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()

    # Price changes over different periods
    for period in [5, 10, 20]:
        df[f"price_change_{period}"] = df["close"].pct_change(period)

    # Volatility
    df["volatility"] = df["returns"].rolling(20).std()

    # Volume features
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_std"] = df["volume"].rolling(20).std()

    # Create feature matrix
    feature_columns = [
        "returns",
        "log_returns",
        "price_change_5",
        "price_change_10",
        "price_change_20",
        "volatility",
        "volume_ma",
        "volume_std",
    ]

    # Drop NaN values
    df = df.dropna()

    return df[feature_columns].values, df["close"].values


def test_modern_lorentzian(features, prices):
    """Test the Modern PyTorch implementation."""
    print("\nTesting Modern PyTorch Implementation...")

    # Initialize model
    input_size = features.shape[1]
    hidden_size = 64

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to GPU
    model = ModernLorentzian(input_size=input_size, hidden_size=hidden_size).to(device)

    # Prepare training data and move to GPU
    X = torch.FloatTensor(features).to(device)

    # Add close price as last column if not present
    if prices is not None:
        prices_tensor = torch.FloatTensor(prices).to(device)
        X = torch.cat([X, prices_tensor.unsqueeze(1)], dim=1)

    # Generate predictions
    print("Generating predictions with hybrid RSI-WMA-kNN system...")
    start_time = time.time()

    with torch.no_grad():
        predictions = model.generate_signals(X)

    # Convert predictions to numpy and pad
    padded_predictions = np.zeros(len(features))
    padded_predictions[: len(predictions)] = predictions.cpu().numpy()

    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")

    return padded_predictions


def test_standalone_lorentzian(features, prices):
    """Test the Standalone implementation."""
    print("\nTesting Standalone Implementation...")
    start_time = time.time()

    # Initialize model
    model = StandaloneLorentzian(lookback_bars=50, prediction_bars=4, k_neighbors=20)

    # Convert inputs to GPU tensors if available
    if torch.cuda.is_available():
        features = torch.FloatTensor(features).cuda()
        prices = torch.FloatTensor(prices).cuda()

    # Train and predict
    model.fit(features, prices)
    predictions = model.predict(features)

    # Move predictions to CPU if they're on GPU
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    end_time = time.time()
    print(f"Training and prediction completed in {end_time - start_time:.2f} seconds")

    return predictions


def test_analysis_lorentzian(features, prices):
    """Test the Analysis implementation."""
    print("\nTesting Analysis Implementation...")
    start_time = time.time()

    # Initialize model
    model = AnalysisLorentzian(lookback_bars=50, prediction_bars=4, k_neighbors=20)

    # Convert inputs to GPU tensors if available
    if torch.cuda.is_available():
        features = torch.FloatTensor(features).cuda()
        prices = torch.FloatTensor(prices).cuda()

    # Train and predict
    model.fit(features, prices)
    predictions = model.predict(features)

    # Move predictions to CPU if they're on GPU
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    end_time = time.time()
    print(f"Training and prediction completed in {end_time - start_time:.2f} seconds")

    return predictions


def plot_comparison(df, modern_preds, standalone_preds, analysis_preds):
    """Plot the results from all three implementations"""
    plt.figure(figsize=(15, 10))

    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["close"], label="Price", color="black", alpha=0.5)
    plt.title("Price and Predictions Comparison")
    plt.legend()
    plt.grid(True)

    # Ensure all predictions have the same length
    n = min(len(modern_preds), len(standalone_preds), len(analysis_preds))
    x_axis = df.index[:n]

    # Plot predictions
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, modern_preds[:n], label="Modern", alpha=0.7)
    plt.plot(x_axis, standalone_preds[:n], label="Standalone", alpha=0.7)
    plt.plot(x_axis, analysis_preds[:n], label="Analysis", alpha=0.7)
    plt.title("Model Predictions Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()


def print_prediction_stats(name, predictions):
    """Print statistics about the predictions"""
    print(f"\n{name} Prediction Stats:")
    print(f"Mean: {np.mean(predictions):.4f}")
    print(f"Std: {np.std(predictions):.4f}")
    print(f"Min: {np.min(predictions):.4f}")
    print(f"Max: {np.max(predictions):.4f}")
    print(f"Unique values: {len(np.unique(predictions))}")


def calculate_metrics(predictions, prices):
    """Calculate trading metrics."""
    returns = np.diff(prices) / prices[:-1]

    # Ensure predictions array matches returns array length
    predictions = predictions[:-1]  # Trim predictions to match returns length

    # Calculate win rate
    trades = predictions[predictions != 0]
    trade_returns = returns[predictions != 0]
    wins = np.sum((trade_returns > 0) & (trades == 1)) + np.sum(
        (trade_returns < 0) & (trades == -1)
    )
    win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0

    # Calculate signal distribution
    buy_signals = np.sum(predictions == 1)
    sell_signals = np.sum(predictions == -1)
    hold_signals = np.sum(predictions == 0)
    signal_activity = ((buy_signals + sell_signals) / len(predictions)) * 100

    # Calculate profit metrics
    position_returns = returns * predictions
    total_profit = np.sum(position_returns) * 100

    # Calculate drawdown
    cumulative_returns = np.cumsum(position_returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = rolling_max - cumulative_returns
    max_drawdown = np.max(drawdowns) * 100

    # Calculate average profit per trade
    avg_profit = (total_profit / len(trades)) if len(trades) > 0 else 0

    # Calculate Sharpe ratio
    sharpe_ratio = (
        np.mean(position_returns) / np.std(position_returns)
        if len(position_returns) > 0 and np.std(position_returns) != 0
        else 0
    )

    return {
        "win_rate": win_rate,
        "total_trades": len(trades),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "signal_activity": signal_activity,
        "total_profit": total_profit,
        "max_drawdown": max_drawdown,
        "avg_profit": avg_profit,
        "sharpe_ratio": sharpe_ratio,
    }


def plot_metrics(modern_metrics, standalone_metrics, analysis_metrics):
    """Plot performance metrics comparison."""
    metrics = [
        "win_rate",
        "total_profit",
        "max_drawdown",
        "sharpe_ratio",
        "signal_activity",
    ]
    implementations = ["Modern", "Standalone", "Analysis"]

    values = {
        "Modern": [
            modern_metrics["win_rate"],
            modern_metrics["total_profit"],
            modern_metrics["max_drawdown"],
            modern_metrics["sharpe_ratio"],
            modern_metrics["signal_activity"],
        ],
        "Standalone": [
            standalone_metrics["win_rate"],
            standalone_metrics["total_profit"],
            standalone_metrics["max_drawdown"],
            standalone_metrics["sharpe_ratio"],
            standalone_metrics["signal_activity"],
        ],
        "Analysis": [
            analysis_metrics["win_rate"],
            analysis_metrics["total_profit"],
            analysis_metrics["max_drawdown"],
            analysis_metrics["sharpe_ratio"],
            analysis_metrics["signal_activity"],
        ],
    }

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, impl in enumerate(implementations):
        ax.bar(x + i * width, values[impl], width, label=impl)

    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png")
    plt.close()


def main():
    print("Starting Lorentzian Model Comparison Test...")

    # Fetch data
    df = fetch_training_data()
    features, prices = prepare_features(df)

    # Test each implementation
    modern_predictions = test_modern_lorentzian(features, prices)
    standalone_predictions = test_standalone_lorentzian(features, prices)
    analysis_predictions = test_analysis_lorentzian(features, prices)

    # Calculate metrics
    modern_metrics = calculate_metrics(modern_predictions, prices)
    standalone_metrics = calculate_metrics(standalone_predictions, prices)
    analysis_metrics = calculate_metrics(analysis_predictions, prices)

    # Print metrics
    print("\nModern Trading Metrics:")
    print(f"Win Rate: {modern_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {modern_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {modern_metrics['buy_signals']}")
    print(f"  - Sell Signals: {modern_metrics['sell_signals']}")
    print(f"  - Hold Signals: {modern_metrics['hold_signals']}")
    print(f"Signal Activity: {modern_metrics['signal_activity']:.2f}%")
    print(f"Total Profit: {modern_metrics['total_profit']:.2f}%")
    print(f"Max Drawdown: {modern_metrics['max_drawdown']:.2f}%")
    print(f"Avg Profit per Trade: {modern_metrics['avg_profit']:.4f}")
    print(f"Sharpe Ratio: {modern_metrics['sharpe_ratio']:.4f}\n")

    print("Standalone Trading Metrics:")
    print(f"Win Rate: {standalone_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {standalone_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {standalone_metrics['buy_signals']}")
    print(f"  - Sell Signals: {standalone_metrics['sell_signals']}")
    print(f"  - Hold Signals: {standalone_metrics['hold_signals']}")
    print(f"Signal Activity: {standalone_metrics['signal_activity']:.2f}%")
    print(f"Total Profit: {standalone_metrics['total_profit']:.2f}%")
    print(f"Max Drawdown: {standalone_metrics['max_drawdown']:.2f}%")
    print(f"Avg Profit per Trade: {standalone_metrics['avg_profit']:.4f}")
    print(f"Sharpe Ratio: {standalone_metrics['sharpe_ratio']:.4f}\n")

    print("Analysis Trading Metrics:")
    print(f"Win Rate: {analysis_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {analysis_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {analysis_metrics['buy_signals']}")
    print(f"  - Sell Signals: {analysis_metrics['sell_signals']}")
    print(f"  - Hold Signals: {analysis_metrics['hold_signals']}")
    print(f"Signal Activity: {analysis_metrics['signal_activity']:.2f}%")
    print(f"Total Profit: {analysis_metrics['total_profit']:.2f}%")
    print(f"Max Drawdown: {analysis_metrics['max_drawdown']:.2f}%")
    print(f"Avg Profit per Trade: {analysis_metrics['avg_profit']:.4f}")
    print(f"Sharpe Ratio: {analysis_metrics['sharpe_ratio']:.4f}\n")

    # Plot metrics
    plot_metrics(modern_metrics, standalone_metrics, analysis_metrics)
    print("Test completed! Results saved to model_performance_comparison.png")


if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    main()
