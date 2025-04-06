"""
Lorentzian Model Comparison Test

This script compares the performance of all four Lorentzian implementations:
1. Your Implementation (src/models/strategy/lorentzian_classifier.py)
2. Standalone Version (example-files/strategies/LorentzianStrategy/lorentzian_classifier.py)
3. Analysis Version (example-files/analyze_lorentzian_ann.py)
4. Modern PyTorch Version (example-files/strategies/LorentzianStrategy/models/primary/lorentzian_classifier.py)

Each model will be tested on the same dataset with the same metrics for fair comparison.
"""

import sys
import os

# Add paths for easier imports
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import pandas as pd
import numpy as np
import torch
import ccxt
import time
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Import implementations with relative paths
from your_implementation import LorentzianANN as YourImplementation
from standalone_implementation import LorentzianANN as StandaloneLorentzian
from analysis_implementation import LorentzianANN as AnalysisLorentzian
from modern_pytorch_implementation import ModernLorentzian

# ==========================================
# CONFIGURATION SECTION
# ==========================================

# Default configuration (can be overridden with command-line arguments or config file)
DEFAULT_CONFIG = {
    # Exchange settings
    "exchange": "bitget",  # Options: binance, bitget, etc.
    "market_type": "spot",  # Options: spot, futures
    # Asset settings
    "symbol": "SOL/USDT",  # Options: BTC/USDT, ETH/USDT, SOL/USDT
    "timeframe": "5m",  # Options: 1m, 5m, 15m, 1h, 4h, 1d
    "data_limit": 1000,  # Number of candles to fetch
    # Order settings
    "order_type": "GTC",  # Options: FOK (Fill-or-Kill), GTC (Good-Till-Canceled), IOC (Immediate-or-Cancel)
    # Portfolio settings
    "starting_balance": 10000,  # Starting balance in USDT
    "position_size": 0.1,  # Percentage of balance to use per trade (0.1 = 10%)
    "leverage": 1,  # Leverage for futures trading (1 = no leverage)
    # Backtesting settings
    "fee_rate": 0.001,  # Trading fee as a percentage (0.001 = 0.1%)
    "slippage": 0.0005,  # Slippage as a percentage (0.0005 = 0.05%)
}


def load_config(config_path):
    """Load configuration from a JSON file."""
    # Handle relative paths from either src/comparison or project root
    if not os.path.isabs(config_path):
        # Try relative to script location first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, config_path)

        # If not found, try relative to project root
        if not os.path.exists(full_path):
            project_root = os.path.abspath(os.path.join(script_dir, "../.."))
            full_path = os.path.join(project_root, config_path)

        config_path = full_path

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Looked in:")
        print(f"- {os.path.abspath(config_path)}")
        print(
            f"- {os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)}"
        )
        print(
            f"- {os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), config_path)}"
        )
        return {}


def save_config(config, config_path):
    """Save configuration to a JSON file."""
    # Handle relative paths
    if not os.path.isabs(config_path):
        # Save relative to script location by default
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # If path seems to reference project root (contains 'config_samples' at root level)
        if config_path.startswith("config_samples/") or config_path == "config_samples":
            project_root = os.path.abspath(os.path.join(script_dir, "../.."))
            full_path = os.path.join(project_root, config_path)
        else:
            full_path = os.path.join(script_dir, config_path)

        config_path = full_path

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Lorentzian Model Comparison Test")

    # Config file options
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument(
        "--save_config", type=str, help="Save current configuration to JSON file"
    )

    # Exchange settings
    parser.add_argument(
        "--exchange", type=str, choices=["binance", "bitget"], help="Exchange to use"
    )
    parser.add_argument(
        "--market_type", type=str, choices=["spot", "futures"], help="Market type"
    )

    # Asset settings
    parser.add_argument(
        "--symbol",
        type=str,
        choices=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Trading pair",
    )
    parser.add_argument("--timeframe", type=str, help="Chart timeframe")
    parser.add_argument("--data_limit", type=int, help="Number of candles to fetch")

    # Order settings
    parser.add_argument(
        "--order_type", type=str, choices=["FOK", "GTC", "IOC"], help="Order type"
    )

    # Portfolio settings
    parser.add_argument(
        "--starting_balance", type=float, help="Starting balance in USDT"
    )
    parser.add_argument(
        "--position_size", type=float, help="Percentage of balance per trade"
    )
    parser.add_argument("--leverage", type=int, help="Leverage for futures trading")

    # Backtesting settings
    parser.add_argument("--fee_rate", type=float, help="Trading fee percentage")
    parser.add_argument("--slippage", type=float, help="Slippage percentage")

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output files",
    )

    args = parser.parse_args()

    # Start with default configuration
    config = DEFAULT_CONFIG.copy()

    # Load configuration from file if specified
    if args.config:
        file_config = load_config(args.config)
        config.update(file_config)

    # Override with command-line arguments if provided
    for key, value in vars(args).items():
        if key not in ["config", "save_config", "output_dir"] and value is not None:
            config[key] = value

    # Add output directory to config only if explicitly provided
    if args.output_dir != "results" or "output_dir" not in config:
        config["output_dir"] = args.output_dir

    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)

    return config


def fetch_training_data(config):
    """Fetch recent price data for testing"""
    print(
        f"Fetching {config['data_limit']} {config['timeframe']} candles for {config['symbol']} from {config['exchange']}..."
    )

    try:
        # Initialize exchange
        if config["exchange"].lower() == "bitget":
            exchange_class = ccxt.bitget
        else:
            exchange_class = ccxt.binance

        # Configure exchange options
        exchange_options = {
            "enableRateLimit": True,  # respect API rate limits
        }

        # For futures trading
        if config["market_type"].lower() == "futures":
            if config["exchange"].lower() == "bitget":
                exchange_options["options"] = {
                    "defaultType": "swap",  # USDT-M futures on Bitget
                }
            else:
                exchange_options["options"] = {
                    "defaultType": "future",  # USDT-M futures on Binance
                }

        # Initialize exchange
        exchange = exchange_class(exchange_options)

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            config["symbol"], config["timeframe"], limit=config["data_limit"]
        )

        # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        print(f"Successfully fetched data with shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_data(config["data_limit"])


def generate_synthetic_data(length=1000):
    """Generate synthetic OHLCV data for testing (fallback if API fails)."""
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
        momentum = 0.1 * (close[i - 1] - close[max(0, i - 5)]) / close[max(0, i - 5)]
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


def test_your_implementation(features, prices):
    """Test your new implementation."""
    print("\nTesting Your Implementation...")
    start_time = time.time()

    # Initialize model
    model = YourImplementation(lookback_bars=50, prediction_bars=4, k_neighbors=20)

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


def plot_metrics(
    your_metrics,
    modern_metrics,
    standalone_metrics,
    analysis_metrics,
    output_dir="results",
):
    """Plot performance metrics comparison."""
    metrics = [
        "win_rate",
        "total_return_pct",
        "max_drawdown",
        "sharpe_ratio",
        "signal_activity",
    ]
    implementations = ["Your Impl", "Modern", "Standalone", "Analysis"]

    values = {
        "Your Impl": [
            your_metrics["win_rate"],
            your_metrics["total_return_pct"],
            your_metrics["max_drawdown"],
            your_metrics["sharpe_ratio"],
            your_metrics["signal_activity"],
        ],
        "Modern": [
            modern_metrics["win_rate"],
            modern_metrics["total_return_pct"],
            modern_metrics["max_drawdown"],
            modern_metrics["sharpe_ratio"],
            modern_metrics["signal_activity"],
        ],
        "Standalone": [
            standalone_metrics["win_rate"],
            standalone_metrics["total_return_pct"],
            standalone_metrics["max_drawdown"],
            standalone_metrics["sharpe_ratio"],
            standalone_metrics["signal_activity"],
        ],
        "Analysis": [
            analysis_metrics["win_rate"],
            analysis_metrics["total_return_pct"],
            analysis_metrics["max_drawdown"],
            analysis_metrics["sharpe_ratio"],
            analysis_metrics["signal_activity"],
        ],
    }

    x = np.arange(len(metrics))
    width = 0.2  # Adjusted width for 4 bars

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, impl in enumerate(implementations):
        ax.bar(x + i * width, values[impl], width, label=impl)

    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison")
    ax.set_xticks(x + 1.5 * width)  # Adjusted position for 4 bars
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"))
    plt.close()


def plot_comparison(
    df, your_preds, modern_preds, standalone_preds, analysis_preds, output_dir="results"
):
    """Plot the results from all four implementations"""
    plt.figure(figsize=(15, 10))

    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["close"], color="black", alpha=0.5, label="Price")
    plt.title("Price Chart")
    plt.legend()

    # Plot signals
    plt.subplot(2, 1, 2)

    # Ensure all predictions have the same length
    n = min(
        len(your_preds), len(modern_preds), len(standalone_preds), len(analysis_preds)
    )
    x_axis = df.index[:n]

    plt.plot(x_axis, your_preds[:n], label="Your Implementation", alpha=0.7)
    plt.plot(x_axis, modern_preds[:n], label="Modern PyTorch", alpha=0.7)
    plt.plot(x_axis, standalone_preds[:n], label="Standalone", alpha=0.7)
    plt.plot(x_axis, analysis_preds[:n], label="Analysis", alpha=0.7)

    plt.title("Signal Comparison")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()


def plot_equity_curves(
    your_metrics,
    modern_metrics,
    standalone_metrics,
    analysis_metrics,
    output_dir="results",
):
    """Plot equity curves for all implementations."""
    plt.figure(figsize=(12, 6))

    # Plot equity curves
    plt.plot(your_metrics["balance_history"], label="Your Implementation")
    plt.plot(modern_metrics["balance_history"], label="Modern")
    plt.plot(standalone_metrics["balance_history"], label="Standalone")
    plt.plot(analysis_metrics["balance_history"], label="Analysis")

    # Add initial balance reference line
    plt.axhline(
        y=your_metrics["initial_balance"], color="gray", linestyle="--", alpha=0.5
    )

    plt.title("Equity Curves Comparison")
    plt.xlabel("Trade Number")
    plt.ylabel("Account Balance (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "equity_curves.png"))
    plt.close()


def print_prediction_stats(name, predictions):
    """Print statistics about the predictions"""
    print(f"\n{name} Prediction Stats:")
    print(f"Mean: {np.mean(predictions):.4f}")
    print(f"Std: {np.std(predictions):.4f}")
    print(f"Min: {np.min(predictions):.4f}")
    print(f"Max: {np.max(predictions):.4f}")
    print(f"Unique values: {len(np.unique(predictions))}")


def calculate_metrics(predictions, prices, config):
    """Calculate trading metrics with portfolio balance tracking."""
    returns = np.diff(prices) / prices[:-1]

    # Ensure predictions array matches returns array length
    predictions = predictions[:-1]  # Trim predictions to match returns length

    # Initialize portfolio tracking
    balance = config["starting_balance"]
    balance_history = [balance]
    position_size_usd = balance * config["position_size"]
    position = 0  # Current position: 0=none, 1=long, -1=short
    entry_price = 0

    # Calculate fee and slippage impact
    fee_rate = config["fee_rate"]
    slippage = config["slippage"]
    total_fee_paid = 0

    # Track trades
    trades = []
    wins = 0
    losses = 0

    # Process each signal
    for i in range(len(predictions)):
        current_price = prices[i]
        signal = predictions[i]

        # Check if we need to close existing position
        if position != 0 and (
            signal == 0
            or (position == 1 and signal == -1)
            or (position == -1 and signal == 1)
        ):
            # Calculate profit/loss
            position_value = position * position_size_usd
            if position == 1:  # Long position
                exit_price = current_price * (
                    1 - slippage
                )  # Sell at slightly lower price due to slippage
                pnl = (exit_price / entry_price - 1) * position_value
            else:  # Short position
                exit_price = current_price * (
                    1 + slippage
                )  # Buy at slightly higher price due to slippage
                pnl = (1 - exit_price / entry_price) * position_value

            # Subtract fees
            fee = abs(position_value) * fee_rate
            total_fee_paid += fee
            pnl -= fee

            # Update balance
            balance += pnl

            # Record trade
            trade = {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position": position,
                "pnl": pnl,
                "pnl_pct": pnl / position_size_usd * 100,
                "fee": fee,
            }
            trades.append(trade)

            # Update win/loss count
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            # Close position
            position = 0

        # Open new position if we have a signal and no existing position (or need to flip)
        if signal != 0 and (
            position == 0
            or (position == 1 and signal == -1)
            or (position == -1 and signal == 1)
        ):
            # Update position size based on current balance
            position_size_usd = balance * config["position_size"]

            # Apply leverage if using futures
            if config["market_type"].lower() == "futures":
                position_size_usd *= config["leverage"]

            # Set entry price with slippage
            if signal == 1:  # Long position
                entry_price = current_price * (
                    1 + slippage
                )  # Buy at slightly higher price
            else:  # Short position
                entry_price = current_price * (
                    1 - slippage
                )  # Sell at slightly lower price

            # Pay entry fee
            fee = position_size_usd * fee_rate
            total_fee_paid += fee
            balance -= fee

            # Set position
            position = signal

        # Record balance at each step
        balance_history.append(balance)

    # Calculate basic metrics
    win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0

    # Calculate signal distribution
    buy_signals = np.sum(predictions == 1)
    sell_signals = np.sum(predictions == -1)
    hold_signals = np.sum(predictions == 0)
    signal_activity = ((buy_signals + sell_signals) / len(predictions)) * 100

    # Calculate advanced portfolio metrics
    initial_balance = config["starting_balance"]
    final_balance = balance
    total_return_pct = ((final_balance / initial_balance) - 1) * 100

    # Calculate drawdown
    balance_arr = np.array(balance_history)
    peak = np.maximum.accumulate(balance_arr)
    drawdown = (peak - balance_arr) / peak * 100
    max_drawdown = np.max(drawdown)

    # Calculate average profit per trade
    if len(trades) > 0:
        avg_profit = sum(t["pnl"] for t in trades) / len(trades)
        avg_profit_pct = sum(t["pnl_pct"] for t in trades) / len(trades)
    else:
        avg_profit = 0
        avg_profit_pct = 0

    # Calculate Sharpe ratio if we have balance history
    if len(balance_history) > 1:
        daily_returns = np.diff(balance_history) / balance_history[:-1]
    sharpe_ratio = (
            (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
            if np.std(daily_returns) != 0
        else 0
    )
    else:
        sharpe_ratio = 0

    return {
        "win_rate": win_rate,
        "total_trades": len(trades),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "signal_activity": signal_activity,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "avg_profit": avg_profit,
        "avg_profit_pct": avg_profit_pct,
        "sharpe_ratio": sharpe_ratio,
        "total_fee_paid": total_fee_paid,
        "balance_history": balance_history,
        "trades": trades,
    }


def display_trade_statistics(metrics, name):
    """Display detailed trade statistics for paper trading simulation."""
    trades = metrics["trades"]
    if not trades:
        print(f"\n{name} - No trades executed")
        return

    # Extract trade data
    profits = [t["pnl"] for t in trades]
    profit_pcts = [t["pnl_pct"] for t in trades]

    # Calculate streaks
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for i, trade in enumerate(trades):
        if trade["pnl"] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1

        if current_streak > max_win_streak:
            max_win_streak = current_streak
        if current_streak < max_loss_streak:
            max_loss_streak = current_streak

    # Calculate other statistics
    largest_win = max(profits) if profits and max(profits) > 0 else 0
    largest_loss = min(profits) if profits and min(profits) < 0 else 0
    largest_win_pct = max(profit_pcts) if profit_pcts and max(profit_pcts) > 0 else 0
    largest_loss_pct = min(profit_pcts) if profit_pcts and min(profit_pcts) < 0 else 0

    avg_win = (
        sum([p for p in profits if p > 0]) / len([p for p in profits if p > 0])
        if [p for p in profits if p > 0]
        else 0
    )
    avg_loss = (
        sum([p for p in profits if p < 0]) / len([p for p in profits if p < 0])
        if [p for p in profits if p < 0]
        else 0
    )

    # Calculate profit factor
    gross_profit = (
        sum([p for p in profits if p > 0]) if [p for p in profits if p > 0] else 0
    )
    gross_loss = (
        abs(sum([p for p in profits if p < 0])) if [p for p in profits if p < 0] else 0
    )
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    # Print statistics
    print(f"\n{name} - PAPER TRADING STATISTICS:")
    print(f"{'=' * 50}")
    print(f"Total trades: {len(trades)}")
    print(
        f"Winning trades: {len([t for t in trades if t['pnl'] > 0])} ({len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100:.2f}%)"
    )
    print(
        f"Losing trades: {len([t for t in trades if t['pnl'] < 0])} ({len([t for t in trades if t['pnl'] < 0]) / len(trades) * 100:.2f}%)"
    )
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Max win streak: {max_win_streak}")
    print(f"Max loss streak: {abs(max_loss_streak)}")
    print(f"Largest win: ${largest_win:.2f} ({largest_win_pct:.2f}%)")
    print(f"Largest loss: ${largest_loss:.2f} ({largest_loss_pct:.2f}%)")
    print(f"Average win: ${avg_win:.2f}")
    print(f"Average loss: ${avg_loss:.2f}")
    print(
        f"Win/Loss ratio: {avg_win / abs(avg_loss) if avg_loss != 0 else float('inf'):.2f}"
    )
    print(
        f"Portfolio growth: ${metrics['final_balance'] - metrics['initial_balance']:.2f} ({metrics['total_return_pct']:.2f}%)"
    )
    print(f"Starting balance: ${metrics['initial_balance']:.2f}")
    print(f"Final balance: ${metrics['final_balance']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Total fees paid: ${metrics['total_fee_paid']:.2f}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"{'=' * 50}")


def main():
    # Parse command-line arguments
    config = parse_args()

    print("Starting Lorentzian Model Comparison Test...")
    print(f"Configuration: {config}")

    # Fetch data
    df = fetch_training_data(config)
    features, prices = prepare_features(df)

    # Test each implementation
    your_predictions = test_your_implementation(features, prices)
    modern_predictions = test_modern_lorentzian(features, prices)
    standalone_predictions = test_standalone_lorentzian(features, prices)
    analysis_predictions = test_analysis_lorentzian(features, prices)

    # Calculate metrics with portfolio settings
    your_metrics = calculate_metrics(your_predictions, prices, config)
    modern_metrics = calculate_metrics(modern_predictions, prices, config)
    standalone_metrics = calculate_metrics(standalone_predictions, prices, config)
    analysis_metrics = calculate_metrics(analysis_predictions, prices, config)

    # Print stats for each prediction set
    print_prediction_stats("Your Implementation", your_predictions)
    print_prediction_stats("Modern PyTorch", modern_predictions)
    print_prediction_stats("Standalone", standalone_predictions)
    print_prediction_stats("Analysis", analysis_predictions)

    # Get output directory
    output_dir = config.get("output_dir", "results")

    # Create CSV report with results
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Win Rate (%)",
                "Total Return (%)",
                "Max Drawdown (%)",
                "Sharpe Ratio",
                "Signal Activity (%)",
                "Total Trades",
                "Winning Trades",
                "Losing Trades",
                "Profit Factor",
                "Starting Balance",
                "Final Balance",
                "Total Fees Paid",
            ],
            "Your Implementation": [
                your_metrics["win_rate"],
                your_metrics["total_return_pct"],
                your_metrics["max_drawdown"],
                your_metrics["sharpe_ratio"],
                your_metrics["signal_activity"],
                your_metrics["total_trades"],
                len([t for t in your_metrics["trades"] if t["pnl"] > 0])
                if your_metrics["trades"]
                else 0,
                len([t for t in your_metrics["trades"] if t["pnl"] < 0])
                if your_metrics["trades"]
                else 0,
                calculate_profit_factor(your_metrics["trades"]),
                your_metrics["initial_balance"],
                your_metrics["final_balance"],
                your_metrics["total_fee_paid"],
            ],
            "Modern PyTorch": [
                modern_metrics["win_rate"],
                modern_metrics["total_return_pct"],
                modern_metrics["max_drawdown"],
                modern_metrics["sharpe_ratio"],
                modern_metrics["signal_activity"],
                modern_metrics["total_trades"],
                len([t for t in modern_metrics["trades"] if t["pnl"] > 0])
                if modern_metrics["trades"]
                else 0,
                len([t for t in modern_metrics["trades"] if t["pnl"] < 0])
                if modern_metrics["trades"]
                else 0,
                calculate_profit_factor(modern_metrics["trades"]),
                modern_metrics["initial_balance"],
                modern_metrics["final_balance"],
                modern_metrics["total_fee_paid"],
            ],
            "Standalone": [
                standalone_metrics["win_rate"],
                standalone_metrics["total_return_pct"],
                standalone_metrics["max_drawdown"],
                standalone_metrics["sharpe_ratio"],
                standalone_metrics["signal_activity"],
                standalone_metrics["total_trades"],
                len([t for t in standalone_metrics["trades"] if t["pnl"] > 0])
                if standalone_metrics["trades"]
                else 0,
                len([t for t in standalone_metrics["trades"] if t["pnl"] < 0])
                if standalone_metrics["trades"]
                else 0,
                calculate_profit_factor(standalone_metrics["trades"]),
                standalone_metrics["initial_balance"],
                standalone_metrics["final_balance"],
                standalone_metrics["total_fee_paid"],
            ],
            "Analysis": [
                analysis_metrics["win_rate"],
                analysis_metrics["total_return_pct"],
                analysis_metrics["max_drawdown"],
                analysis_metrics["sharpe_ratio"],
                analysis_metrics["signal_activity"],
                analysis_metrics["total_trades"],
                len([t for t in analysis_metrics["trades"] if t["pnl"] > 0])
                if analysis_metrics["trades"]
                else 0,
                len([t for t in analysis_metrics["trades"] if t["pnl"] < 0])
                if analysis_metrics["trades"]
                else 0,
                calculate_profit_factor(analysis_metrics["trades"]),
                analysis_metrics["initial_balance"],
                analysis_metrics["final_balance"],
                analysis_metrics["total_fee_paid"],
            ],
        }
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    results_df.to_csv(
        os.path.join(output_dir, "model_comparison_results.csv"), index=False
    )

    # Print original trading metrics
    print("\n" + "=" * 80)
    print("TRADITIONAL MODEL EVALUATION METRICS")
    print("=" * 80)

    print("\nYour Implementation Trading Metrics:")
    print(f"Win Rate: {your_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {your_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {your_metrics['buy_signals']}")
    print(f"  - Sell Signals: {your_metrics['sell_signals']}")
    print(f"  - Hold Signals: {your_metrics['hold_signals']}")
    print(f"Signal Activity: {your_metrics['signal_activity']:.2f}%")
    print(f"Total Return: {your_metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {your_metrics['max_drawdown']:.2f}%")
    print(
        f"Avg Profit per Trade: ${your_metrics['avg_profit']:.2f} ({your_metrics['avg_profit_pct']:.2f}%)"
    )
    print(f"Sharpe Ratio: {your_metrics['sharpe_ratio']:.4f}\n")

    print("Modern Trading Metrics:")
    print(f"Win Rate: {modern_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {modern_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {modern_metrics['buy_signals']}")
    print(f"  - Sell Signals: {modern_metrics['sell_signals']}")
    print(f"  - Hold Signals: {modern_metrics['hold_signals']}")
    print(f"Signal Activity: {modern_metrics['signal_activity']:.2f}%")
    print(f"Total Return: {modern_metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {modern_metrics['max_drawdown']:.2f}%")
    print(
        f"Avg Profit per Trade: ${modern_metrics['avg_profit']:.2f} ({modern_metrics['avg_profit_pct']:.2f}%)"
    )
    print(f"Sharpe Ratio: {modern_metrics['sharpe_ratio']:.4f}\n")

    print("Standalone Trading Metrics:")
    print(f"Win Rate: {standalone_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {standalone_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {standalone_metrics['buy_signals']}")
    print(f"  - Sell Signals: {standalone_metrics['sell_signals']}")
    print(f"  - Hold Signals: {standalone_metrics['hold_signals']}")
    print(f"Signal Activity: {standalone_metrics['signal_activity']:.2f}%")
    print(f"Total Return: {standalone_metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {standalone_metrics['max_drawdown']:.2f}%")
    print(
        f"Avg Profit per Trade: ${standalone_metrics['avg_profit']:.2f} ({standalone_metrics['avg_profit_pct']:.2f}%)"
    )
    print(f"Sharpe Ratio: {standalone_metrics['sharpe_ratio']:.4f}\n")

    print("Analysis Trading Metrics:")
    print(f"Win Rate: {analysis_metrics['win_rate']:.2f}%")
    print(f"Total Trades: {analysis_metrics['total_trades']}")
    print("Signal Distribution:")
    print(f"  - Buy Signals: {analysis_metrics['buy_signals']}")
    print(f"  - Sell Signals: {analysis_metrics['sell_signals']}")
    print(f"  - Hold Signals: {analysis_metrics['hold_signals']}")
    print(f"Signal Activity: {analysis_metrics['signal_activity']:.2f}%")
    print(f"Total Return: {analysis_metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {analysis_metrics['max_drawdown']:.2f}%")
    print(
        f"Avg Profit per Trade: ${analysis_metrics['avg_profit']:.2f} ({analysis_metrics['avg_profit_pct']:.2f}%)"
    )
    print(f"Sharpe Ratio: {analysis_metrics['sharpe_ratio']:.4f}\n")

    # Print detailed paper trading statistics
    print("\n" + "=" * 80)
    print("PAPER TRADING SIMULATION RESULTS")
    print("=" * 80)

    display_trade_statistics(your_metrics, "Your Implementation")
    display_trade_statistics(modern_metrics, "Modern PyTorch")
    display_trade_statistics(standalone_metrics, "Standalone")
    display_trade_statistics(analysis_metrics, "Analysis")

    # Plot metrics and comparison
    plot_metrics(
        your_metrics, modern_metrics, standalone_metrics, analysis_metrics, output_dir
    )
    plot_comparison(
        df,
        your_predictions,
        modern_predictions,
        standalone_predictions,
        analysis_predictions,
        output_dir,
    )

    # Plot equity curves
    plot_equity_curves(
        your_metrics, modern_metrics, standalone_metrics, analysis_metrics, output_dir
    )

    print(
        f"Test completed! Results saved to {output_dir}/model_comparison.png, {output_dir}/model_performance_comparison.png, and {output_dir}/equity_curves.png"
    )
    print(f"Detailed results exported to {output_dir}/model_comparison_results.csv")


def calculate_profit_factor(trades):
    """Calculate profit factor from trade list"""
    if not trades:
        return 0

    profits = sum([t["pnl"] for t in trades if t["pnl"] > 0])
    losses = abs(sum([t["pnl"] for t in trades if t["pnl"] < 0]))

    if losses == 0:
        return float("inf") if profits > 0 else 0

    return profits / losses


if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    main()
