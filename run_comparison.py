#!/usr/bin/env python
"""
Strategy Comparison Runner

This script compares different trading strategies:
1. Lorentzian Classifier
2. Logistic Regression
3. Chandelier Exit
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Also add src directory explicitly
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Python path set to: {sys.path}")


def main():
    """Parse arguments and run the strategy comparison"""
    parser = argparse.ArgumentParser(description="Run Strategy Comparison")
    parser.add_argument(
        "--config",
        type=str,
        default="config_samples/quick_test_config.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/strategy_comparison",
        help="Directory to save output files",
    )

    args, unknown = parser.parse_known_args()

    # Print the configuration
    print(f"Using config file: {args.config}")
    print(f"Output directory: {args.output_dir}")

    # Set up any additional environment variables or configurations here
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    # Import strategies
    try:
        from src.models.strategy.lorentzian_classifier import LorentzianANN
        from src.models.strategy.logistic_regression_torch import LogisticRegression
        from src.models.strategy.chandelier_exit import ChandelierExit

        print("Successfully imported all strategy implementations")
    except ImportError as e:
        print(f"Error importing strategies: {e}")
        sys.exit(1)

    # Import data fetcher
    try:
        from src.comparison.data_fetcher import DataFetcher

        print("Successfully imported data fetcher")
    except ImportError as e:
        print(f"Error importing data fetcher: {e}")
        sys.exit(1)

    # Run the comparison
    try:
        # Initialize strategies
        lorentzian = LorentzianANN(lookback_bars=20, prediction_bars=4, k_neighbors=20)
        logistic = LogisticRegression()
        chandelier = ChandelierExit()

        # Create data fetcher and get data
        data_fetcher = DataFetcher()
        df = data_fetcher.fetch_data()
        print(f"Fetched data shape: {df.shape}")

        # Test each strategy
        results = {}

        print("\nTesting Lorentzian Classifier...")
        lorentzian_signals = lorentzian.calculate_signals(df)
        results["Lorentzian"] = calculate_metrics(df, lorentzian_signals)

        print("\nTesting Logistic Regression...")
        logistic_signals = logistic.calculate_signals(df)
        results["Logistic"] = calculate_metrics(df, logistic_signals)

        print("\nTesting Chandelier Exit...")
        chandelier_signals = chandelier.calculate_signals(df)
        results["Chandelier"] = calculate_metrics(df, chandelier_signals)

        # Print comparison
        print("\n=== Strategy Comparison ===")
        for strategy, metrics in results.items():
            print(f"\n{strategy} Results:")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Number of Trades: {metrics['total_trades']}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{args.output_dir}/strategy_comparison.csv")
        print(f"\nResults saved to {args.output_dir}/strategy_comparison.csv")

    except Exception as e:
        print(f"Error running comparison: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def calculate_metrics(df, signals):
    """Calculate performance metrics for a strategy"""
    # Extract signals (assuming they're in the standard format)
    if isinstance(signals, dict):
        if "buy_signals" in signals and "sell_signals" in signals:
            combined_signals = signals["buy_signals"] - signals["sell_signals"]
        else:
            combined_signals = signals.get("signal", signals.get("predictions", None))
    else:
        combined_signals = signals

    # Convert to numpy if needed
    if isinstance(combined_signals, torch.Tensor):
        combined_signals = combined_signals.cpu().numpy()

    # Calculate returns
    price_returns = df["close"].pct_change().fillna(0)
    strategy_returns = price_returns * combined_signals

    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    win_rate = (strategy_returns > 0).mean()
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    # Calculate drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Count trades (signal changes)
    signal_changes = np.diff(combined_signals != 0)
    total_trades = np.sum(signal_changes)

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
    }


if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    main()
