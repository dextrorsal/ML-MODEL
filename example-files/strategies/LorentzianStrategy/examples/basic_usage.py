#!/usr/bin/env python3
"""
Basic Usage Example for the Integrated ML Trading System

This example demonstrates how to:
1. Load and prepare data
2. Initialize the trading system
3. Train models
4. Generate signals
5. Run a backtest
6. Visualize results

For more advanced usage, see the documentation and other examples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))

# Import our components
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader
from strategies.LorentzianStrategy.config import default_config
from analyze_lorentzian_ann import prepare_indicators

def create_example_dir():
    """Create example directories if they don't exist"""
    example_data_dir = Path("example_data")
    example_data_dir.mkdir(exist_ok=True)
    
    example_models_dir = Path("example_models")
    example_models_dir.mkdir(exist_ok=True)
    
    example_results_dir = Path("example_results")
    example_results_dir.mkdir(exist_ok=True)
    
    return example_data_dir, example_models_dir, example_results_dir

def download_example_data(data_dir):
    """Download example data if it doesn't exist"""
    example_data_path = data_dir / "example_data.feather"
    
    if example_data_path.exists():
        print(f"Using existing example data: {example_data_path}")
        return example_data_path
    
    print("Example data not found. To use this script, you need to:")
    print("1. Obtain cryptocurrency price data (e.g., BTC/USDT)")
    print("2. Convert it to feather format with columns: date, open, high, low, close, volume")
    print("3. Save it as 'example_data/example_data.feather'")
    print()
    print("Alternatively, you can provide your own data file path when running this script.")
    
    return None

def basic_usage_example(data_path=None):
    """Demonstrate basic usage of the Integrated ML Trading System"""
    print("="*50)
    print("Integrated ML Trading System - Basic Usage Example")
    print("="*50)
    
    # Create example directories
    data_dir, models_dir, results_dir = create_example_dir()
    
    # Get data path
    if data_path is None:
        data_path = download_example_data(data_dir)
        if data_path is None:
            print("No data found. Please provide a data file path.")
            return
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_feather(data_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    
    # Take a subset for faster processing
    print(f"Original data size: {len(df)}")
    test_size = 5000  # Adjust as needed
    df = df.iloc[-test_size:]
    print(f"Using {len(df)} bars for example")
    
    # Prepare indicators
    print("Calculating technical indicators...")
    df = prepare_indicators(df)
    
    # Create trader instance
    print("Initializing trading system...")
    trader = IntegratedMLTrader(
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20,
        atr_period=22,
        atr_multiplier=3.0,
        logistic_threshold=0.6,
        model_dir=models_dir
    )
    
    # Train models
    print("Training models...")
    trader.train_models(df)
    
    # Save models
    print("Saving trained models...")
    trader.save_models()
    
    # Generate signals
    print("Generating trading signals...")
    signal_df = trader.generate_signals(df)
    
    # Show signal distribution
    long_signals = (signal_df['combined_signal'] == 1).sum()
    short_signals = (signal_df['combined_signal'] == -1).sum()
    neutral_signals = (signal_df['combined_signal'] == 0).sum()
    
    print("\nSignal Distribution:")
    print(f"Long signals: {long_signals} ({long_signals/len(signal_df)*100:.1f}%)")
    print(f"Short signals: {short_signals} ({short_signals/len(signal_df)*100:.1f}%)")
    print(f"Neutral: {neutral_signals} ({neutral_signals/len(signal_df)*100:.1f}%)")
    
    # Run backtest
    print("\nRunning backtest...")
    results = trader.backtest(signal_df, initial_capital=10000.0)
    
    # Save results
    results_path = results_dir / "backtest_results.csv"
    results.to_csv(results_path)
    print(f"Results saved to {results_path}")
    
    # Plot results
    print("Plotting results...")
    trader.plot_results(results)
    plt.savefig(results_dir / "backtest_plot.png")
    print(f"Plot saved to {results_dir / 'backtest_plot.png'}")
    
    # Print final performance
    print("\nPerformance Summary:")
    initial_capital = 10000.0
    final_capital = results['equity'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Final capital: ${final_capital:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    
    # Calculate key metrics
    total_trades = (results['trade_pnl'] != 0).sum()
    winning_trades = (results['trade_pnl'] > 0).sum()
    losing_trades = (results['trade_pnl'] < 0).sum()
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        print(f"Total trades: {total_trades}")
        print(f"Win rate: {win_rate:.2f}%")
    
    print("\nExample complete! Check the example_results directory for outputs.")
    
    return trader, results

if __name__ == "__main__":
    # Check if a data path is provided as argument
    data_path = None
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
        if not data_path.exists():
            print(f"Error: Data file {data_path} not found")
            sys.exit(1)
    
    # Run the example
    trader, results = basic_usage_example(data_path)
    
    # Show the plot
    plt.show() 