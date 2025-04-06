#!/usr/bin/env python3
"""
TESTING COMPONENT: Integrated Backtesting System

This script provides a complete backtesting environment for the Integrated ML Trading System.
It combines all three core components (Lorentzian ANN, Logistic Regression, and Chandelier Exit)
to simulate trading on historical data and evaluate performance metrics.

Key features:
- Support for different configuration profiles (default, aggressive, conservative)
- Comprehensive backtesting with realistic trade execution
- Performance metrics calculation and visualization
- GPU acceleration support
- Model training, saving and loading capabilities
- Trading parameter optimization

This backtester is ideal for:
1. Evaluating overall system performance on historical data
2. Comparing different configuration profiles
3. Testing model stability and consistency
4. Analyzing risk-reward characteristics
5. Optimizing trading parameters

Command-line usage:
```
python run_backtest.py --data path/to/ohlcv.feather --config default --periods 10000 --capital 10000
```

Arguments:
  --data         Path to OHLCV data file (Feather format)
  --config       Configuration profile to use (default, aggressive, conservative)
  --periods      Number of periods to use for testing
  --capital      Initial capital for backtest
  --leverage     Leverage to use for trades
  --gpu          Use GPU acceleration
  --save-models  Save models after training
  --load-models  Load pre-trained models
  --output       Output directory for results
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

# Import our components
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader
from strategies.LorentzianStrategy.config import (
    default_config,
    aggressive_config,
    conservative_config,
    IntegratedSystemConfig
)

# Import indicator preparation function
try:
    from analyze_lorentzian_ann import prepare_indicators
except ImportError:
    print("Warning: Could not import prepare_indicators from analyze_lorentzian_ann.py")
    print("You'll need to prepare indicators yourself")
    
    def prepare_indicators(df):
        """Placeholder for indicator preparation function"""
        print("Using placeholder prepare_indicators function")
        print("Please implement proper indicator calculation")
        return df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run a backtest of the Integrated ML Trading System")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to the data file (.feather format)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        choices=["default", "aggressive", "conservative"],
        default="default",
        help="Configuration profile to use"
    )
    
    parser.add_argument(
        "--periods",
        type=int,
        default=10000,
        help="Number of periods to use for testing"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for backtest"
    )
    
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage to use for trades"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration"
    )
    
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save models after training"
    )
    
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load pre-trained models"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results",
        help="Output directory for results"
    )
    
    return parser.parse_args()

def run_backtest(args):
    """Run the backtest with the specified arguments"""
    start_time = time.time()
    
    # Select configuration
    if args.config == "aggressive":
        config = aggressive_config
    elif args.config == "conservative":
        config = conservative_config
    else:
        config = default_config
    
    # Apply command-line overrides
    config.backtest.initial_capital = args.capital
    config.backtest.leverage = args.leverage
    config.use_gpu = args.gpu
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}")
    try:
        df = pd.read_feather(args.data)
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Limit data size if specified
    if args.periods and args.periods < len(df):
        print(f"Using last {args.periods} periods out of {len(df)}")
        df = df.iloc[-args.periods:]
    
    # Prepare indicators
    print("Calculating technical indicators...")
    df = prepare_indicators(df)
    
    # Create trader instance
    print(f"Initializing trader with {args.config} configuration")
    trader = IntegratedMLTrader(
        lookback_bars=config.lorentzian.lookback_bars,
        prediction_bars=config.lorentzian.prediction_bars,
        k_neighbors=config.lorentzian.k_neighbors,
        use_regime_filter=config.lorentzian.use_regime_filter,
        use_volatility_filter=config.lorentzian.use_volatility_filter,
        use_adx_filter=config.lorentzian.use_adx_filter,
        adx_threshold=config.lorentzian.adx_threshold,
        regime_threshold=config.lorentzian.regime_threshold,
        atr_period=config.chandelier.atr_period,
        atr_multiplier=config.chandelier.atr_multiplier,
        logistic_threshold=config.logistic.threshold
    )
    
    # Load or train models
    if args.load_models:
        print("Loading pre-trained models...")
        trader.load_models()
    else:
        print("Training models...")
        trader.train_models(df)
        
        if args.save_models:
            print("Saving trained models...")
            trader.save_models()
    
    # Generate signals
    print("Generating trading signals...")
    df = trader.generate_signals(df)
    
    # Run backtest
    print("Running backtest...")
    results_df = trader.backtest(
        df, 
        initial_capital=config.backtest.initial_capital
    )
    
    # Save results
    results_path = output_dir / "backtest_results.csv"
    results_df.to_csv(results_path)
    print(f"Backtest results saved to {results_path}")
    
    # Plot results
    print("Generating plots...")
    trader.plot_results(results_df)
    plot_path = output_dir / "backtest_plots.png"
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Backtest completed in {elapsed_time:.2f} seconds")
    
    # Calculate additional metrics
    total_return = results_df['equity'].iloc[-1] / config.backtest.initial_capital - 1
    annualized_return = (1 + total_return) ** (252 / len(results_df)) - 1
    volatility = results_df['returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = results_df['drawdown'].min()
    
    # Print summary
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    print(f"Strategy: Integrated ML Trading System ({args.config} configuration)")
    print(f"Period: {results_df.index[0]} to {results_df.index[-1]}")
    print(f"Number of bars: {len(results_df)}")
    print(f"Initial capital: ${config.backtest.initial_capital:,.2f}")
    print(f"Final capital: ${results_df['equity'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")
    print(f"Win rate: {(results_df['trade_pnl'] > 0).sum() / (results_df['trade_pnl'] != 0).sum():.2%}")
    print("="*50)
    
    return results_df

if __name__ == "__main__":
    args = parse_args()
    results = run_backtest(args) 