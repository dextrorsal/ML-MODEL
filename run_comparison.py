#!/usr/bin/env python3
"""
Run Lorentzian Model Evaluation

This script provides a simple way to run the advanced model evaluation
with different configurations. It allows testing of different trading
pairs, timeframes, and market types to compare the performance of
different Lorentzian model implementations.

Usage examples:
- Basic evaluation with default settings:
  python run_comparison.py

- Compare models on Binance BTC futures market:
  python run_comparison.py --config binance_btc_futures

- Compare on Bitget SOL futures with custom initial balance:
  python run_comparison.py --config bitget_sol_futures --balance 5000

- Compare with custom symbol and timeframe:
  python run_comparison.py --symbol ETH/USDT --timeframe 1h
"""

import os
import sys
import asyncio
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variable to use virtual display for matplotlib
os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive backend


async def run_comparison(args):
    """Run the comparison with the given arguments"""
    try:
        # Import the main function from the evaluation script
        from model_evaluation.compare_with_backtester import main

        # Override sys.argv with our arguments for the parser inside main()
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # Keep the script name

        # Add all args as command-line parameters
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                if isinstance(arg_value, bool) and arg_value:
                    sys.argv.append(f"--{arg_name}")
                elif not isinstance(arg_value, bool):
                    sys.argv.append(f"--{arg_name}")
                    sys.argv.append(str(arg_value))

        # Run the main function
        await main()

        # Restore original argv
        sys.argv = original_argv

        return 0
    except Exception as e:
        print(f"Error running comparison: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lorentzian Model Evaluation")
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

    # Pass arguments via environment variables
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            os.environ[
                f"LORENTZIAN_COMPARISON_{arg_name.upper().replace('-', '_')}"
            ] = str(arg_value)

    # Run the comparison
    exit_code = asyncio.run(run_comparison(args))
    sys.exit(exit_code)
