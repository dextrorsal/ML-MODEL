import pandas as pd
import yfinance as yf
from strategy_backtest import StrategyBacktest
from torch_model import TradingModel, ModelConfig
import torch
import pandas_ta as ta

def main():
    # Download Solana data (last year, hourly timeframe)
    print("Downloading Solana data...")
    sol = yf.download('SOL-USD', start='2023-01-01', end='2024-01-01', interval='1h')
    
    # Initialize model config
    config = ModelConfig(
        input_size=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        sequence_length=30
    )
    
    # Initialize model
    print("\nInitializing PyTorch model...")
    model = TradingModel(config)
    
    # Initialize backtest
    print("\nInitializing backtest...")
    backtest = StrategyBacktest(
        initial_capital=10000,  # Start with $10k
        max_positions=2,        # Max 2 positions at once
        min_confidence=0.3,     # Minimum 30% confidence for entry
        model=model            # Use our PyTorch model
    )
    
    # Prepare features and run backtest
    print("Preparing features and running backtest...")
    features = model.prepare_features(sol)
    results = backtest.run_backtest(sol)
    
    # Print stats
    backtest.print_stats()
    
    # Plot results
    backtest.plot_results()

if __name__ == "__main__":
    main() 