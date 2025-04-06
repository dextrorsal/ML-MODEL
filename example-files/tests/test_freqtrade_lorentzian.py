#!/usr/bin/env python3
"""
Script to test the Lorentzian ANN strategy with Freqtrade's backtesting data
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add freqtrade module to path
freqtrade_path = Path.home() / "freqtrade"
if freqtrade_path.exists():
    sys.path.append(str(freqtrade_path))

try:
    from freqtrade.data.history import load_pair_history
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver
except ImportError:
    print("FreqTrade modules not found. Make sure FreqTrade is installed.")
    sys.exit(1)

# Import the strategy directly
from strategies.LorentzianStrategy.lorentzian_strategy import LorentzianStrategy, LorentzianANN, prepare_indicators, prepare_features

def load_data(pair="BTC/USDT", timeframe="5m", data_dir="user_data/data/bitget/futures"):
    """Load historical data for testing"""
    print(f"Loading data for {pair} ({timeframe})...")
    
    # Convert pair name to filename format
    filename = pair.replace("/", "_") + f"-{timeframe}-futures.feather"
    file_path = Path(data_dir) / filename
    
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return None
        
    # Load data
    df = pd.read_feather(file_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

def test_strategy(pair="SOL/USDT", timeframe="5m"):
    """Test the Lorentzian ANN strategy with Freqtrade data"""
    
    # Load data
    df = load_data(pair, timeframe)
    if df is None:
        return
    
    # Prepare indicators
    df = prepare_indicators(df)
    
    # Initialize model
    model = LorentzianANN(
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20
    )
    
    # Check if model exists
    model_path = Path("models/lorentzian_model.pt")
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        if model.load_model(model_path):
            print("Model loaded successfully")
        else:
            print("Failed to load model")
            return
    else:
        print(f"Model file not found at {model_path}")
        print("Training new model...")
        
        # Prepare features
        features, scaler = prepare_features(df)
        
        # Train model
        import torch
        features_tensor = torch.tensor(features, dtype=torch.float32)
        prices_tensor = torch.tensor(df['close'].values, dtype=torch.float32)
        
        # Train
        model.scaler = scaler
        model.fit(features_tensor, prices_tensor)
        model.save_model(model_path)
        
    # Generate predictions
    feature_cols = ['rsi_14', 'wt1', 'wt2', 'cci', 'adx']
    features = df[feature_cols].values
    
    # Scale features
    if model.scaler:
        scaled_features = np.zeros((len(features), len(feature_cols)))
        for i, col in enumerate(feature_cols):
            if col in model.scaler:
                mean = model.scaler[col]['mean']
                std = model.scaler[col]['std']
                scaled_features[:, i] = (features[:, i] - mean) / (std if std > 0 else 1)
    
    # Convert to tensor
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model.predict(features_tensor)
    
    # Add predictions to dataframe
    df['signal'] = predictions.cpu().numpy()
    
    # Calculate returns
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    
    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    
    # Print performance metrics
    total_trades = (df['position'].diff() != 0).sum()
    winning_trades = (df['strategy_returns'] > 0).sum()
    losing_trades = (df['strategy_returns'] < 0).sum()
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    final_return = df['cumulative_returns'].iloc[-1] if len(df) > 0 else 1.0
    
    print("\nPerformance Metrics:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Final Return: {final_return - 1:.2%}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.title(f"{pair} Price with Trading Signals")
    plt.plot(df.index, df['close'], label='Price', alpha=0.7)
    
    # Plot buy and sell signals
    buy_points = df[df['signal'] == 1].index
    sell_points = df[df['signal'] == -1].index
    
    plt.scatter(buy_points, df.loc[buy_points, 'close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_points, df.loc[sell_points, 'close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.legend()
    
    # Plot returns
    plt.subplot(2, 1, 2)
    plt.title('Strategy Cumulative Returns')
    plt.plot(df.index, df['cumulative_returns'], label='Strategy Returns')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.ylabel('Cumulative Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'lorentzian_freqtrade_{pair.replace("/", "_")}.png')
    print(f"Saved chart to lorentzian_freqtrade_{pair.replace('/', '_')}.png")
    
    return df

def test_freqtrade_strategy(config_file="config.json"):
    """Test strategy using Freqtrade's backtesting engine"""
    
    # This would use Freqtrade's backtesting engine
    # For now, we'll just initialize the strategy
    
    # Load strategy
    strategy = LorentzianStrategy({})
    print(f"Initialized Freqtrade strategy: {strategy.__class__.__name__}")
    print(f"Timeframe: {strategy.timeframe}")
    print(f"Stop loss: {strategy.stoploss:.2%}")
    print(f"ROI: {strategy.minimal_roi}")
    
    return strategy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Lorentzian ANN strategy with Freqtrade data')
    parser.add_argument('--pair', type=str, default='SOL/USDT', help='Trading pair to analyze')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe to use')
    parser.add_argument('--freqtrade', action='store_true', help='Use Freqtrade backtesting engine')
    
    args = parser.parse_args()
    
    if args.freqtrade:
        # Use Freqtrade backtesting engine
        test_freqtrade_strategy()
    else:
        # Run our custom test
        test_strategy(pair=args.pair, timeframe=args.timeframe) 