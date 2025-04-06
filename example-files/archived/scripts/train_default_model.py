#!/usr/bin/env python3
"""
Script to train a default Lorentzian ANN model for use with live trading
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('default_model_training')

# Import our Lorentzian ANN implementation
try:
    from strategies.LorentzianStrategy.lorentzian_strategy import LorentzianANN, prepare_indicators, prepare_features
except ImportError:
    logger.error("Could not import LorentzianANN. Make sure the strategy is properly installed.")
    sys.exit(1)

def train_default_model(pairs=None, timeframe="5m", data_dir="data/bitget/futures", output_path="models/lorentzian_model.pt"):
    """Train a default model using data from multiple pairs"""
    
    # Default pairs
    if pairs is None:
        pairs = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
    
    logger.info(f"Training default model with pairs: {pairs}")
    
    # Initialize model
    model = LorentzianANN(
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each pair
    all_features = []
    all_prices = []
    
    for pair in pairs:
        # Convert pair name to filename format
        filename = pair.replace("/", "_") + f"-{timeframe}-futures.feather"
        file_path = Path(data_dir) / filename
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            continue
            
        # Load data
        df = pd.read_feather(file_path)
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        
        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]} for {pair}")
        
        # Prepare indicators
        df = prepare_indicators(df)
        
        # Prepare features
        features, scaler = prepare_features(df)
        
        # Store scaler from first pair
        if not hasattr(model, 'scaler') or model.scaler is None:
            model.scaler = scaler
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        prices_tensor = torch.tensor(df['close'].values, dtype=torch.float32)
        
        # Collect data
        all_features.append(features_tensor)
        all_prices.append(prices_tensor)
    
    if not all_features:
        logger.error("No valid data found for training")
        return False
    
    # Combine data from all pairs
    combined_features = torch.cat(all_features)
    combined_prices = torch.cat(all_prices)
    
    logger.info(f"Training model with {len(combined_features)} samples from {len(pairs)} pairs")
    
    # Train model
    model.fit(combined_features, combined_prices)
    
    # Save model
    if model.save_model(output_path):
        logger.info(f"Default model saved to {output_path}")
        return True
    else:
        logger.error("Failed to save model")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a default Lorentzian ANN model')
    parser.add_argument('--pairs', type=str, nargs='+', help='Trading pairs to use for training')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe to use')
    parser.add_argument('--output', type=str, default='models/lorentzian_model.pt', help='Output path for model')
    
    args = parser.parse_args()
    
    train_default_model(
        pairs=args.pairs,
        timeframe=args.timeframe,
        output_path=args.output
    ) 