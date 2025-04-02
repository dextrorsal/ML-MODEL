#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for SOL Trading Strategy
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.features.technical.indicators.technical_indicators import calculate_indicators
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier
from src.utils.performance_metrics import calculate_trading_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train trading model')
    parser.add_argument('--data-days', type=int, default=90, 
                        help='Number of days of historical data to use')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Percentage of data to use for testing')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save-path', type=str, default='models/trained',
                        help='Path to save the trained model')
    return parser.parse_args()

async def fetch_training_data(days=90):
    """Fetch historical data for training"""
    logger.info(f"Fetching {days} days of historical data for training")
    
    # Initialize data collector
    collector = SOLDataCollector()
    
    # Fetch data for multiple timeframes
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        df = await collector.fetch_historical_data(timeframe=tf, days=days)
        logger.info(f"Fetched {len(df)} records for {tf} timeframe")
        data[tf] = df
    
    return data

def preprocess_data(data):
    """Preprocess data for model training"""
    logger.info("Preprocessing data for training")
    
    # Use 1-hour timeframe as base
    df = data['1h'].copy()
    
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Drop rows with NaN values from indicator calculations
    df = df.dropna()
    
    # Create labels based on future price movement (simplified approach)
    # 1 for price going up by 1% or more in next 24 hours, 0 otherwise
    price_shift = 24  # 24 hours ahead
    threshold = 0.01  # 1% price increase
    
    df['future_return'] = df['close'].pct_change(price_shift).shift(-price_shift)
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    # Drop rows with NaN in labels
    df = df.dropna()
    
    # Select features
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'wt1', 'wt2', 'adx', 'atr', 'bb_upper', 'bb_lower',
        'cci', 'stoch_k', 'stoch_d'
    ]
    
    # Make sure we have all the features, else remove them from list
    available_features = [f for f in feature_columns if f in df.columns]
    
    # Create feature matrix and labels
    X = df[available_features].values
    y = df['label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    
    logger.info(f"Preprocessed data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Save feature names for later use
    feature_names = available_features
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

def train_model(X_train, y_train, X_test, y_test, args):
    """Train the model on preprocessed data"""
    logger.info("Training model")
    
    # Create PyTorch datasets
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    # Create model
    input_size = X_train.shape[1]
    model = LorentzianClassifier(input_size=input_size)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions.squeeze() == y_test_tensor).float().mean()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, f"lorentzian_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, losses

def evaluate_trading_performance(model, X_test, y_test, feature_names):
    """Evaluate the model's trading performance using various metrics"""
    logger.info("Evaluating trading performance")
    
    # Generate predictions
    X_test_tensor = torch.FloatTensor(X_test)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    # Create trade signals (1 for buy, -1 for sell, 0 for hold)
    buy_threshold = 0.7  # High confidence for buy
    sell_threshold = 0.3  # Low confidence for sell
    
    signals = np.zeros(len(predictions))
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    # Calculate performance metrics
    metrics = calculate_trading_metrics(y_test, signals)
    
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    return metrics

async def main():
    """Main function to train and evaluate the model"""
    args = parse_arguments()
    
    # Fetch training data
    data = await fetch_training_data(days=args.data_days)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    model, losses = train_model(X_train, y_train, X_test, y_test, args)
    
    # Evaluate trading performance
    metrics = evaluate_trading_performance(model, X_test, y_test, feature_names)
    
    # Save metrics
    metrics_path = os.path.join(args.save_path, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(args.save_path, f"loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(loss_plot_path)
    
    logger.info(f"Training completed. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 