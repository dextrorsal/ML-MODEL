"""
DUPLICATE IMPLEMENTATION: Lorentzian Strategy

This file contains a standalone implementation of the Lorentzian Strategy.
It duplicates much of the functionality from strategies/LorentzianStrategy/lorentzian_strategy.py.

RECOMMENDATION:
- Use this file OR the modular implementation in the LorentzianStrategy directory
- Avoid modifying both implementations to prevent divergence and confusion
- For a more modular and maintainable approach, prefer the implementation in the LorentzianStrategy directory

This implementation includes only the Lorentzian ANN component without integration of
Logistic Regression confirmation and Chandelier Exit risk management.
"""

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import logging
import torch
import os
from pathlib import Path
import torch.nn as nn
import sys

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib

# Set up logging
logger = logging.getLogger(__name__)

# Set up device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add numpy._core.multiarray.scalar to safe globals for torch loading
import torch.serialization
torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])

# We'll use TA-Lib directly instead of importing external indicator modules
logger.info("Using TA-Lib for indicators")

class LorentzianANN:
    """
    Implements a classifier using Approximate Nearest Neighbors with Lorentzian distance,
    similar to TradingView's approach.
    """
    def __init__(
            self,
            lookback_bars=50,      # How many historical bars to consider
            prediction_bars=4,     # How many bars into the future to predict
            k_neighbors=20,        # Number of nearest neighbors to consider
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True,
            adx_threshold=20.0,
            regime_threshold=-0.1
        ):
        self.lookback_bars = lookback_bars
        self.prediction_bars = prediction_bars
        self.k_neighbors = k_neighbors
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_adx_filter = use_adx_filter
        self.adx_threshold = adx_threshold
        self.regime_threshold = regime_threshold
        
        # These will store our historical data
        self.feature_arrays = None
        self.labels = None
        self.scaler = None
        
        # Path for model persistence
        self.model_dir = Path("models")
        self.model_path = self.model_dir / "lorentzian_model.pt"
        self.is_fitted = False
        
        # Create models directory if it doesn't exist
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.model_dir}")
    
    def lorentzian_distance(self, features, historical_features):
        """
        Calculate Lorentzian distance between features and historical features
        """
        # Process in batches to save memory
        batch_size = 100  # Adjust based on available memory
        n_samples = features.shape[0]
        n_historical = historical_features.shape[0]
        
        # Initialize distances tensor
        distances = torch.zeros((n_samples, n_historical), device=device)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_features = features[i:end_i]
            
            # For each feature vector in the batch
            for j in range(batch_features.shape[0]):
                # Get the feature vector
                x = batch_features[j]
                
                # Calculate differences
                diff = torch.abs(x.unsqueeze(0) - historical_features)
                
                # Calculate Lorentzian distance: ln(1 + |x - y|)
                log_diff = torch.log1p(diff)
                
                # Sum over features dimension
                batch_distances = torch.sum(log_diff, dim=1)
                
                # Store in the distances tensor
                distances[i + j] = batch_distances
                
        return distances
    
    def generate_training_data(self, features, prices):
        """Generate training labels for the prediction task"""
        # Using the same TradingView approach: look ahead a fixed number of bars
        # to determine if price went up or down
        future_prices = prices[self.prediction_bars:]
        current_prices = prices[:-self.prediction_bars]
        
        # Generate labels: 1 for long (price went up), -1 for short (price went down), 0 for neutral
        labels = torch.zeros(len(current_prices), dtype=torch.long)
        labels[future_prices > current_prices] = 1  # Long
        labels[future_prices < current_prices] = -1  # Short
        
        # We can only generate labels for data points that have future data
        return features[:len(labels)], labels
        
    def fit(self, features, prices):
        """
        Use ANN (Approximate Nearest Neighbors) to fit the model
        """
        # Convert to tensors if they aren't already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float32)
            
        # Generate training data
        features, labels = self.generate_training_data(features, prices)
        
        # Store for lookup
        self.feature_arrays = features.to(device)
        self.labels = labels.to(device)
        self.is_fitted = True  # Set this flag to indicate the model is fitted
        
        return self
    
    def predict(self, features):
        """
        Predict using Approximate Nearest Neighbors with Lorentzian distance
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
            
        # Move to same device as model
        features = features.to(device)
        
        # Process in batches to avoid memory issues
        batch_size = 1000  # Adjust based on GPU memory
        n_samples = len(features)
        all_predictions = []
        
        logger.info(f"Processing {n_samples} samples in batches of {batch_size}...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = features[start_idx:end_idx]
            
            # Calculate Lorentzian distances for this batch
            batch_distances = self.lorentzian_distance(batch_features, self.feature_arrays)
            
            # Get indices of k-nearest neighbors
            _, indices = torch.topk(batch_distances, min(self.k_neighbors, len(batch_distances[0])), 
                                   largest=False, dim=1)
            
            # Get labels of k-nearest neighbors
            batch_neighbor_labels = [self.labels[idx] for idx in indices]
            batch_neighbor_labels = torch.stack(batch_neighbor_labels)
            
            # Calculate the sum of neighbor labels
            batch_predictions = torch.sum(batch_neighbor_labels, dim=1)
            
            # Convert to direction: 1 for long, -1 for short, 0 for neutral
            batch_final = torch.zeros_like(batch_predictions)
            batch_final[batch_predictions > 0] = 1
            batch_final[batch_predictions < 0] = -1
            
            all_predictions.append(batch_final)
            
            # Log progress
            progress = min(100, (end_idx / n_samples) * 100)
            if end_idx % (batch_size * 5) == 0:
                logger.info(f"Progress: {progress:.1f}%")
            
        # Combine all batches
        final_predictions = torch.cat(all_predictions)
        logger.info("Prediction complete!")
        
        return final_predictions

    def save_model(self, path=None):
        """Save model state to file"""
        if path is None:
            path = self.model_path
            
        if not self.is_fitted:
            logger.warning("Model not fitted yet, nothing to save")
            return False
        
        # Create a dictionary containing all necessary components
        save_dict = {
            'feature_arrays': self.feature_arrays.cpu(),
            'labels': self.labels.cpu(),
            'scaler': self.scaler,
            'config': {
                'lookback_bars': self.lookback_bars,
                'prediction_bars': self.prediction_bars,
                'k_neighbors': self.k_neighbors,
                'use_regime_filter': self.use_regime_filter,
                'use_volatility_filter': self.use_volatility_filter,
                'use_adx_filter': self.use_adx_filter
            },
            'metadata': {
                'date_saved': pd.Timestamp.now().isoformat(),
                'samples': len(self.feature_arrays)
            }
        }
        
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
            logger.info(f"Saved {len(self.feature_arrays)} samples with {len(self.scaler) if self.scaler else 0} features")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path=None):
        """Load model state from file"""
        if path is None:
            path = self.model_path
            
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return False
            
        try:
            # Load model
            checkpoint = torch.load(path, map_location=device)
            
            # Get feature arrays and labels
            self.feature_arrays = checkpoint['feature_arrays'].to(device)
            self.labels = checkpoint['labels'].to(device)
            self.scaler = checkpoint['scaler']
            
            # Get config
            config = checkpoint.get('config', {})
            self.lookback_bars = config.get('lookback_bars', self.lookback_bars)
            self.prediction_bars = config.get('prediction_bars', self.prediction_bars)
            self.k_neighbors = config.get('k_neighbors', self.k_neighbors)
            self.use_regime_filter = config.get('use_regime_filter', self.use_regime_filter)
            self.use_volatility_filter = config.get('use_volatility_filter', self.use_volatility_filter)
            self.use_adx_filter = config.get('use_adx_filter', self.use_adx_filter)
            
            self.is_fitted = True
            
            # Log metadata
            metadata = checkpoint.get('metadata', {})
            logger.info(f"Loaded model from {path} with {len(self.feature_arrays)} samples")
            logger.info(f"Model saved on: {metadata.get('date_saved', 'unknown')}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def update_model(self, features, prices, max_samples=500):
        """
        Update the model with new data, keeping a maximum number of samples to avoid memory issues.
        Useful for incremental learning.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float32)
            
        # Generate training data
        features, labels = self.generate_training_data(features, prices)
        
        # If model is already fitted, combine with existing data
        if self.is_fitted:
            # Move existing data to CPU for concatenation
            existing_features = self.feature_arrays.cpu()
            existing_labels = self.labels.cpu()
            
            # Combine old and new data
            combined_features = torch.cat([existing_features, features])
            combined_labels = torch.cat([existing_labels, labels])
            
            # Keep only the most recent max_samples
            if len(combined_features) > max_samples:
                combined_features = combined_features[-max_samples:]
                combined_labels = combined_labels[-max_samples:]
                
            # Store back to device
            self.feature_arrays = combined_features.to(device)
            self.labels = combined_labels.to(device)
        else:
            # First time training
            self.feature_arrays = features.to(device)
            self.labels = labels.to(device)
            
        self.is_fitted = True
        logger.info(f"Model updated with new data, now has {len(self.feature_arrays)} samples")
        
        return self


def prepare_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Add technical indicators to dataframe
    """
    # RSI
    dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    
    # WaveTrend
    ap = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3.0
    esa = ta.EMA(ap, timeperiod=10)
    d = ta.EMA(abs(ap - esa), timeperiod=10)
    ci = (ap - esa) / (0.015 * d)
    dataframe['wt1'] = ta.EMA(ci, timeperiod=21)
    dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=4)
    
    # CCI
    dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
    
    # ADX
    dataframe['adx'] = ta.ADX(dataframe)
    
    # Bollinger Bands
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe['bb_lowerband'] = bollinger['lower']
    dataframe['bb_middleband'] = bollinger['mid']
    dataframe['bb_upperband'] = bollinger['upper']
    
    # Normalize indicators to help with training
    for col in ['rsi', 'wt1', 'wt2', 'cci', 'adx']:
        if col in dataframe.columns:
            # Simple min-max scaling
            min_val = dataframe[col].rolling(window=100, min_periods=30).min()
            max_val = dataframe[col].rolling(window=100, min_periods=30).max()
            # Avoid division by zero by adding small epsilon
            dataframe[f'{col}_norm'] = (dataframe[col] - min_val) / (max_val - min_val + 1e-6)
            
    return dataframe


def prepare_features(dataframe: DataFrame):
    """
    Prepare and scale features for model input
    """
    # List of normalized features to use for model
    feature_columns = ['rsi_norm', 'wt1_norm', 'wt2_norm', 'cci_norm', 'adx_norm']
    
    # Drop NaN values
    dataframe = dataframe.dropna(subset=feature_columns)
    
    if len(dataframe) == 0:
        return np.array([]), None
    
    # Extract features
    features = dataframe[feature_columns].values
    
    # Create a simple scaler for future use
    scaler = {
        'feature_names': feature_columns
    }
    
    return features, scaler


class LorentzianStrategy(IStrategy):
    """
    Lorentzian ANN Strategy based on k-Nearest Neighbors with Lorentzian distance
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.05  # 5% profit is enough to exit
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.10  # 10% stop loss

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = "5m"
    
    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Strategy parameters
    lookback_bars = IntParameter(30, 100, default=50, space="buy")
    prediction_bars = IntParameter(1, 10, default=4, space="buy")
    k_neighbors = IntParameter(5, 40, default=20, space="buy")
    
    # Class level variables
    model = None
    last_training_time = None
    training_interval_hours = 6  # Train the model every 6 hours
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the strategy
        """
        super().__init__(config)
        
        # Initialize model at strategy creation
        self.model = LorentzianANN(
            lookback_bars=self.lookback_bars.value,
            prediction_bars=self.prediction_bars.value,
            k_neighbors=self.k_neighbors.value
        )
        
        # Try to load a saved model
        models_dir = Path("models")
        model_path = models_dir / "lorentzian_model.pt"
        
        if model_path.exists():
            self.model.load_model(model_path)
            logger.info("Loaded existing Lorentzian model")
        else:
            logger.info("No existing model found, will train on first data")
            
        self.last_training_time = pd.Timestamp.now() - pd.Timedelta(hours=self.training_interval_hours)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # Skip if dataframe is empty
        if len(dataframe) == 0:
            return dataframe
            
        # Add indicators
        dataframe = prepare_indicators(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        # Initialize empty signal columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Skip if dataframe is empty
        if len(dataframe) < self.lookback_bars.value + self.prediction_bars.value:
            return dataframe
            
        # Prepare features for model
        features, scaler = prepare_features(dataframe)
        
        # Only proceed if we have features
        if len(features) == 0:
            return dataframe
            
        # Save scaler if not set
        if self.model.scaler is None and scaler is not None:
            self.model.scaler = scaler
        
        # Check if we need to train the model (first time or periodically)
        current_time = pd.Timestamp.now()
        training_due = current_time - self.last_training_time > pd.Timedelta(hours=self.training_interval_hours)
        
        if not self.model.is_fitted or training_due:
            # Use a window of data for training
            train_features = features[-min(500, len(features)):]
            train_prices = dataframe['close'].values[-min(500, len(features)):]
            
            # Update model
            self.model.update_model(
                features=train_features, 
                prices=train_prices, 
                max_samples=500
            )
            
            self.last_training_time = current_time
            logger.info(f"Model trained/updated at {current_time}")
        
        # Generate predictions
        try:
            predictions = self.model.predict(features).cpu().numpy()
            
            # Apply trading signals based on predictions
            dataframe.loc[:, 'lorentzian_prediction'] = 0  # Initialize with zeros
            
            # Map predictions to the corresponding rows in dataframe
            # Since we dropped NaN rows in feature preparation, we need to match indices
            valid_indices = dataframe.dropna(subset=self.model.scaler['feature_names']).index
            
            if len(valid_indices) == len(predictions):
                # Set predictions in dataframe
                for i, idx in enumerate(valid_indices):
                    dataframe.loc[idx, 'lorentzian_prediction'] = predictions[i]
            
                # Set entry signals
                dataframe.loc[dataframe['lorentzian_prediction'] == 1, 'enter_long'] = 1
                
                if self.can_short:
                    dataframe.loc[dataframe['lorentzian_prediction'] == -1, 'enter_short'] = 1
            else:
                logger.warning(f"Length mismatch: predictions {len(predictions)}, valid indices {len(valid_indices)}")
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # Initialize empty signal columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # We primarily use ROI and stoploss for exits, but can add exit signals here
        # Example: exit when prediction changes direction
        if 'lorentzian_prediction' in dataframe.columns:
            # Exit long positions when prediction is -1 (bearish)
            dataframe.loc[dataframe['lorentzian_prediction'] == -1, 'exit_long'] = 1
            
            # Exit short positions when prediction is 1 (bullish)
            if self.can_short:
                dataframe.loc[dataframe['lorentzian_prediction'] == 1, 'exit_short'] = 1
        
        return dataframe
        
    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot loop (usually every 5 seconds).
        Ideal for updating the model periodically.
        """
        # Check if we need to save the model periodically
        if self.model and self.model.is_fitted:
            current_time = pd.Timestamp.now()
            last_save = getattr(self, 'last_model_save', pd.Timestamp.min)
            
            # Save model every 24 hours
            if current_time - last_save > pd.Timedelta(hours=24):
                self.model.save_model()
                self.last_model_save = current_time
                logger.info(f"Model saved at {current_time}")
                
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Customize stake size based on model confidence or other factors
        """
        # For now, simply use the proposed stake
        return proposed_stake