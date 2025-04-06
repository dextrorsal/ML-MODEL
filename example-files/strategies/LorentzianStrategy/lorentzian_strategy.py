"""
MAIN STRATEGY IMPLEMENTATION: Lorentzian ANN Trading Strategy

This is the main implementation file for the Lorentzian ANN trading strategy.
It integrates the core components (signal generation, confirmation, and risk management)
into a complete trading system that can be used with Freqtrade.

This implementation currently only uses the Lorentzian ANN classifier, but can be
extended to integrate the Logistic Regression confirmation model and Chandelier Exit
risk management component.

To fully integrate all three components (as per user requirement):
1. Import all three components
2. Use Lorentzian ANN for primary signals
3. Confirm signals with Logistic Regression
4. Manage risk with Chandelier Exit

File Structure:
- LorentzianANN class: Implementation of the primary signal generator
- Strategy class: FreqTrade strategy implementation with entry/exit logic
"""

import logging
from functools import reduce
from datetime import datetime
import numpy as np
import torch
from pandas import DataFrame
import os
import sys
import pandas as pd
from pathlib import Path

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

# Import indicators directly
from ..features.rsi import RSIIndicator
from ..features.wave_trend import WaveTrendIndicator
from ..features.cci import CCIIndicator
from ..features.adx import ADXIndicator

logger = logging.getLogger(__name__)

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LorentzianANN:
    """
    Implements a classifier using Approximate Nearest Neighbors with Lorentzian distance,
    similar to TradingView's approach.
    """

    def __init__(
        self,
        lookback_bars=50,  # How many historical bars to consider
        prediction_bars=4,  # How many bars into the future to predict
        k_neighbors=20,  # Number of nearest neighbors to consider
        use_regime_filter=True,
        use_volatility_filter=True,
        use_adx_filter=True,
        adx_threshold=20.0,
        regime_threshold=-0.1,
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

        Args:
            features: torch.Tensor of shape (n_samples, n_features)
            historical_features: torch.Tensor of shape (n_historical, n_features)

        Returns:
            distances: torch.Tensor of shape (n_samples, n_historical)
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
                # We use log1p for numerical stability
                # This is the true formula from TradingView
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
        future_prices = prices[self.prediction_bars :]
        current_prices = prices[: -self.prediction_bars]

        # Generate labels: 1 for long (price went up), -1 for short (price went down), 0 for neutral
        labels = torch.zeros(len(current_prices), dtype=torch.long)
        labels[future_prices > current_prices] = 1  # Long
        labels[future_prices < current_prices] = -1  # Short

        # We can only generate labels for data points that have future data
        return features[: len(labels)], labels

    def fit(self, features, prices):
        """
        Use ANN (Approximate Nearest Neighbors) to fit the model
        This stores the historical feature arrays for later lookup
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

        Args:
            features: torch.Tensor of shape (n_samples, n_features)

        Returns:
            Predicted labels: 1 for long, -1 for short, 0 for neutral
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
            batch_distances = self.lorentzian_distance(
                batch_features, self.feature_arrays
            )

            # Get indices of k-nearest neighbors
            _, indices = torch.topk(
                batch_distances,
                min(self.k_neighbors, len(batch_distances[0])),
                largest=False,
                dim=1,
            )

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
            "feature_arrays": self.feature_arrays.cpu(),
            "labels": self.labels.cpu(),
            "scaler": self.scaler,
            "config": {
                "lookback_bars": self.lookback_bars,
                "prediction_bars": self.prediction_bars,
                "k_neighbors": self.k_neighbors,
                "use_regime_filter": self.use_regime_filter,
                "use_volatility_filter": self.use_volatility_filter,
                "use_adx_filter": self.use_adx_filter,
            },
            "metadata": {
                "date_saved": pd.Timestamp.now().isoformat(),
                "samples": len(self.feature_arrays),
            },
        }

        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save model
            torch.save(save_dict, path, weights_only=False)
            logger.info(f"Model saved to {path}")
            logger.info(
                f"Saved {len(self.feature_arrays)} samples with {len(self.scaler) if self.scaler else 0} features"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, path=None):
        """Load model state from file"""
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            logger.warning(f"Model file {path} does not exist")
            return False

        try:
            # Load with weights_only=False to allow loading complex objects
            # Note: Only use this with models from trusted sources
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Load configuration
            config = checkpoint["config"]
            self.lookback_bars = config["lookback_bars"]
            self.prediction_bars = config["prediction_bars"]
            self.k_neighbors = config["k_neighbors"]
            self.use_regime_filter = config["use_regime_filter"]
            self.use_volatility_filter = config["use_volatility_filter"]
            self.use_adx_filter = config["use_adx_filter"]

            # Load model data - make sure to move to the correct device
            self.feature_arrays = checkpoint["feature_arrays"].to(device)
            self.labels = checkpoint["labels"].to(device)
            self.scaler = checkpoint["scaler"]

            # Print metadata if available
            if "metadata" in checkpoint:
                metadata = checkpoint["metadata"]
                logger.info(f"Model saved on: {metadata.get('date_saved', 'Unknown')}")
                logger.info(f"Samples in model: {metadata.get('samples', 'Unknown')}")

            self.is_fitted = True
            logger.info(
                f"Model loaded from {path} with {len(self.feature_arrays)} samples"
            )
            logger.info(
                f"Configuration: lookback={self.lookback_bars}, prediction={self.prediction_bars}, k={self.k_neighbors}"
            )

            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def update_model(self, new_features, new_prices, max_samples=20000):
        """
        Update the model with new data without retraining from scratch
        This allows the model to adapt to new market conditions
        """
        if not self.is_fitted:
            logger.info("Model not fitted yet, using initial fit instead")
            return self.fit(new_features, new_prices)

        # Convert new data to tensors
        if not isinstance(new_features, torch.Tensor):
            new_features = torch.tensor(new_features, dtype=torch.float32)
        if not isinstance(new_prices, torch.Tensor):
            new_prices = torch.tensor(new_prices, dtype=torch.float32)

        # Move to device
        new_features = new_features.to(device)
        new_prices = new_prices.to(device)

        # Generate labels for new data
        new_features, new_labels = self.generate_training_data(new_features, new_prices)

        logger.info(f"Adding {len(new_features)} new samples to model")

        # Combine with existing data (keeping most recent samples)
        if len(self.feature_arrays) + len(new_features) > max_samples:
            # Keep most recent data
            keep_samples = max_samples - len(new_features)

            logger.info(
                f"Limiting model to {max_samples} samples (removing {len(self.feature_arrays) - keep_samples} old samples)"
            )

            self.feature_arrays = self.feature_arrays[-keep_samples:]
            self.labels = self.labels[-keep_samples:]

        # Make sure both tensors are on the same device before concatenating
        self.feature_arrays = self.feature_arrays.to(device)
        self.labels = self.labels.to(device)
        new_features = new_features.to(device)
        new_labels = new_labels.to(device)

        # Add new data
        self.feature_arrays = torch.cat([self.feature_arrays, new_features])
        self.labels = torch.cat([self.labels, new_labels])

        logger.info(f"Model updated: {len(self.feature_arrays)} total samples")

        return self


def prepare_indicators(df):
    """Add technical indicators"""
    df = df.copy()
    logger.info(f"Preparing indicators for data with shape: {df.shape}")

    # Initialize indicators
    rsi_14 = RSIIndicator(period=14, device=device)
    rsi_9 = RSIIndicator(period=9, device=device)
    wavetrend = WaveTrendIndicator(channel_length=10, average_length=11, device=device)
    cci = CCIIndicator(period=20, device=device)
    adx = ADXIndicator(period=20, device=device)

    # Convert data to tensors and move to GPU
    close = torch.tensor(df["close"].values, dtype=torch.float32).to(device)
    high = torch.tensor(df["high"].values, dtype=torch.float32).to(device)
    low = torch.tensor(df["low"].values, dtype=torch.float32).to(device)

    try:
        # Calculate indicators
        rsi_14_values = rsi_14.forward(close)["rsi"]
        rsi_9_values = rsi_9.forward(close)["rsi"]
        wt_values = wavetrend.forward(high, low, close)
        cci_values = cci.forward(high, low, close)["cci"]
        adx_values = adx.forward(high, low, close)["adx"]

        # Move tensors back to CPU
        df["rsi_14"] = rsi_14_values.cpu().numpy()
        df["rsi_9"] = rsi_9_values.cpu().numpy()
        df["wt1"] = wt_values["wt1"].cpu().numpy()
        df["wt2"] = wt_values["wt2"].cpu().numpy()
        df["cci"] = cci_values.cpu().numpy()
        df["adx"] = adx_values.cpu().numpy()

        # Calculate additional features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()
        df["regime"] = df["returns"].rolling(window=50).mean()

        # Drop NaN values
        df_before = len(df)
        df = df.dropna()
        df_after = len(df)
        logger.info(f"Dropped {df_before - df_after} rows with NaN values")

    except Exception as e:
        logger.error(f"Error in prepare_indicators: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    return df


def prepare_features(df):
    """Prepare and scale features for model input"""
    # Select features
    feature_cols = ["rsi_14", "wt1", "wt2", "cci", "adx"]

    # Simple standardization (mean=0, std=1)
    scaler = {}
    scaled_features = np.zeros((len(df), len(feature_cols)))

    for i, col in enumerate(feature_cols):
        mean, std = df[col].mean(), df[col].std()
        scaler[col] = {"mean": mean, "std": std}
        scaled_features[:, i] = (df[col].values - mean) / (std if std > 0 else 1)

    logger.info(f"Prepared features with shape: {scaled_features.shape}")
    return scaled_features, scaler


class LorentzianStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy parameters
    minimal_roi = {
        "0": 0.05,  # 5% minimum ROI
        "30": 0.025,  # 2.5% after 30 minutes
        "60": 0.01,  # 1% after 1 hour
        "120": 0,  # Exit after 2 hours regardless of profit
    }

    stoploss = -0.1  # 10% stop loss

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = "5m"

    # Hyperopt parameters
    lookback_bars = IntParameter(30, 60, default=50, space="buy", optimize=True)
    prediction_bars = IntParameter(2, 6, default=4, space="buy", optimize=True)
    k_neighbors = IntParameter(10, 30, default=20, space="buy", optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Initialize device
        self.device = device
        logger.info(f"Using device: {self.device}")

        # Initialize Lorentzian ANN model
        self.model = LorentzianANN(
            lookback_bars=self.lookback_bars.value,
            prediction_bars=self.prediction_bars.value,
            k_neighbors=self.k_neighbors.value,
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True,
        )

        # Load model weights if available
        model_path = Path("models/lorentzian_model.pt")
        if os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")
        else:
            logger.warning(f"No pre-trained model found at {model_path}")

        # Cache for indicator data
        self._indicator_cache = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators and model predictions"""

        pair = metadata["pair"]

        # Deep copy dataframe to avoid modifying the original
        df = dataframe.copy()

        # Add date/time columns for compatibility
        df["date"] = df["date"].astype(np.int64) // 1000000  # Convert to UNIX timestamp
        df["datetime"] = pd.to_datetime(df["date"], unit="s")

        # Calculate indicators
        try:
            # Check if we have calculated indicators for this pair recently
            if pair in self._indicator_cache:
                df_with_indicators = self._indicator_cache[pair]
                # If we have more data, just calculate for the new data
                if len(df) > len(df_with_indicators):
                    new_data = df.iloc[len(df_with_indicators) :].copy()
                    new_data_with_indicators = prepare_indicators(new_data)
                    df_with_indicators = pd.concat(
                        [df_with_indicators, new_data_with_indicators]
                    )
                    self._indicator_cache[pair] = df_with_indicators
            else:
                # Calculate indicators for all data
                df_with_indicators = prepare_indicators(df)
                self._indicator_cache[pair] = df_with_indicators

            # Add model predictions
            if self.model.is_fitted:
                # Prepare features for prediction
                feature_cols = ["rsi_14", "wt1", "wt2", "cci", "adx"]
                features = df_with_indicators[feature_cols].values

                # Scale features
                if self.model.scaler:
                    scaled_features = np.zeros((len(features), len(feature_cols)))
                    for i, col in enumerate(feature_cols):
                        if col in self.model.scaler:
                            mean = self.model.scaler[col]["mean"]
                            std = self.model.scaler[col]["std"]
                            scaled_features[:, i] = (features[:, i] - mean) / (
                                std if std > 0 else 1
                            )
                else:
                    # Simple standardization if no scaler available
                    scaled_features = (features - np.mean(features, axis=0)) / (
                        np.std(features, axis=0) + 1e-8
                    )

                # Convert to tensor
                features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(
                    self.device
                )

                # Get predictions
                with torch.no_grad():
                    predictions = self.model.predict(features_tensor)

                # Add predictions to dataframe
                dataframe["signal"] = predictions.cpu().numpy()
            else:
                # No model available, use neutral signal
                dataframe["signal"] = 0

        except Exception as e:
            logger.error(f"Error calculating indicators/predictions: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            dataframe["signal"] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals"""

        # Long entries
        dataframe.loc[
            (dataframe["signal"] == 1),  # Model predicts long
            "enter_long",
        ] = 1

        # Short entries
        dataframe.loc[
            (dataframe["signal"] == -1),  # Model predicts short
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals"""

        # Exit long when model predicts short
        dataframe.loc[(dataframe["signal"] == -1), "exit_long"] = 1

        # Exit short when model predicts long
        dataframe.loc[(dataframe["signal"] == 1), "exit_short"] = 1

        return dataframe

    def bot_start(self, **kwargs) -> None:
        """Called when bot starts"""
        logger.info("Lorentzian Strategy starting...")

    def bot_cleanup(self) -> None:
        """Called when bot stops"""
        # Clear cache
        self._indicator_cache.clear()
