"""
Example Logistic Regression Implementation from example-files directory

This is a copy of the Logistic Regression implementation from:
'example-files/strategies/LorentzianStrategy/models/confirmation/logistic_regression_torch.py'

Features:
- PyTorch-based implementation with GPU support
- Configurable feature engineering
- Probability-based signal generation
- Support for both binary and multi-class classification
- Enhanced implementation combining traditional methods with deep learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class LogisticConfig:
    """Configuration parameters for logistic regression"""

    lookback: int = 3  # Lookback period for features
    learning_rate: float = 0.0009  # Learning rate for gradient descent
    iterations: int = 1000  # Number of training iterations
    use_amp: bool = False  # Whether to use automatic mixed precision
    threshold: float = 0.5  # Probability threshold for classification


@dataclass
class BacktestMetrics:
    """Metrics for trading performance evaluation"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    avg_trade: float = 0.0


def minimax_scale(
    x: np.ndarray, min_val: float = 0.0, max_val: float = 1.0
) -> np.ndarray:
    """Scale array to range [min_val, max_val]"""
    try:
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            return np.full_like(x, 0.5 * (min_val + max_val))
        return min_val + (max_val - min_val) * (x - x_min) / (x_max - x_min)
    except Exception as e:
        print(f"Error in minimax_scale: {e}")
        return np.zeros_like(x)


def normalize_array(x: np.ndarray) -> np.ndarray:
    """Normalize array to zero mean and unit variance"""
    try:
        mean, std = np.mean(x), np.std(x)
        if std == 0:
            return np.zeros_like(x)
        return (x - mean) / std
    except Exception as e:
        print(f"Error in normalize_array: {e}")
        return np.zeros_like(x)


class SingleDimLogisticRegression:
    """Fallback single-dimension logistic regression model"""

    def __init__(self, learning_rate: float = 0.001, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to input"""
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression model"""
        # Initialize parameters
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        # Gradient descent
        for i in range(self.iterations):
            # Linear model: z = X * w + b
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            dw = (1 / samples) * np.dot(X.T, (y_pred - y))
            db = (1 / samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for samples"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)


class LogisticRegressionTorch:
    """PyTorch implementation of logistic regression"""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.001,
        iterations: int = 1000,
        use_amp: bool = False,
    ):
        """
        Initialize PyTorch logistic regression model

        Args:
            input_dim: Number of input features
            learning_rate: Learning rate for optimization
            iterations: Number of training iterations
            use_amp: Whether to use automatic mixed precision
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()).to(
            self.device
        )
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.use_amp = use_amp
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using PyTorch"""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)

        # Training loop
        self.model.train()
        for i in range(self.iterations):
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_pred = self.model(X_tensor)
                loss = self.loss_fn(y_pred, y_tensor)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.cpu().numpy().flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


class LogisticRegression:
    """
    Enhanced logistic regression implementation combining traditional methods with deep learning.

    Features:
    - GPU acceleration via PyTorch
    - Automatic feature engineering
    - Fallback to single-dimension model if needed
    - Classification for trading signal generation
    """

    def __init__(
        self,
        lookback: int = 3,
        learning_rate: float = 0.0009,
        iterations: int = 1000,
        use_amp: bool = False,
        threshold: float = 0.5,
    ):
        """
        Initialize the logistic regression model.

        Args:
            lookback: Number of bars to look back for features
            learning_rate: Learning rate for optimization
            iterations: Number of training iterations
            use_amp: Whether to use automatic mixed precision
            threshold: Probability threshold for classification
        """
        self.config = LogisticConfig(
            lookback=lookback,
            learning_rate=learning_rate,
            iterations=iterations,
            use_amp=use_amp,
            threshold=threshold,
        )
        self.metrics = BacktestMetrics()
        self.model = None
        self.feature_names = []
        self.feature_importances = {}

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for logistic regression.

        Args:
            data: DataFrame with price data

        Returns:
            Array of features
        """
        # Reset feature names
        self.feature_names = []

        features = []

        # Price-based features
        close = data["close"].values
        if "open" in data.columns:
            open_prices = data["open"].values
            high_prices = data["high"].values
            low_prices = data["low"].values

            # Price differences
            features.append(normalize_array(close - open_prices))
            self.feature_names.append("close_minus_open")

            # High-Low range
            features.append(normalize_array(high_prices - low_prices))
            self.feature_names.append("high_minus_low")

            # Position within range
            range_pct = np.where(
                high_prices > low_prices,
                (close - low_prices) / (high_prices - low_prices),
                0.5,
            )
            features.append(range_pct)
            self.feature_names.append("range_position")

        # Returns over multiple timeframes
        for period in [1, 2, 3, 5, 8, 13, 21]:
            if len(close) > period:
                returns = np.zeros_like(close)
                returns[period:] = (close[period:] - close[:-period]) / close[:-period]
                features.append(returns)
                self.feature_names.append(f"return_{period}")

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            if len(close) > period:
                ma = np.zeros_like(close)
                for i in range(period, len(close)):
                    ma[i] = np.mean(close[i - period : i])

                # Distance from MA
                ma_dist = (close - ma) / ma
                features.append(ma_dist)
                self.feature_names.append(f"ma_dist_{period}")

                # MA slope
                ma_slope = np.zeros_like(close)
                ma_slope[1:] = (ma[1:] - ma[:-1]) / ma[:-1]
                features.append(ma_slope)
                self.feature_names.append(f"ma_slope_{period}")

        # Volume features if available
        if "volume" in data.columns:
            volume = data["volume"].values

            # Normalized volume
            norm_volume = normalize_array(volume)
            features.append(norm_volume)
            self.feature_names.append("norm_volume")

            # Volume change
            vol_change = np.zeros_like(volume)
            vol_change[1:] = (volume[1:] - volume[:-1]) / (volume[:-1] + 1)
            features.append(vol_change)
            self.feature_names.append("volume_change")

            # Price-volume correlation
            if len(close) > 10:
                pv_corr = np.zeros_like(close)
                for i in range(10, len(close)):
                    price_returns = (close[i - 10 : i] - close[i - 11 : i - 1]) / close[
                        i - 11 : i - 1
                    ]
                    vol_changes = (volume[i - 10 : i] - volume[i - 11 : i - 1]) / (
                        volume[i - 11 : i - 1] + 1
                    )
                    if np.std(price_returns) > 0 and np.std(vol_changes) > 0:
                        corr = np.corrcoef(price_returns, vol_changes)[0, 1]
                        pv_corr[i] = 0 if np.isnan(corr) else corr
                features.append(pv_corr)
                self.feature_names.append("price_volume_corr")

        # Stack features and handle NaN and infinity values
        X = np.column_stack(features)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return X

    def prepare_labels(self, data: pd.DataFrame, forward_bars: int = 5) -> np.ndarray:
        """
        Prepare target labels for binary classification.

        Args:
            data: DataFrame with price data
            forward_bars: Number of bars to look ahead for label generation

        Returns:
            Binary labels (1 for price increase, 0 for decrease)
        """
        close = data["close"].values
        labels = np.zeros(len(close))

        # Future returns: 1 if price increases, 0 if it decreases
        for i in range(len(close) - forward_bars):
            future_return = (close[i + forward_bars] - close[i]) / close[i]
            labels[i] = 1 if future_return > 0 else 0

        return labels

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression model to the data.

        Args:
            X: Feature matrix
            y: Target labels
        """
        try:
            # Initialize and train PyTorch model
            input_dim = X.shape[1]
            self.model = LogisticRegressionTorch(
                input_dim=input_dim,
                learning_rate=self.config.learning_rate,
                iterations=self.config.iterations,
                use_amp=self.config.use_amp,
            )
            self.model.fit(X, y)

            # Calculate feature importances
            if hasattr(self.model.model[0], "weight"):
                weights = self.model.model[0].weight.detach().cpu().numpy().flatten()
                for i, name in enumerate(self.feature_names):
                    if i < len(weights):
                        self.feature_importances[name] = abs(weights[i])

        except Exception as e:
            print(f"Error in logistic regression training: {e}")
            print("Falling back to single-dimension logistic regression")

            # Fallback to single-dimension model
            if X.shape[1] == 1:
                self.model = SingleDimLogisticRegression(
                    learning_rate=self.config.learning_rate,
                    iterations=self.config.iterations,
                )
                self.model.fit(X, y)
            else:
                # Use first principal component
                from sklearn.decomposition import PCA

                pca = PCA(n_components=1)
                X_pca = pca.fit_transform(X)

                self.model = SingleDimLogisticRegression(
                    learning_rate=self.config.learning_rate,
                    iterations=self.config.iterations,
                )
                self.model.fit(X_pca, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        if self.model is None:
            return np.zeros(len(X))

        return self.model.predict(X, threshold=self.config.threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if self.model is None:
            return np.zeros(len(X))

        return self.model.predict_proba(X)

    def fit_predict(
        self, data: pd.DataFrame, forward_bars: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data, fit model, and make predictions in one step.

        Args:
            data: DataFrame with price data
            forward_bars: Number of bars to look ahead for label generation

        Returns:
            Tuple of (binary predictions, probability predictions)
        """
        # Prepare features and labels
        X = self.prepare_features(data)
        y = self.prepare_labels(data, forward_bars)

        # Split data for training
        train_size = int(0.7 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]

        # Fit model
        self.fit(X_train, y_train)

        # Predict
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        # Calculate metrics on test set
        if train_size < len(X):
            X_test, y_test = X[train_size:], y[train_size:]
            y_pred_test = self.predict(X_test)

            true_pos = np.sum((y_pred_test == 1) & (y_test == 1))
            false_pos = np.sum((y_pred_test == 1) & (y_test == 0))
            true_neg = np.sum((y_pred_test == 0) & (y_test == 0))
            false_neg = np.sum((y_pred_test == 0) & (y_test == 1))

            # Avoid division by zero
            if true_pos + false_pos > 0:
                self.metrics.precision = true_pos / (true_pos + false_pos)
            if true_pos + false_neg > 0:
                self.metrics.recall = true_pos / (true_pos + false_neg)
            if self.metrics.precision + self.metrics.recall > 0:
                self.metrics.f1_score = (
                    2
                    * (self.metrics.precision * self.metrics.recall)
                    / (self.metrics.precision + self.metrics.recall)
                )
            if len(y_test) > 0:
                self.metrics.accuracy = np.sum(y_pred_test == y_test) / len(y_test)

        return y_pred, y_prob

    def generate_signals(
        self, data: pd.DataFrame, threshold: float = None
    ) -> np.ndarray:
        """
        Generate trading signals based on the logistic regression model.

        Args:
            data: DataFrame with price data
            threshold: Optional probability threshold (overrides config threshold)

        Returns:
            Array of signals: 1 (buy), -1 (sell), 0 (neutral)
        """
        # Prepare features
        X = self.prepare_features(data)

        # Get probabilities
        probs = self.predict_proba(X)

        # Use provided threshold or default from config
        if threshold is None:
            threshold = self.config.threshold

        # Generate signals: 1 for buy (prob > threshold), -1 for sell (prob < 1-threshold)
        signals = np.zeros(len(probs))
        signals[probs > threshold] = 1
        signals[probs < (1 - threshold)] = -1

        return signals

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances as a dictionary"""
        return self.feature_importances

    def get_metrics(self) -> BacktestMetrics:
        """Get current performance metrics"""
        return self.metrics
