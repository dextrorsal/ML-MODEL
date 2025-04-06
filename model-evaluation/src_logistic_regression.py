"""
Source Logistic Regression Implementation from src directory

This is a copy of the Logistic Regression implementation from:
'src/models/strategy/logistic_regression_torch.py'

Features:
- PyTorch implementation with GPU acceleration
- Classification algorithm for BUY/SELL signals
- Uses sigmoid curve for data separation
- Gradient descent for parameter optimization
- Normalization for prediction stability
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LogisticConfig:
    """Configuration for logistic regression model"""

    lookback: int = 3  # Number of bars to look back
    learning_rate: float = 0.0009  # Learning rate for optimization
    iterations: int = 1000  # Number of training iterations
    use_amp: bool = False  # Whether to use automatic mixed precision
    threshold: float = 0.5  # Classification threshold


@dataclass
class TradingMetrics:
    """Metrics for trading performance evaluation"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0


def minimax_scale(
    arr: np.ndarray, target_min: float = 0, target_max: float = 1
) -> np.ndarray:
    """
    Scale array values to target range

    Args:
        arr: Input array
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Scaled array with values in [target_min, target_max]
    """
    # Handle edge case
    if len(arr) == 0:
        return arr

    # Get min and max
    arr_min, arr_max = np.min(arr), np.max(arr)

    # Handle constant array
    if arr_min == arr_max:
        return np.full_like(arr, 0.5 * (target_min + target_max))

    # Scale to target range
    scaled = target_min + (target_max - target_min) * (arr - arr_min) / (
        arr_max - arr_min
    )
    return scaled


class LogisticRegression:
    """
    Logistic Regression model with PyTorch implementation

    Features:
    - Binary classification for trading signals
    - Gradient descent optimization
    - Simple and efficient implementation
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
        Initialize logistic regression model

        Args:
            lookback: Number of bars to look back
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
            use_amp: Whether to use automatic mixed precision
            threshold: Classification threshold
        """
        self.config = LogisticConfig(
            lookback=lookback,
            learning_rate=learning_rate,
            iterations=iterations,
            use_amp=use_amp,
            threshold=threshold,
        )
        self.metrics = TradingMetrics()

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model parameters
        self.weight = None
        self.bias = None

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor"""
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid function to input tensor"""
        return 1 / (1 + torch.exp(-z))

    def dot_product(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute the dot product of two tensors"""
        # Reshape w to match x's first dimension if needed
        if x.shape[0] != w.shape[0] and w.shape[0] != 1:
            w = w.expand(x.shape[0], -1)
        return torch.sum(x * w, dim=1, keepdim=True)

    def logistic_regression(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Logistic regression forward pass

        Args:
            x: Input features
            w: Weights
            b: Bias

        Returns:
            Predictions in range [0, 1]
        """
        z = self.dot_product(x, w) + b
        return self.sigmoid(z)

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for logistic regression

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Feature matrix
        """
        features = []

        # Price-based features
        close = data["close"].values

        # Returns over multiple timeframes
        for period in [1, 3, 5, 10]:
            if len(close) > period:
                returns = np.zeros_like(close)
                returns[period:] = (close[period:] - close[:-period]) / close[:-period]
                features.append(returns)

        # Moving averages
        for period in [10, 20, 50]:
            if len(close) > period:
                ma = np.zeros_like(close)
                for i in range(period, len(close)):
                    ma[i] = np.mean(close[i - period : i])

                # Distance from MA
                ma_dist = (close - ma) / ma
                features.append(ma_dist)

        # Convert to feature matrix
        X = np.column_stack(features)

        # Handle NaN and Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def prepare_labels(self, data: pd.DataFrame, forward_bars: int = 5) -> np.ndarray:
        """
        Prepare binary labels for classification

        Args:
            data: DataFrame with OHLCV data
            forward_bars: Number of bars to look ahead

        Returns:
            Binary labels (1 for price increase, 0 for decrease)
        """
        close = data["close"].values
        labels = np.zeros(len(close))

        # Future price movement (1 if increases, 0 if decreases)
        for i in range(len(close) - forward_bars):
            future_price = close[i + forward_bars]
            current_price = close[i]
            labels[i] = 1 if future_price > current_price else 0

        return labels

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model to the data"""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(
            -1, 1
        )

        # Initialize weights and bias if not already done
        if self.weight is None or self.bias is None:
            self.weight = torch.randn(
                1, X_tensor.shape[1], device=self.device, requires_grad=True
            )
            self.bias = torch.zeros(1, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [self.weight, self.bias], lr=self.config.learning_rate
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

        for i in range(self.config.iterations):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                y_pred = self.logistic_regression(X_tensor, self.weight, self.bias)
                loss = self.binary_cross_entropy(y_pred, y_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0 or 1)
        """
        # Handle case where model hasn't been trained
        if self.weight is None or self.bias is None:
            return np.zeros(len(X))

        # Convert to tensor
        X_tensor = self.to_tensor(X)

        # Make predictions
        with torch.no_grad():
            pred = self.logistic_regression(X_tensor, self.weight, self.bias)

        # Convert to numpy and apply threshold
        pred_np = pred.cpu().numpy().flatten()
        return (pred_np >= self.config.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability predictions [0, 1]
        """
        # Handle case where model hasn't been trained
        if self.weight is None or self.bias is None:
            return np.zeros(len(X))

        # Convert to tensor
        X_tensor = self.to_tensor(X)

        # Make predictions
        with torch.no_grad():
            pred = self.logistic_regression(X_tensor, self.weight, self.bias)

        # Convert to numpy
        return pred.cpu().numpy().flatten()

    def volatility_break(self, data: pd.DataFrame, window: int = 20) -> np.ndarray:
        """
        Calculate volatility breakout signal

        Args:
            data: DataFrame with OHLCV data
            window: Window size for calculation

        Returns:
            Boolean array where True indicates volatility breakout
        """
        close = data["close"].values

        # Calculate rolling standard deviation
        std = np.zeros_like(close)
        for i in range(window, len(close)):
            std[i] = np.std(close[i - window : i])

        # Calculate 2-standard deviation bands
        upper_band = np.zeros_like(close)
        lower_band = np.zeros_like(close)

        for i in range(window, len(close)):
            mean = np.mean(close[i - window : i])
            upper_band[i] = mean + 2 * std[i]
            lower_band[i] = mean - 2 * std[i]

        # Generate breakout signals
        breakout = np.zeros_like(close, dtype=bool)
        breakout[window:] = (close[window:] > upper_band[window:]) | (
            close[window:] < lower_band[window:]
        )

        return breakout

    def volume_break(self, data: pd.DataFrame, window: int = 20) -> np.ndarray:
        """
        Calculate volume breakout signal

        Args:
            data: DataFrame with OHLCV data
            window: Window size for calculation

        Returns:
            Boolean array where True indicates volume breakout
        """
        if "volume" not in data.columns:
            return np.zeros(len(data), dtype=bool)

        volume = data["volume"].values

        # Calculate rolling mean and standard deviation
        vol_mean = np.zeros_like(volume)
        vol_std = np.zeros_like(volume)

        for i in range(window, len(volume)):
            vol_mean[i] = np.mean(volume[i - window : i])
            vol_std[i] = np.std(volume[i - window : i])

        # Calculate 2-standard deviation upper band
        upper_band = vol_mean + 2 * vol_std

        # Generate breakout signals
        breakout = np.zeros_like(volume, dtype=bool)
        breakout[window:] = volume[window:] > upper_band[window:]

        return breakout

    def calculate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate trading signals

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Array of signals: 1 (buy), -1 (sell), 0 (neutral)
        """
        # Prepare features
        X = self.prepare_features(data)

        # Get probabilities
        probs = self.predict_proba(X)

        # Generate signals based on probabilities
        signals = np.zeros(len(probs))
        signals[probs > self.config.threshold] = 1  # Buy signals
        signals[probs < (1 - self.config.threshold)] = -1  # Sell signals

        # Apply volatility and volume filters
        vol_break = self.volatility_break(data)
        volume_break = self.volume_break(data)

        # Only take signals during breakouts (optional)
        # signals = signals * (vol_break | volume_break)

        return signals

    def update_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Update model metrics based on predictions

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        # Calculate classification metrics
        true_pos = np.sum((y_pred == 1) & (y_true == 1))
        false_pos = np.sum((y_pred == 1) & (y_true == 0))
        true_neg = np.sum((y_pred == 0) & (y_true == 0))
        false_neg = np.sum((y_pred == 0) & (y_true == 1))

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
        if len(y_true) > 0:
            self.metrics.accuracy = np.sum(y_pred == y_true) / len(y_true)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics as dictionary

        Returns:
            Dictionary of metric names and values
        """
        return {
            "accuracy": self.metrics.accuracy,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1_score": self.metrics.f1_score,
            "profit_factor": self.metrics.profit_factor,
            "win_rate": self.metrics.win_rate,
            "max_drawdown": self.metrics.max_drawdown,
        }

    def plot_signals(self, data: pd.DataFrame, signals: np.ndarray) -> None:
        """
        Plot price data with trading signals

        Args:
            data: DataFrame with OHLCV data
            signals: Array of trading signals
        """
        try:
            import matplotlib.pyplot as plt

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot close price
            ax.plot(data.index, data["close"], label="Close Price")

            # Plot buy signals
            buy_idx = data.index[signals == 1]
            if len(buy_idx) > 0:
                ax.scatter(
                    buy_idx,
                    data.loc[buy_idx, "close"],
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy Signal",
                )

            # Plot sell signals
            sell_idx = data.index[signals == -1]
            if len(sell_idx) > 0:
                ax.scatter(
                    sell_idx,
                    data.loc[sell_idx, "close"],
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell Signal",
                )

            # Add labels and legend
            ax.set_title("Logistic Regression Trading Signals")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting signals: {e}")

    def binary_cross_entropy(self, y_pred, y_true):
        """
        Compute binary cross entropy loss

        Args:
            y_pred: Predicted values (0-1)
            y_true: Actual labels (0 or 1)
        """
        return torch.mean(
            -y_true * torch.log(y_pred + 1e-10)
            - (1 - y_true) * torch.log(1 - y_pred + 1e-10)
        )


# Alias for backward compatibility
LogisticRegressionTorch = LogisticRegression
