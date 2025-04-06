"""
COMPONENT: PRIMARY SIGNAL GENERATOR - Lorentzian Classifier

Enhanced Lorentzian Classifier with PyTorch Implementation

This module provides a PyTorch-based implementation of the Lorentzian Classifier,
combining traditional technical analysis with modern deep learning capabilities.

Features:
- GPU acceleration support
- Pine Script compatibility
- Real-time signal generation
- Advanced feature engineering
- Customizable filters
- Built-in visualization tools
- Backtesting metrics

This component serves as the PRIMARY SIGNAL GENERATOR in our trading system.
It generates the initial trading signals that will be confirmed by a secondary model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from contextlib import nullcontext
from ....features.rsi import RSIIndicator, RSIConfig
from ....features.wave_trend import WaveTrendIndicator
from ....features.cci import CCIIndicator
from ....features.adx import ADXIndicator


class Direction(Enum):
    """Trading direction enumeration"""

    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class MLFeatures:
    """Container for ML features"""

    momentum: torch.Tensor = None
    volatility: torch.Tensor = None
    trend: torch.Tensor = None
    volume: torch.Tensor = None


@dataclass
class LorentzianSettings:
    """Configuration for Lorentzian Classifier"""

    # General Settings
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = True
    use_amp: bool = False

    # Device Settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Feature Parameters
    momentum_lookback: int = 20
    volatility_lookback: int = 10
    trend_lookback: int = 50
    volume_lookback: int = 10

    # Filter Settings
    volatility_threshold: float = 1.2
    regime_threshold: float = 0.5
    adx_threshold: float = 25.0

    # Kernel Settings
    kernel_size: int = 3
    kernel_std: float = 1.0

    # Dynamic Exit Settings
    use_dynamic_exits: bool = True
    profit_target_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0

    # Display Settings
    show_plots: bool = True
    plot_lookback: int = 100


@dataclass
class FilterSettings:
    """Settings for signal filters"""

    volatility_enabled: bool = True
    regime_enabled: bool = True
    adx_enabled: bool = True
    volume_enabled: bool = True


@dataclass
class TradeStats:
    """Container for trading statistics"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0


class LorentzianDistance(nn.Module):
    """
    Custom Lorentzian distance layer for more robust classification
    """

    def __init__(self, sigma: float = 1.0):
        super(LorentzianDistance, self).__init__()
        self.sigma = sigma

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Lorentzian distance between two tensors
        """
        # Get squared distance
        squared_distance = torch.sum((x1 - x2) ** 2, dim=1)

        # Apply Lorentzian formula: log(1 + d²/sigma²)
        lorentzian_distance = torch.log(1 + squared_distance / (self.sigma**2))

        return lorentzian_distance


class WMACVotingSystem:
    """
    Weighted Moving Average Consensus Voting System
    Combines multiple timeframe signals with confidence weighting
    """

    def __init__(self):
        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # More balanced timeframe weights
        self.weights = {
            "short": 0.5,  # Was 0.7
            "medium": 0.3,  # Was 0.2
            "long": 0.2,  # Was 0.1
        }

    def calculate_regime_weight(self, volatility: torch.Tensor) -> torch.Tensor:
        """Calculate market regime weights based on volatility"""
        # Normalize volatility to [0, 1]
        vol_norm = (volatility - volatility.min()) / (
            volatility.max() - volatility.min() + 1e-8
        )

        # Higher weight for high volatility regimes (increased impact)
        regime_weight = 1.0 + 2.5 * vol_norm  # Was 2.0
        return regime_weight

    def apply_vote(
        self, probabilities: torch.Tensor, volatility: torch.Tensor
    ) -> torch.Tensor:
        """Apply weighted moving average consensus voting."""
        # Move inputs to device if needed
        probabilities = probabilities.to(self.device)
        volatility = volatility.to(self.device)

        # Normalize volatility to [0, 1] range for regime weighting
        vol_norm = (volatility - volatility.min()) / (
            volatility.max() - volatility.min() + 1e-8
        )

        # Calculate regime weights with reduced volatility impact
        regime_weights = (1.0 + 2.0 * vol_norm).to(self.device)  # Was 3.0

        # Temperature scaling for more balanced distributions
        temperature = 0.5  # Was 0.3
        scaled_probs = probabilities / temperature

        # Apply softmax to get normalized probabilities
        exp_probs = torch.exp(scaled_probs)
        softmax_probs = exp_probs / exp_probs.sum(dim=1, keepdim=True)

        # Calculate directional scores with balanced bias
        scores = torch.zeros_like(softmax_probs[:, 0], device=self.device)
        scores = scores + (
            softmax_probs[:, 0] * 1.1
        )  # Long probability with slight boost
        scores = scores - (
            softmax_probs[:, 1] * 0.9
        )  # Short probability with slight reduction

        # Initialize vote windows
        short_votes = torch.zeros_like(scores, device=self.device)
        medium_votes = torch.zeros_like(scores, device=self.device)
        long_votes = torch.zeros_like(scores, device=self.device)

        # Calculate votes for each timeframe with balanced windows
        for i in range(len(scores)):
            # Short-term: 5 periods
            start_idx = max(0, i - 5)
            short_votes[i] = scores[start_idx : i + 1].mean() * 1.2  # Was 1.4

            # Medium-term: 10 periods
            start_idx = max(0, i - 10)
            medium_votes[i] = (
                scores[start_idx : i + 1].mean() * 1.0
            )  # Added neutral scaling

            # Long-term: 20 periods
            start_idx = max(0, i - 20)
            long_votes[i] = scores[start_idx : i + 1].mean() * 0.8  # Was 0.6

        # Combine votes with timeframe weights and regime weighting
        final_votes = (
            self.weights["short"] * short_votes * regime_weights
            + self.weights["medium"] * medium_votes * regime_weights
            + self.weights["long"] * long_votes * regime_weights
        )

        # Convert votes to signals with balanced thresholds
        signals = torch.zeros_like(final_votes, device=self.device)
        signals[final_votes > 0.15] = 1  # Was 0.10
        signals[final_votes < -0.15] = -1  # Was -0.10

        return signals


class ModernLorentzian(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize indicators with TradingView settings
        self.rsi14 = RSIIndicator(
            period=14,
            overbought=70.0,
            oversold=30.0,
            device=self.device,
            dtype=torch.float32,
        )

        self.rsi9 = RSIIndicator(
            period=9,
            overbought=70.0,
            oversold=30.0,
            device=self.device,
            dtype=torch.float32,
        )

        self.wt = WaveTrendIndicator(
            channel_length=10,
            average_length=11,
            device=self.device,
        )

        self.cci = CCIIndicator(
            period=20,
            constant=0.015,
            device=self.device,
        )

        self.adx = ADXIndicator(
            period=20,
            threshold=20.0,  # ADX filter threshold
            device=self.device,
        )

        # k-NN parameters (from TradingView)
        self.k_neighbors = 8
        self.history_size = 2000  # Max bars back
        self.pattern_memory = []
        self.pattern_labels = []

        # Filter settings (from TradingView)
        self.use_volatility_filter = True
        self.use_regime_filter = True
        self.use_adx_filter = True
        self.regime_threshold = -0.1
        self.adx_threshold = 20.0

        # Kernel settings (from TradingView)
        self.use_kernel_smoothing = True
        self.smoothing_lag = 2
        self.regression_level = 25
        self.relative_weighting = 8.0

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(6, hidden_size),  # 6 inputs: RSI14, WT1, WT2, CCI20, ADX20, RSI9
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ).to(self.device)

    def calculate_sma(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate Simple Moving Average with proper padding"""
        # Pad the input to maintain size
        x_padded = torch.cat([x[:1].repeat(window - 1), x])

        # Calculate SMA
        sma = F.avg_pool1d(
            x_padded.unsqueeze(0).unsqueeze(0), kernel_size=window, stride=1
        ).squeeze()

        return sma

    def calculate_atr(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
        period: int = 14,
    ) -> torch.Tensor:
        """Calculate Average True Range for volatility filter"""
        high_low = high - low
        high_close = torch.abs(high - F.pad(close, (1, 0))[:-1])
        low_close = torch.abs(low - F.pad(close, (1, 0))[:-1])

        tr = torch.maximum(high_low, torch.maximum(high_close, low_close))
        atr = torch.zeros_like(tr)

        # Calculate initial value
        if len(tr) >= period:
            atr[period - 1] = torch.mean(tr[:period])
            # Calculate subsequent values
            for i in range(period, len(tr)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    def calculate_rolling_std(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling standard deviation"""
        # Pad with edge values
        x_padded = torch.cat([x[:1].repeat(window - 1), x])

        # Calculate rolling mean
        rolling_mean = F.avg_pool1d(
            x_padded.unsqueeze(0).unsqueeze(0), kernel_size=window, stride=1
        ).squeeze()

        # Calculate squared deviations
        squared_diff = (x - rolling_mean) ** 2

        # Pad squared differences
        squared_diff_padded = torch.cat(
            [squared_diff[:1].repeat(window - 1), squared_diff]
        )

        # Calculate variance
        variance = F.avg_pool1d(
            squared_diff_padded.unsqueeze(0).unsqueeze(0), kernel_size=window, stride=1
        ).squeeze()

        return torch.sqrt(variance + 1e-8)  # Add small epsilon for numerical stability

    def match_size(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Match tensor size by repeating last value"""
        if len(x) < target_len:
            return torch.cat([x, x[-1].repeat(target_len - len(x))])
        return x[:target_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        # Extract price data
        high = x[:, 1]
        low = x[:, 2]
        close = x[:, 3]

        # Calculate all indicators
        rsi14_values = self.rsi14.forward(close)
        wt_values = self.wt.forward(high, low, close)
        cci_values = self.cci.forward(high, low, close)
        adx_values = self.adx.forward(high, low, close)
        rsi9_values = self.rsi9.forward(close)

        # Create feature tensor
        features = torch.stack(
            [
                rsi14_values["rsi"],
                wt_values["wt1"],
                wt_values["wt2"],
                cci_values["cci"],
                adx_values["adx"],
                rsi9_values["rsi"],
            ],
            dim=1,
        )

        # Extract features
        extracted = self.feature_extractor(features)

        # Store pattern if memory isn't full
        if len(self.pattern_memory) < self.history_size:
            self.pattern_memory.append(extracted[-1].detach())
            self.pattern_labels.append(1 if extracted[-1].mean() > 0 else -1)

        # Get k-NN prediction if we have enough patterns
        if len(self.pattern_memory) >= self.k_neighbors:
            memory_tensor = torch.stack(self.pattern_memory)
            knn_signal = self.knn_predict(extracted, memory_tensor)
            combined_signal = 0.6 * extracted.mean(dim=1) + 0.4 * knn_signal
        else:
            combined_signal = extracted.mean(dim=1)

        # Apply kernel smoothing if enabled
        if self.use_kernel_smoothing:
            smoothed_signal = self.calculate_sma(combined_signal, self.smoothing_lag)
            combined_signal = smoothed_signal

        # Apply filters
        if self.use_volatility_filter:
            # Calculate returns and volatility
            returns = torch.diff(close)
            returns = F.pad(returns, (1, 0), mode="constant", value=0)
            volatility = self.calculate_rolling_std(returns, window=20)
            volatility_mask = volatility > torch.mean(volatility)
            # Ensure same size as combined_signal
            volatility_mask = self.match_size(volatility_mask, len(combined_signal))
            combined_signal = combined_signal * volatility_mask.float()

        if self.use_regime_filter:
            # Simple trend detection using price vs SMA
            sma = self.calculate_sma(close, 50)
            regime = close > sma
            regime_mask = regime.float()
            # Ensure same size as combined_signal
            regime_mask = self.match_size(regime_mask, len(combined_signal))
            combined_signal = combined_signal * regime_mask

        if self.use_adx_filter:
            # Use ADX for trend strength filtering
            adx_mask = (adx_values["adx"] > self.adx_threshold).float()
            # Ensure same size as combined_signal
            adx_mask = self.match_size(adx_mask, len(combined_signal))
            combined_signal = combined_signal * adx_mask

        return combined_signal

    def find_k_nearest(self, features: torch.Tensor) -> torch.Tensor:
        """Find k-nearest neighbors and their labels"""
        if len(self.pattern_memory) < self.k_neighbors:
            return torch.zeros(len(features), device=self.device)

        # Convert memory to tensor
        memory_tensor = torch.stack(self.pattern_memory)
        labels_tensor = torch.tensor(self.pattern_labels, device=self.device)

        # Calculate distances using cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        memory_norm = F.normalize(memory_tensor, p=2, dim=1)
        similarity = torch.mm(features_norm, memory_norm.t())
        distances = 1 - similarity

        # Get k nearest indices
        _, indices = torch.topk(
            distances, min(self.k_neighbors, len(memory_tensor)), dim=1, largest=False
        )

        # Weight votes by relative_weighting parameter
        weights = torch.exp(
            -distances[torch.arange(len(features)).unsqueeze(1), indices]
            * self.relative_weighting
        )
        weighted_votes = (labels_tensor[indices] * weights).sum(dim=1) / (
            weights.sum(dim=1) + 1e-8
        )

        return weighted_votes

    def generate_signals(self, x: torch.Tensor) -> torch.Tensor:
        """Generate trading signals based on model predictions"""
        # Get predictions from forward pass
        scores = self.forward(x)

        # Initialize weights for different timeframes
        self.weights = {
            "short": 0.4,  # Was 0.3
            "medium": 0.35,  # Was 0.4
            "long": 0.25,  # Was 0.3
        }

        # Initialize vote windows
        short_votes = torch.zeros_like(scores, device=self.device)
        medium_votes = torch.zeros_like(scores, device=self.device)
        long_votes = torch.zeros_like(scores, device=self.device)

        # Calculate votes for each timeframe with balanced windows
        for i in range(len(scores)):
            # Short-term: 5 periods
            start_idx = max(0, i - 5)
            short_votes[i] = scores[start_idx : i + 1].mean() * 1.3  # Was 1.2

            # Medium-term: 10 periods
            start_idx = max(0, i - 10)
            medium_votes[i] = scores[start_idx : i + 1].mean() * 1.1  # Was 1.0

            # Long-term: 20 periods
            start_idx = max(0, i - 20)
            long_votes[i] = scores[start_idx : i + 1].mean() * 0.9  # Was 0.8

        # Combine votes with timeframe weights and regime weighting
        final_votes = (
            self.weights["short"] * short_votes
            + self.weights["medium"] * medium_votes
            + self.weights["long"] * long_votes
        )

        # Convert votes to signals with more aggressive thresholds
        signals = torch.zeros_like(final_votes, device=self.device)
        signals[final_votes > 0.10] = 1  # Was 0.15
        signals[final_votes < -0.10] = -1  # Was -0.15

        return signals

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns:
        --------
        torch.Tensor
            Class probabilities of shape (batch_size, 1)
        """
        with torch.no_grad():
            return self.forward(x)

    def get_feature_importance(self, x: torch.Tensor) -> Dict[int, float]:
        """
        Calculate feature importance using gradient-based approach

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns:
        --------
        Dict[int, float]
            Dictionary mapping feature indices to importance scores
        """
        # Enable gradients
        x.requires_grad = True

        # Forward pass
        outputs = self.forward(x)

        # Backward pass
        outputs.sum().backward()

        # Get gradients
        gradients = x.grad.abs()

        # Calculate feature importance as mean gradient magnitude
        importance = gradients.mean(dim=0)

        # Create dictionary of feature indices to importance scores
        importance_dict = {i: float(score) for i, score in enumerate(importance)}

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        return importance_dict

    def calculate_features(self, data: pd.DataFrame) -> MLFeatures:
        """
        Calculate ML features from price data matching TradingView setup

        Args:
            data: DataFrame with OHLCV data

        Returns:
            MLFeatures object with calculated features
        """
        # Calculate all indicator features using same parameters as TradingView
        rsi1_features = self.rsi1.calculate_signals(data)  # RSI(14)
        wt_features = self.wt.calculate_signals(data)  # WT(10,11)
        cci_features = self.cci.calculate_signals(data)  # CCI(20)
        adx_features = self.adx.calculate_signals(data)  # ADX(20)
        rsi2_features = self.rsi2.calculate_signals(data)  # RSI(9)

        # Convert to tensors
        close = self.to_tensor(data["close"].values)
        high = self.to_tensor(data["high"].values)
        low = self.to_tensor(data["low"].values)
        volume = self.to_tensor(data["volume"].values)

        # Calculate returns
        returns = torch.diff(close) / close[:-1]
        returns = F.pad(returns, (1, 0))

        # Calculate features using convolution and combine with indicators
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            # Momentum (using RSI 14 and WT)
            momentum = rsi1_features["rsi"].float() * wt_features["wave_trend"].float()

            # Volatility (using CCI and ADX)
            volatility = cci_features["cci"].float() * adx_features["adx"].float()

            # Trend (using RSI 9)
            trend = rsi2_features["rsi"].float()

            # Volume
            vol_change = torch.diff(volume) / volume[:-1]
            vol_change = F.pad(vol_change, (1, 0))
            volume_feature = F.conv1d(
                vol_change.unsqueeze(0).unsqueeze(0),
                self.volume_calc.weight,
                padding="same",
            ).squeeze()

        return MLFeatures(
            momentum=momentum, volatility=volatility, trend=trend, volume=volume_feature
        )

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate trading signals from data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signals and predictions
        """
        # Calculate features
        features = self.calculate_features(data)

        # Generate predictions
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            results = self.forward_with_features(features)

        return results

    def update_stats(self, current_price: float, signal: int, last_signal: int) -> None:
        """Update trading statistics"""
        if last_signal != 0 and signal != last_signal:
            self.stats.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal

            if pnl > 0:
                self.stats.winning_trades += 1
                self.stats.avg_win = (
                    self.stats.avg_win * (self.stats.winning_trades - 1) + pnl
                ) / self.stats.winning_trades
            else:
                self.stats.losing_trades += 1
                self.stats.avg_loss = (
                    self.stats.avg_loss * (self.stats.losing_trades - 1) + abs(pnl)
                ) / self.stats.losing_trades

            if self.stats.total_trades > 0:
                self.stats.win_rate = (
                    self.stats.winning_trades / self.stats.total_trades
                )

            if self.stats.avg_loss > 0:
                self.stats.profit_factor = (
                    self.stats.avg_win * self.stats.winning_trades
                ) / (self.stats.avg_loss * self.stats.losing_trades)

        self.last_price = current_price

    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot signals and predictions"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

            # Plot price and signals
            ax1.plot(df.index, df["close"], label="Price", alpha=0.7)
            buy_points = df.index[signals["buy_signals"] == 1]
            sell_points = df.index[signals["sell_signals"] == 1]

            if len(buy_points) > 0:
                ax1.scatter(
                    buy_points,
                    df.loc[buy_points, "close"],
                    color="green",
                    marker="^",
                    label="Buy",
                )
            if len(sell_points) > 0:
                ax1.scatter(
                    sell_points,
                    df.loc[sell_points, "close"],
                    color="red",
                    marker="v",
                    label="Sell",
                )

            ax1.set_title("Price with Lorentzian Classifier Signals")
            ax1.legend()

            # Plot predictions
            ax2.plot(df.index, signals["predictions"], label="Prediction", color="blue")
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax2.set_title("Classifier Predictions")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting signals: {str(e)}")

    def get_stats(self) -> Dict:
        """Get current trading statistics"""
        return {
            "total_trades": self.stats.total_trades,
            "win_rate": self.stats.win_rate,
            "profit_factor": self.stats.profit_factor,
            "avg_win": self.stats.avg_win,
            "avg_loss": self.stats.avg_loss,
            "sharpe_ratio": self.stats.sharpe_ratio,
        }
