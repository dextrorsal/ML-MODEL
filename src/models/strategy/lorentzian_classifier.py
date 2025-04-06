"""
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
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from ....features.technical.indicators.base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig
from ....features.technical.indicators.rsi import RSIIndicator
from ....features.technical.indicators.cci import CCIIndicator
from ....features.technical.indicators.wave_trend import WaveTrendIndicator
from ....features.technical.indicators.adx import ADXIndicator
from contextlib import nullcontext

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
        lorentzian_distance = torch.log(1 + squared_distance / (self.sigma ** 2))
        
        return lorentzian_distance

class LorentzianClassifier(nn.Module):
    """
    Neural network classifier using Lorentzian distance for signal generation
    """
    def __init__(self, input_size: int = 18, hidden_size: int = 64, 
                 dropout_rate: float = 0.2, sigma: float = 1.0):
        """
        Initialize the Lorentzian Classifier
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Size of hidden layers
        dropout_rate : float
            Dropout rate for regularization
        sigma : float
            Sigma parameter for Lorentzian distance
        """
        super(LorentzianClassifier, self).__init__()
        
        # Save parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.sigma = sigma
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Prototype vectors (learnable centroids)
        self.positive_prototype = nn.Parameter(torch.randn(hidden_size))
        self.negative_prototype = nn.Parameter(torch.randn(hidden_size))
        
        # Lorentzian distance layer
        self.lorentzian = LorentzianDistance(sigma=sigma)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(2, 1),  # 2 distances (to positive and negative prototypes)
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output probabilities of shape (batch_size, 1)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Calculate distances to prototypes
        positive_distances = self.lorentzian(attended_features, 
                                            self.positive_prototype.expand(attended_features.size(0), -1))
        negative_distances = self.lorentzian(attended_features, 
                                            self.negative_prototype.expand(attended_features.size(0), -1))
        
        # Concatenate distances
        distances = torch.stack([positive_distances, negative_distances], dim=1)
        
        # Final classification
        output = self.classifier(distances)
        
        return output

    # The original forward method using MLFeatures is moved to this method for backward compatibility
    def forward_with_features(self, features: MLFeatures) -> Dict[str, torch.Tensor]:
        """
        Forward pass for feature-based input
        
        Parameters:
        -----------
        features : MLFeatures
            Feature container with momentum, volatility, trend, and volume
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary with output signals and metrics
        """
        # Initialize results dictionary
        results = {}
        
        # Process features
        feature_length = features.momentum.shape[0]
        
        # Concatenate all features
        combined_features = torch.cat([
            features.momentum.view(feature_length, -1),
            features.volatility.view(feature_length, -1),
            features.trend.view(feature_length, -1),
            features.volume.view(feature_length, -1) if features.volume is not None else torch.zeros(feature_length, 1)
        ], dim=1)
        
        # Calculate output probability
        results['probability'] = self.forward(combined_features)
        
        # Generate signal
        results['signal'] = torch.zeros_like(results['probability'])
        results['signal'][results['probability'] > 0.5] = 1.0  # Long signal
        results['signal'][results['probability'] < 0.5] = -1.0  # Short signal
        
        return results
    
    def generate_signals(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Generate trading signals from model predictions
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)
        threshold : float
            Probability threshold for generating signals
            
        Returns:
        --------
        torch.Tensor
            Trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Get predictions
        with torch.no_grad():
            predictions = self.forward(x)
        
        # Generate signals
        signals = torch.zeros_like(predictions)
        signals[predictions > threshold + 0.1] = 1  # Strong buy
        signals[predictions < threshold - 0.1] = -1  # Strong sell
        
        return signals.squeeze()
    
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
        importance_dict = dict(sorted(importance_dict.items(), 
                                      key=lambda item: item[1], 
                                      reverse=True))
        
        return importance_dict

    def calculate_features(
        self,
        data: pd.DataFrame
    ) -> MLFeatures:
        """
        Calculate ML features from price data matching TradingView setup
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            MLFeatures object with calculated features
        """
        # Calculate all indicator features using same parameters as TradingView
        rsi1_features = self.rsi1.calculate_signals(data)    # RSI(14)
        wt_features = self.wt.calculate_signals(data)        # WT(10,11)
        cci_features = self.cci.calculate_signals(data)      # CCI(20)
        adx_features = self.adx.calculate_signals(data)      # ADX(20)
        rsi2_features = self.rsi2.calculate_signals(data)    # RSI(9)
        
        # Convert to tensors
        close = self.to_tensor(data['close'].values)
        high = self.to_tensor(data['high'].values)
        low = self.to_tensor(data['low'].values)
        volume = self.to_tensor(data['volume'].values)
        
        # Calculate returns
        returns = torch.diff(close) / close[:-1]
        returns = F.pad(returns, (1, 0))
        
        # Calculate features using convolution and combine with indicators
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            # Momentum (using RSI 14 and WT)
            momentum = (rsi1_features['rsi'].float() * wt_features['wave_trend'].float())
            
            # Volatility (using CCI and ADX)
            volatility = (cci_features['cci'].float() * adx_features['adx'].float())
            
            # Trend (using RSI 9)
            trend = rsi2_features['rsi'].float()
            
            # Volume
            vol_change = torch.diff(volume) / volume[:-1]
            vol_change = F.pad(vol_change, (1, 0))
            volume_feature = F.conv1d(
                vol_change.unsqueeze(0).unsqueeze(0),
                self.volume_calc.weight,
                padding='same'
            ).squeeze()
        
        return MLFeatures(
            momentum=momentum,
            volatility=volatility,
            trend=trend,
            volume=volume_feature
        )
        
    def calculate_signals(
        self,
        data: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
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
    
    def update_stats(
        self,
        current_price: float,
        signal: int,
        last_signal: int
    ) -> None:
        """Update trading statistics"""
        if last_signal != 0 and signal != last_signal:
            self.stats.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal
            
            if pnl > 0:
                self.stats.winning_trades += 1
                self.stats.avg_win = (
                    (self.stats.avg_win * (self.stats.winning_trades - 1) + pnl) /
                    self.stats.winning_trades
                )
            else:
                self.stats.losing_trades += 1
                self.stats.avg_loss = (
                    (self.stats.avg_loss * (self.stats.losing_trades - 1) + abs(pnl)) /
                    self.stats.losing_trades
                )
            
            if self.stats.total_trades > 0:
                self.stats.win_rate = (
                    self.stats.winning_trades /
                    self.stats.total_trades
                )
                
            if self.stats.avg_loss > 0:
                self.stats.profit_factor = (
                    (self.stats.avg_win * self.stats.winning_trades) /
                    (self.stats.avg_loss * self.stats.losing_trades)
                )
        
        self.last_price = current_price
    
    def plot_signals(
        self,
        df: pd.DataFrame,
        signals: Dict[str, pd.Series]
    ) -> None:
        """Plot signals and predictions"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(15, 10),
                height_ratios=[2, 1]
            )
            
            # Plot price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            buy_points = df.index[signals['buy_signals'] == 1]
            sell_points = df.index[signals['sell_signals'] == 1]
            
            if len(buy_points) > 0:
                ax1.scatter(
                    buy_points,
                    df.loc[buy_points, 'close'],
                    color='green',
                    marker='^',
                    label='Buy'
                )
            if len(sell_points) > 0:
                ax1.scatter(
                    sell_points,
                    df.loc[sell_points, 'close'],
                    color='red',
                    marker='v',
                    label='Sell'
                )
            
            ax1.set_title('Price with Lorentzian Classifier Signals')
            ax1.legend()
            
            # Plot predictions
            ax2.plot(
                df.index,
                signals['predictions'],
                label='Prediction',
                color='blue'
            )
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_title('Classifier Predictions')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting signals: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get current trading statistics"""
        return {
            'total_trades': self.stats.total_trades,
            'win_rate': self.stats.win_rate,
            'profit_factor': self.stats.profit_factor,
            'avg_win': self.stats.avg_win,
            'avg_loss': self.stats.avg_loss,
            'sharpe_ratio': self.stats.sharpe_ratio
        } 