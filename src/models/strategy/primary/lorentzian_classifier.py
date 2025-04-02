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

class LorentzianClassifier(BaseTorchIndicator):
    """
    Enhanced Lorentzian Classifier with PyTorch backend.
    
    This classifier combines traditional technical analysis with modern machine learning,
    using Lorentzian kernels for robust signal generation.
    """
    
    def __init__(
        self,
        config: Optional[LorentzianSettings] = None,
        torch_config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize the classifier with configuration settings.
        
        Args:
            config: Lorentzian classifier specific configuration
            torch_config: PyTorch configuration for GPU/CPU
        """
        super().__init__(torch_config)
        
        self.config = config or LorentzianSettings()
        self.stats = TradeStats()
        self.features = MLFeatures()
        
        # Initialize indicators exactly as in TradingView
        self.rsi1 = RSIIndicator(period=14, config=torch_config)  # Feature 1: RSI(14)
        self.wt = WaveTrendIndicator(config=torch_config)         # Feature 2: WT(10,11)
        self.cci = CCIIndicator(period=20, config=torch_config)   # Feature 3: CCI(20)
        self.adx = ADXIndicator(period=20, config=torch_config)   # Feature 4: ADX(20)
        self.rsi2 = RSIIndicator(period=9, config=torch_config)   # Feature 5: RSI(9)
        
        # Initialize kernels
        self.momentum_kernel = self._create_lorentzian_kernel(
            self.config.kernel_size,
            self.config.kernel_std
        )
        
        # Initialize feature calculators
        self._init_feature_calculators()
        
    def _init_feature_calculators(self):
        """Initialize feature calculation components"""
        self.momentum_calc = nn.Conv1d(
            1, 1, self.config.momentum_lookback,
            padding='same',
            bias=False
        )
        self.volatility_calc = nn.Conv1d(
            1, 1, self.config.volatility_lookback,
            padding='same',
            bias=False
        )
        self.trend_calc = nn.Conv1d(
            1, 1, self.config.trend_lookback,
            padding='same',
            bias=False
        )
        self.volume_calc = nn.Conv1d(
            1, 1, self.config.volume_lookback,
            padding='same',
            bias=False
        )
        
        # Move to device
        self.momentum_calc.to(self.device)
        self.volatility_calc.to(self.device)
        self.trend_calc.to(self.device)
        self.volume_calc.to(self.device)
        
    def _create_lorentzian_kernel(
        self,
        size: int,
        std: float
    ) -> torch.Tensor:
        """Create Lorentzian kernel for feature calculation"""
        x = torch.linspace(-size//2, size//2, size)
        kernel = 1 / (1 + (x/std)**2)
        return kernel.to(self.device)
        
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
        
    def forward(
        self,
        features: MLFeatures
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the classifier
        
        Args:
            features: MLFeatures object with input features
            
        Returns:
            Dictionary with predictions and signals
        """
        # Ensure kernel is correctly broadcast to match feature size
        # The kernel should be small (e.g., size 3) and needs to be broadcast to match feature size (100)
        feature_length = features.momentum.shape[0]
        
        # Either broadcast the kernel or apply it individually to each element
        # Option 1: Create a weighted sum for each element based on the kernel
        combined = torch.zeros_like(features.momentum, device=self.device)
        
        # Apply feature weights
        combined = (
            features.momentum * self.config.momentum_lookback / 100 +
            features.volatility * self.config.volatility_threshold +
            features.trend * self.config.regime_threshold +
            features.volume * self.config.adx_threshold
        )
        
        # Generate signals
        buy_signals = (combined > 0).float()
        sell_signals = (combined < 0).float()
        
        return {
            'predictions': combined,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'long_signals': buy_signals,  # Alias for test compatibility
            'short_signals': sell_signals  # Alias for test compatibility
        }
    
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
            results = self.forward(features)
        
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