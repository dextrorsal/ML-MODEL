"""
Enhanced RSI (Relative Strength Index) Implementation using PyTorch

Features:
- GPU acceleration for faster calculations
- Real-time signal generation
- Configurable parameters
- Advanced signal filtering
- Backtesting metrics
- Debug capabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Union
from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

@dataclass
class RsiMetrics:
    """Container for RSI trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

class RsiIndicator(BaseTorchIndicator):
    """
    PyTorch-based RSI implementation with advanced features
    """
    
    def __init__(
        self,
        period: int = 14,
        smoothing: int = 1,
        overbought: float = 70.0,
        oversold: float = 30.0,
        config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize RSI indicator with PyTorch backend
        
        Args:
            period: RSI calculation period
            smoothing: Smoothing factor for RSI
            overbought: Overbought threshold
            oversold: Oversold threshold
            config: Optional PyTorch configuration
        """
        super().__init__(config)
        
        self.period = period
        self.smoothing = smoothing
        self.overbought = overbought
        self.oversold = oversold
        
        # Trading metrics
        self.metrics = RsiMetrics()
        
        # Initialize learnable parameters if needed
        self.alpha = nn.Parameter(torch.tensor(2.0 / (period + 1)))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate RSI values using PyTorch operations
        
        Args:
            x: Input price tensor
            
        Returns:
            Dictionary with RSI values and signals
        """
        # Calculate price changes
        delta = x.diff()
        
        # Separate gains and losses
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Calculate smoothed averages using EMA
        avg_gains = self.torch_ema(gains, self.alpha.item())
        avg_losses = self.torch_ema(losses, self.alpha.item())
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-8)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        buy_signals = (rsi < self.oversold).float()
        sell_signals = (rsi > self.overbought).float()
        
        return {
            'rsi': rsi,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate RSI and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with RSI values and signals
        """
        # Convert price data to tensor
        close_prices = self.to_tensor(data['close'])
        
        # Calculate RSI and signals
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            results = self.forward(close_prices)
        
        return results
    
    def update_metrics(self, current_price: float, signal: int, last_signal: int) -> None:
        """Update trading metrics"""
        if last_signal != 0 and signal != last_signal:
            self.metrics.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal
            
            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = ((self.metrics.avg_win * 
                    (self.metrics.winning_trades - 1) + pnl) / 
                    self.metrics.winning_trades)
            else:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = ((self.metrics.avg_loss * 
                    (self.metrics.losing_trades - 1) + abs(pnl)) / 
                    self.metrics.losing_trades)
            
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (self.metrics.winning_trades / 
                    self.metrics.total_trades)
                
            if self.metrics.avg_loss > 0:
                self.metrics.profit_factor = ((self.metrics.avg_win * 
                    self.metrics.winning_trades) / 
                    (self.metrics.avg_loss * self.metrics.losing_trades))
        
        self.last_price = current_price
    
    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot RSI with signals using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
            
            # Plot price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            buy_points = df.index[signals['buy_signals'] == 1]
            sell_points = df.index[signals['sell_signals'] == 1]
            
            if len(buy_points) > 0:
                ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                          color='green', marker='^', label='Buy')
            if len(sell_points) > 0:
                ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
                          color='red', marker='v', label='Sell')
            
            ax1.set_title('Price with RSI Signals')
            ax1.legend()
            
            # Plot RSI
            ax2.plot(df.index, signals['rsi'], label='RSI', color='blue')
            ax2.axhline(y=self.overbought, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.oversold, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI')
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting signals: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get current trading metrics"""
        return {
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'profit_factor': self.metrics.profit_factor,
            'avg_win': self.metrics.avg_win,
            'avg_loss': self.metrics.avg_loss
        } 