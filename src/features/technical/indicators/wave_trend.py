"""
Enhanced WaveTrend Implementation using PyTorch

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
from ..base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

@dataclass
class WaveTrendMetrics:
    """Container for WaveTrend trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

class WaveTrendIndicator(BaseTorchIndicator):
    """
    PyTorch-based WaveTrend implementation with advanced features
    """
    
    def __init__(
        self,
        channel_length: int = 10,
        average_length: int = 11,
        overbought: float = 60.0,
        oversold: float = -60.0,
        config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize WaveTrend indicator with PyTorch backend
        
        Args:
            channel_length: The channel length for initial calculations
            average_length: The average length for the wave trend
            overbought: Overbought threshold
            oversold: Oversold threshold
            config: Optional PyTorch configuration
        """
        super().__init__(config)
        
        self.channel_length = channel_length
        self.average_length = average_length
        self.overbought = overbought
        self.oversold = oversold
        
        # Trading metrics
        self.metrics = WaveTrendMetrics()
        self.last_price = None
        
    def forward(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate WaveTrend values using PyTorch operations
        
        Args:
            high: High prices tensor
            low: Low prices tensor
            close: Close prices tensor
            
        Returns:
            Dictionary with WaveTrend values and signals
        """
        # Calculate HLC3 (typical price)
        hlc3 = (high + low + close) / 3.0
        
        # Calculate ESA = EMA(HLC3, channel_length)
        esa = self.torch_ema(hlc3, alpha=2.0/(self.channel_length + 1))
        
        # Calculate absolute difference
        abs_diff = torch.abs(hlc3 - esa)
        
        # Calculate D = EMA(abs(HLC3 - ESA), channel_length)
        d = self.torch_ema(abs_diff, alpha=2.0/(self.channel_length + 1))
        
        # Calculate CI = (HLC3 - ESA) / (0.015 * D)
        ci = (hlc3 - esa) / (0.015 * d)
        
        # Calculate Wave Trend = EMA(CI, average_length)
        wt = self.torch_ema(ci, alpha=2.0/(self.average_length + 1))
        
        # Generate signals
        buy_signals = (wt < self.oversold).float()
        sell_signals = (wt > self.overbought).float()
        
        return {
            'wt': wt,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate WaveTrend and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with WaveTrend values and signals
        """
        # Convert price data to tensors
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Calculate WaveTrend and signals
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            results = self.forward(high, low, close)
        
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
        """Plot WaveTrend with signals using matplotlib"""
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
            
            ax1.set_title('Price with WaveTrend Signals')
            ax1.legend()
            
            # Plot WaveTrend
            ax2.plot(df.index, signals['wt'], label='WaveTrend', color='blue')
            ax2.axhline(y=self.overbought, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.oversold, color='g', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.set_title('WaveTrend')
            
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