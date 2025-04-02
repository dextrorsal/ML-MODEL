"""
Enhanced CCI (Commodity Channel Index) Implementation using PyTorch

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
class CciMetrics:
    """Container for CCI trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

class CciIndicator(BaseTorchIndicator):
    """
    PyTorch-based CCI implementation with advanced features
    """
    
    def __init__(
        self,
        period: int = 20,
        constant: float = 0.015,
        overbought: float = 100.0,
        oversold: float = -100.0,
        config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize CCI indicator with PyTorch backend
        
        Args:
            period: CCI calculation period
            constant: CCI constant (typically 0.015)
            overbought: Overbought threshold
            oversold: Oversold threshold
            config: Optional PyTorch configuration
        """
        super().__init__(config)
        
        self.period = period
        self.constant = constant
        self.overbought = overbought
        self.oversold = oversold
        
        # Trading metrics
        self.metrics = CciMetrics()
        self.last_price = None
        
    def forward(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate CCI values using PyTorch operations
        
        Args:
            high: High prices tensor
            low: Low prices tensor
            close: Close prices tensor
            
        Returns:
            Dictionary with CCI values and signals
        """
        # Calculate typical price
        tp = (high + low + close) / 3.0
        
        # Calculate SMA of typical price
        sma_tp = self.torch_sma(tp, self.period)
        
        # Calculate Mean Deviation
        # First create rolling windows of typical price
        rolling_tp = tp.unfold(0, self.period, 1)
        
        # Calculate absolute deviations from SMA for each window
        deviations = torch.abs(rolling_tp - sma_tp.unsqueeze(1))
        
        # Calculate mean deviation
        mean_dev = torch.mean(deviations, dim=1)
        
        # Add padding for the initial values
        padding = torch.full((self.period - 1,), torch.nan, device=self.device)
        mean_dev = torch.cat([padding, mean_dev])
        
        # Calculate CCI
        cci = (tp - sma_tp) / (self.constant * mean_dev)
        
        # Generate signals
        buy_signals = (cci < self.oversold).float()
        sell_signals = (cci > self.overbought).float()
        
        return {
            'cci': cci,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate CCI and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with CCI values and signals
        """
        # Convert price data to tensors
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Calculate CCI and signals
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
        """Plot CCI with signals using matplotlib"""
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
            
            ax1.set_title('Price with CCI Signals')
            ax1.legend()
            
            # Plot CCI
            ax2.plot(df.index, signals['cci'], label='CCI', color='blue')
            ax2.axhline(y=self.overbought, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.oversold, color='g', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.set_title('CCI')
            
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