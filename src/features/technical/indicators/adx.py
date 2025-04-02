"""
ADX (Average Directional Index) Implementation using PyTorch

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
class AdxMetrics:
    """Container for ADX trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

class ADXIndicator(BaseTorchIndicator):
    """
    PyTorch-based ADX implementation matching TradingView
    """
    
    def __init__(
        self,
        period: int = 14,
        smoothing: int = 14,
        threshold: float = 25.0,
        config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize ADX indicator with PyTorch backend
        
        Args:
            period: ADX calculation period
            smoothing: Smoothing period for DI calculations
            threshold: ADX threshold for signal generation
            config: Optional PyTorch configuration
        """
        super().__init__(config)
        
        self.period = period
        self.smoothing = smoothing
        self.threshold = threshold
        
        # Trading metrics
        self.metrics = AdxMetrics()
        
    def forward(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate ADX values using PyTorch operations
        
        Args:
            high: High prices tensor
            low: Low prices tensor
            close: Close prices tensor
            
        Returns:
            Dictionary with ADX values and signals
        """
        # Calculate True Range
        prev_close = torch.roll(close, 1)
        tr = torch.max(
            torch.max(
                high - low,
                torch.abs(high - prev_close)
            ),
            torch.abs(low - prev_close)
        )
        
        # Calculate directional movement
        pos_dm = high - torch.roll(high, 1)
        neg_dm = torch.roll(low, 1) - low
        
        pos_dm = torch.where(
            (pos_dm > neg_dm) & (pos_dm > 0),
            pos_dm,
            torch.zeros_like(pos_dm)
        )
        neg_dm = torch.where(
            (neg_dm > pos_dm) & (neg_dm > 0),
            neg_dm,
            torch.zeros_like(neg_dm)
        )
        
        # Smooth the TR and DM
        tr_smooth = self.torch_ema(tr, 2.0 / (self.smoothing + 1))
        pos_dm_smooth = self.torch_ema(pos_dm, 2.0 / (self.smoothing + 1))
        neg_dm_smooth = self.torch_ema(neg_dm, 2.0 / (self.smoothing + 1))
        
        # Calculate the Directional Indexes
        pdi = 100 * pos_dm_smooth / (tr_smooth + 1e-8)
        ndi = 100 * neg_dm_smooth / (tr_smooth + 1e-8)
        
        # Calculate the Directional Index
        dx = 100 * torch.abs(pdi - ndi) / (pdi + ndi + 1e-8)
        
        # Calculate ADX
        adx = self.torch_ema(dx, 2.0 / (self.period + 1))
        
        # Generate signals based on ADX strength
        strong_trend = (adx > self.threshold).float()
        
        return {
            'adx': adx,
            'pdi': pdi,
            'ndi': ndi,
            'strong_trend': strong_trend
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate ADX and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with ADX values and signals
        """
        # Convert price data to tensors
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Calculate ADX and signals
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
        """Plot ADX with signals"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
            
            # Plot price
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            ax1.set_title('Price with ADX Signals')
            ax1.legend()
            
            # Plot ADX
            ax2.plot(df.index, signals['adx'], label='ADX', color='blue')
            ax2.plot(df.index, signals['pdi'], label='+DI', color='green')
            ax2.plot(df.index, signals['ndi'], label='-DI', color='red')
            ax2.axhline(y=self.threshold, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('ADX')
            ax2.legend()
            
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