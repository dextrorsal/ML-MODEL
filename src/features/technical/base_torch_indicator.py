"""
Base PyTorch Indicator Class

This module provides a base class for all PyTorch-based technical indicators.
Features:
- GPU acceleration
- Batch processing
- Automatic differentiation
- Memory efficient operations
- Real-time signal generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple
from dataclasses import dataclass
from contextlib import nullcontext

@dataclass
class TorchIndicatorConfig:
    """Configuration for PyTorch indicators"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32
    batch_size: int = 128
    use_amp: bool = True  # Automatic Mixed Precision for faster computation
    
class BaseTorchIndicator(nn.Module):
    """Base class for all PyTorch-based indicators"""
    
    def __init__(self, config: Optional[TorchIndicatorConfig] = None):
        super().__init__()
        self.config = config or TorchIndicatorConfig()
        self.device = torch.device(self.config.device)
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None
        
    def to_tensor(self, data: Union[np.ndarray, pd.Series, torch.Tensor]) -> torch.Tensor:
        """Convert input data to PyTorch tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.config.dtype)
        elif isinstance(data, pd.Series):
            return torch.tensor(data.values, device=self.device, dtype=self.config.dtype)
        else:
            return torch.tensor(data, device=self.device, dtype=self.config.dtype)
            
    @staticmethod
    def torch_sma(x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-accelerated Simple Moving Average"""
        if len(x) < window:
            return torch.full_like(x, torch.nan)
            
        return F.avg_pool1d(
            x.view(1, 1, -1),
            kernel_size=window,
            stride=1,
            padding=window//2
        ).view(-1)
    
    @staticmethod
    def torch_ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
        """GPU-accelerated Exponential Moving Average"""
        if len(x) == 0:
            return x
            
        # Initialize weights for EMA
        weights = (1 - alpha) ** torch.arange(len(x), device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        
        # Use convolution for efficient calculation
        return F.conv1d(
            x.view(1, 1, -1),
            weights.view(1, 1, -1),
            padding=len(x)-1
        ).view(-1)
    
    @staticmethod
    def torch_stddev(x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-accelerated Rolling Standard Deviation"""
        if len(x) < window:
            return torch.full_like(x, torch.nan)
            
        # Use unfold for rolling window
        rolling = x.unfold(0, window, 1)
        return torch.std(rolling, dim=1)
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate indicator signals. Must be implemented by child classes.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated signals
        """
        raise NotImplementedError("Subclasses must implement calculate_signals")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the indicator. Must be implemented by child classes.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of calculated values
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Main calculation method that handles data conversion and processing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated values as pandas Series
        """
        try:
            # Convert to tensors and calculate
            with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
                signals = self.calculate_signals(data)
            
            # Convert back to pandas
            return {
                k: pd.Series(v.cpu().numpy(), index=data.index) 
                for k, v in signals.items()
            }
            
        except Exception as e:
            print(f"Error in indicator calculation: {str(e)}")
            return {} 