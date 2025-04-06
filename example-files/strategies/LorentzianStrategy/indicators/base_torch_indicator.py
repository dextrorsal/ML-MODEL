"""
BASE COMPONENT: PyTorch Technical Indicators Foundation

This module provides the base class for all PyTorch-based technical indicators used in the trading system.
It implements common functionality like tensor conversion, device management (CPU/GPU support),
and automatic mixed precision (AMP) for improved performance.

All technical indicators in the system inherit from this base class to ensure consistent
behavior and compatible interfaces. The class follows a modular design pattern that allows
easy extension with new indicators while maintaining the same API.

Key features:
- GPU acceleration with automatic fallback to CPU
- Tensor conversion utilities for various input types
- Common utility functions like EMA calculation
- Consistent interface for all indicators
- Error handling and type safety
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from dataclasses import dataclass
from contextlib import nullcontext

@dataclass
class TorchIndicatorConfig:
    """Base configuration for torch indicators"""
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_amp: bool = False

class BaseTorchIndicator(nn.Module):
    """Base class for all PyTorch-based indicators"""
    
    def __init__(self, config: Optional[TorchIndicatorConfig] = None):
        """Initialize base indicator"""
        super().__init__()
        self.config = config or TorchIndicatorConfig()
        
        # Setup device and data type
        self.device = (
            torch.device(config.device) if config.device 
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = config.dtype or torch.float32
        self.use_amp = config.use_amp
        
    def to_tensor(
        self, 
        data: Union[np.ndarray, pd.Series, torch.Tensor]
    ) -> torch.Tensor:
        """Convert input data to tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.dtype)
        elif isinstance(data, pd.Series):
            return torch.tensor(
                data.values, 
                device=self.device, 
                dtype=self.dtype
            )
        else:
            return torch.tensor(
                data, 
                device=self.device, 
                dtype=self.dtype
            )
            
    def torch_ema(self, data: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
        """Calculate EMA using PyTorch"""
        # Initialize with the first value
        ema = torch.zeros_like(data)
        ema[0] = data[0]
        
        # Calculate EMA
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def calculate_signals(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Calculate indicator signals. Must be implemented by child classes.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated signals
        """
        raise NotImplementedError(
            "Subclasses must implement calculate_signals"
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for the indicator. Must be implemented by child classes.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of calculated values
        """
        raise NotImplementedError(
            "Subclasses must implement forward"
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Main calculation method that handles data conversion and processing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated values as pandas Series
        """
        try:
            # Convert to tensors and calculate
            with (
                torch.cuda.amp.autocast() 
                if self.use_amp 
                else nullcontext()
            ):
                signals = self.calculate_signals(data)
            
            # Convert back to pandas
            return {
                k: pd.Series(v.cpu().numpy(), index=data.index) 
                for k, v in signals.items()
            }
            
        except Exception as e:
            print(f"Error in indicator calculation: {str(e)}")
            return {} 