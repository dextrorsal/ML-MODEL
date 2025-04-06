# 💎 Style Guide

This guide outlines coding standards and best practices for our GPU-accelerated trading system.

## 📋 Table of Contents
1. [Python Style](#python-style)
2. [Documentation](#documentation)
3. [Type Hints](#type-hints)
4. [GPU Optimization](#gpu-optimization)
5. [Error Handling](#error-handling)
6. [Testing](#testing)
7. [Performance](#performance)

## 🐍 Python Style

### 1. General Guidelines
- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable names
- Keep functions focused and small

### 2. Imports
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import torch
import pandas as pd

# Local imports
from features.base import BaseFeature
from models.classifier import LorentzianClassifier
from utils.optimization import optimize_tensor
```

### 3. Class Structure
```python
class FeatureCalculator:
    """
    Calculate technical indicators with GPU acceleration.
    
    Attributes
    ----------
    name : str
        Feature identifier
    period : int
        Calculation period
    device : torch.device
        Computation device
    """
    
    def __init__(
        self,
        name: str,
        period: int = 14,
        device: Optional[torch.device] = None
    ):
        self.name = name
        self.period = period
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate feature values."""
        return self._process(data)
        
    def _process(self, data: torch.Tensor) -> torch.Tensor:
        """Internal processing logic."""
        pass
```

### 4. Function Structure
```python
def calculate_momentum(
    prices: torch.Tensor,
    period: int = 10,
    smoothing: int = 3
) -> torch.Tensor:
    """
    Calculate price momentum with optional smoothing.
    
    Parameters
    ----------
    prices : torch.Tensor
        Input price data
    period : int, optional
        Lookback period, by default 10
    smoothing : int, optional
        Smoothing period, by default 3
        
    Returns
    -------
    torch.Tensor
        Momentum indicator values
        
    Examples
    --------
    >>> prices = torch.randn(100)
    >>> momentum = calculate_momentum(prices, period=14)
    """
    # Input validation
    if not isinstance(prices, torch.Tensor):
        raise TypeError("prices must be a torch.Tensor")
    
    # Calculation
    momentum = prices / prices.roll(period) - 1
    
    # Optional smoothing
    if smoothing > 1:
        kernel = torch.ones(smoothing) / smoothing
        momentum = torch.conv1d(
            momentum.view(1, 1, -1),
            kernel.view(1, 1, -1),
            padding=smoothing-1
        ).view(-1)
    
    return momentum
```

## 📚 Documentation

### 1. Module Documentation
```python
"""
Technical indicators and features for trading.

This module implements various technical indicators optimized for GPU computation.
Features are designed to work with the LorentzianClassifier for signal generation.

Classes
-------
RSIFeature
    Relative Strength Index calculation
MomentumFeature
    Price momentum calculation
VolatilityFeature
    Price volatility estimation

Functions
---------
calculate_rsi
    Calculate RSI values
calculate_momentum
    Calculate momentum values
calculate_volatility
    Calculate volatility values
"""
```

### 2. Class Documentation
```python
class RSIFeature(BaseFeature):
    """
    Relative Strength Index (RSI) calculation.
    
    The RSI is a momentum indicator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions.
    
    Parameters
    ----------
    period : int, optional
        Lookback period for RSI calculation, by default 14
    device : torch.device, optional
        Computation device, by default None (auto-select)
        
    Attributes
    ----------
    name : str
        Feature identifier
    requires_grad : bool
        Whether the feature requires gradient computation
        
    Methods
    -------
    forward(data)
        Calculate RSI values from input data
    to(device)
        Move the feature to specified device
        
    Examples
    --------
    >>> rsi = RSIFeature(period=14)
    >>> prices = torch.randn(100)
    >>> values = rsi(prices)
    """
```

### 3. Function Documentation
```python
def optimize_batch_size(
    data_size: int,
    model_params: int,
    memory_limit: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on GPU memory.
    
    This function estimates the maximum batch size that can be processed
    given the data size, model parameters, and available GPU memory.
    
    Parameters
    ----------
    data_size : int
        Size of input data samples
    model_params : int
        Number of model parameters
    memory_limit : float, optional
        Maximum fraction of GPU memory to use, by default 0.8
        
    Returns
    -------
    int
        Optimal batch size
        
    Notes
    -----
    The calculation assumes FP32 precision. For FP16, the memory
    requirements will be approximately halved.
    
    Examples
    --------
    >>> optimal_batch = optimize_batch_size(1000, 1000000)
    >>> print(f"Optimal batch size: {optimal_batch}")
    """
```

## 🎯 Type Hints

### 1. Basic Types
```python
def process_data(
    data: torch.Tensor,
    window: int = 10
) -> torch.Tensor:
    """Process data with sliding window."""
    pass

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Calculate performance metrics."""
    pass
```

### 2. Complex Types
```python
from typing import TypeVar, Generic, Sequence

T = TypeVar('T', bound=torch.Tensor)

class DataProcessor(Generic[T]):
    def process(self, data: Sequence[T]) -> T:
        pass

def batch_process(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_size: int
) -> List[torch.Tensor]:
    pass
```

### 3. Custom Types
```python
from typing import NewType, Tuple

Price = NewType('Price', float)
TimeWindow = NewType('TimeWindow', int)

def calculate_returns(
    prices: Sequence[Price],
    window: TimeWindow
) -> Tuple[float, float]:
    """Calculate return metrics."""
    pass
```

## ⚡ GPU Optimization

### 1. Memory Management
```python
def efficient_calculation(data: torch.Tensor) -> torch.Tensor:
    # Move data to GPU once
    data = data.to(device)
    
    # Pre-allocate output
    result = torch.zeros_like(data)
    
    # In-place operations
    result.copy_(data)
    result.mul_(2.0)
    
    return result
```

### 2. Batch Processing
```python
def process_large_dataset(
    data: torch.Tensor,
    batch_size: int = 1024
) -> torch.Tensor:
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        result = process_batch(batch)
        results.append(result)
    
    return torch.cat(results)
```

### 3. Optimization Techniques
```python
def optimized_feature(data: torch.Tensor) -> torch.Tensor:
    # Use vectorized operations
    diff = data.diff()
    
    # Avoid unnecessary copies
    with torch.no_grad():
        result = torch.zeros_like(data)
        mask = diff > 0
        result[mask] = diff[mask]
    
    return result
```

## 🛡️ Error Handling

### 1. Input Validation
```python
def validate_inputs(
    data: torch.Tensor,
    period: int
) -> None:
    """Validate input parameters."""
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor")
    
    if data.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {data.dim()}D")
        
    if period < 1:
        raise ValueError(f"period must be positive, got {period}")
        
    if data.size(1) < period:
        raise ValueError(
            f"Input length {data.size(1)} must be >= period {period}"
        )
```

### 2. Error Messages
```python
class FeatureError(Exception):
    """Base class for feature calculation errors."""
    pass

class InvalidInputError(FeatureError):
    """Raised when input data is invalid."""
    pass

class CalculationError(FeatureError):
    """Raised when feature calculation fails."""
    pass

def calculate_feature(data: torch.Tensor) -> torch.Tensor:
    try:
        result = perform_calculation(data)
    except RuntimeError as e:
        raise CalculationError(f"Feature calculation failed: {str(e)}")
    return result
```

## 🧪 Testing

### 1. Unit Tests
```python
def test_rsi_calculation():
    """Test RSI feature calculation."""
    # Setup
    rsi = RSIFeature(period=14)
    data = torch.randn(100)
    
    # Execute
    result = rsi(data)
    
    # Assert
    assert isinstance(result, torch.Tensor)
    assert result.shape == data.shape
    assert (result >= 0).all() and (result <= 100).all()
    assert not torch.isnan(result).any()
```

### 2. Performance Tests
```python
def test_processing_speed():
    """Test feature processing performance."""
    # Setup
    data = torch.randn(10000, 10)
    
    # Time execution
    start = time.time()
    result = process_features(data)
    duration = time.time() - start
    
    # Assert
    assert duration < MAX_PROCESSING_TIME
```

## 📈 Performance

### 1. Memory Efficiency
```python
def memory_efficient_operation(
    data: torch.Tensor
) -> torch.Tensor:
    # Track memory usage
    initial_memory = torch.cuda.memory_allocated()
    
    # Perform operation
    result = process_data(data)
    
    # Check memory usage
    final_memory = torch.cuda.memory_allocated()
    memory_used = final_memory - initial_memory
    
    logger.debug(f"Memory used: {memory_used / 1024**2:.2f} MB")
    return result
```

### 2. Profiling
```python
@profile
def optimize_calculation(data: torch.Tensor) -> torch.Tensor:
    # Pre-allocate output
    result = torch.zeros_like(data)
    
    # Process in chunks
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result[i:i+chunk_size] = process_chunk(chunk)
    
    return result
```

### 3. Benchmarking
```python
def benchmark_feature(
    feature: BaseFeature,
    data_sizes: List[int]
) -> Dict[str, List[float]]:
    """Benchmark feature calculation performance."""
    results = {
        'size': [],
        'time': [],
        'memory': []
    }
    
    for size in data_sizes:
        data = torch.randn(size)
        
        # Measure time
        start = time.time()
        feature(data)
        duration = time.time() - start
        
        # Record results
        results['size'].append(size)
        results['time'].append(duration)
        results['memory'].append(
            torch.cuda.max_memory_allocated() / 1024**2
        )
        
    return results
``` 