# 🔧 Feature Development Guide

This guide explains how to develop and integrate new technical indicators and features into our GPU-accelerated trading system.

## 📋 Feature Development Process

### 1. Planning
1. Define the feature's purpose and mathematical formula
2. Identify required input data
3. Plan GPU optimization strategy
4. Consider memory and performance implications

### 2. Implementation
1. Create new feature class in `features/` directory
2. Inherit from `BaseFeature`
3. Implement GPU-accelerated calculations
4. Add documentation and tests

## 🏗️ Feature Template

```python
from typing import Optional, Dict, Union
import torch
from features.base import BaseFeature

class NewFeature(BaseFeature):
    """
    Feature Description
    
    Parameters
    ----------
    period : int
        Lookback period for calculations
    param2 : float
        Description of param2
        
    Attributes
    ----------
    name : str
        Feature identifier
    requires_grad : bool
        Whether the feature requires gradient computation
    """
    
    def __init__(
        self,
        period: int = 14,
        param2: float = 0.5,
        device: Optional[torch.device] = None
    ):
        super().__init__(device=device)
        self.period = period
        self.param2 = param2
        self.name = "new_feature"
        self.requires_grad = False
        
    def forward(
        self,
        data: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Calculate feature values
        
        Parameters
        ----------
        data : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Input price data or dictionary of tensors
            
        Returns
        -------
        torch.Tensor
            Calculated feature values
        """
        # Implementation here
        pass
    
    def _validate_inputs(self, data: torch.Tensor) -> None:
        """
        Validate input data
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to validate
            
        Raises
        ------
        ValueError
            If input data is invalid
        """
        if data.dim() != 2:
            raise ValueError("Expected 2D tensor input")
        if data.size(1) < self.period:
            raise ValueError(f"Input length must be >= {self.period}")
```

## 🎯 Example Implementation

Here's an example of implementing a new momentum feature:

```python
class MomentumFeature(BaseFeature):
    def __init__(
        self,
        period: int = 10,
        smoothing: int = 3,
        device: Optional[torch.device] = None
    ):
        super().__init__(device=device)
        self.period = period
        self.smoothing = smoothing
        self.name = "momentum"
        
    def forward(self, close: torch.Tensor) -> torch.Tensor:
        # Ensure data is on the correct device
        close = self.to_device(close)
        
        # Calculate momentum
        momentum = close / close.roll(self.period) - 1
        
        # Apply smoothing if needed
        if self.smoothing > 1:
            kernel = torch.ones(self.smoothing, device=self.device)
            kernel = kernel / kernel.sum()
            momentum = torch.conv1d(
                momentum.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=self.smoothing-1
            ).squeeze()
            
        return momentum
```

## 🧪 Testing Features

### 1. Unit Tests
Create tests in `tests/test_features/`:

```python
import pytest
import torch
from features.momentum import MomentumFeature

def test_momentum_feature():
    # Setup
    feature = MomentumFeature(period=10)
    data = torch.randn(100, device=feature.device)
    
    # Calculate feature
    result = feature(data)
    
    # Assertions
    assert isinstance(result, torch.Tensor)
    assert result.shape == data.shape
    assert not torch.isnan(result).any()
```

### 2. Integration Tests
Test feature in the full pipeline:

```python
def test_feature_integration():
    # Setup system
    classifier = LorentzianClassifier()
    feature = MomentumFeature()
    
    # Add feature
    classifier.add_feature(feature)
    
    # Test predictions
    predictions = classifier.predict(data)
    assert predictions.shape == expected_shape
```

## 📊 Performance Considerations

### 1. Memory Efficiency
```python
def efficient_calculation(self, data: torch.Tensor) -> torch.Tensor:
    # Pre-allocate output tensor
    result = torch.zeros_like(data)
    
    # In-place operations
    result.copy_(data)
    result.div_(result.roll(1))
    result.sub_(1)
    
    return result
```

### 2. Batch Processing
```python
def batch_process(self, data: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_result = self.forward(batch)
        results.append(batch_result)
    return torch.cat(results)
```

## 🔍 Feature Documentation

### 1. Class Documentation
```python
class MyFeature(BaseFeature):
    """
    Feature Name
    
    Description of what the feature does and how it's calculated.
    
    Mathematical Formula:
    Y = f(X) where...
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
        
    Examples
    --------
    >>> feature = MyFeature(param1=10)
    >>> result = feature(data)
    """
```

### 2. Implementation Notes
```python
def forward(self, data: torch.Tensor) -> torch.Tensor:
    """
    Implementation notes:
    1. Data is validated and moved to GPU
    2. Main calculation is vectorized
    3. Results are smoothed if needed
    4. Memory is managed efficiently
    """
```

## 🚀 Integration Steps

1. **Create Feature**
   ```python
   # features/my_feature.py
   class MyFeature(BaseFeature):
       # Implementation
   ```

2. **Add Tests**
   ```python
   # tests/test_features/test_my_feature.py
   def test_my_feature():
       # Tests
   ```

3. **Update Documentation**
   ```markdown
   # docs/FEATURES.md
   ## MyFeature
   Description and usage examples
   ```

4. **Register Feature**
   ```python
   # strategies/LorentzianStrategy/features/__init__.py
   from .my_feature import MyFeature
   ```

## 📈 Feature Lifecycle

1. **Development**
   - Plan implementation
   - Write code and tests
   - Optimize performance

2. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

3. **Documentation**
   - API documentation
   - Usage examples
   - Performance notes

4. **Maintenance**
   - Monitor performance
   - Update as needed
   - Handle bug reports 