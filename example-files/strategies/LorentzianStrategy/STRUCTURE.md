# �� Project Structure: GPU-Accelerated Trading System

This document provides a comprehensive guide to the project structure and explains how all components work together.

## 🎯 System Overview

The trading system is built around three core principles:

1. **🚀 GPU Acceleration**
   - All components optimized for GPU processing
   - Efficient batch operations
   - Real-time performance

2. **📊 Multi-timeframe Analysis**
   - Short-term (5-period window)
   - Medium-term (10-period window)
   - Long-term (20-period window)
   - Dynamic weight adjustment

3. **⚖️ Adaptive Risk Management**
   - Volatility-based position sizing
   - Dynamic stop-loss calculation
   - Market regime detection
   - Performance monitoring

## 📂 Directory Structure & Component Guide

```
strategies/LorentzianStrategy/
│
├── 📊 Core Strategy Files
│   ├── lorentzian_strategy.py      # Main strategy implementation
│   └── config.py                   # Central configuration
│
├── 📈 Models
│   └── primary/
│       └── lorentzian_classifier.py  # Signal generation model
│
├── 📉 Features
│   ├── base.py                     # Base feature class
│   ├── rsi.py                      # RSI implementation
│   ├── wave_trend.py              # WaveTrend oscillator
│   ├── cci.py                     # Commodity Channel Index
│   └── adx.py                     # Average Directional Index
│
├── 🧪 Testing
│   ├── test_model_comparison.py   # Implementation comparison
│   └── test_features/            # Feature unit tests
│
└── 📚 Documentation
    ├── README.md                 # Quick start guide
    ├── DOCUMENTATION.md          # Component documentation
    ├── INTEGRATION.md           # Integration guidelines
    └── STRUCTURE.md             # This file
```

## 🔄 Data Flow & Integration

### Feature Processing Pipeline
1. Raw price data → GPU tensors
2. Tensor data → Feature calculations
3. Features → Multi-timeframe analysis
4. Analysis → Signal generation
5. Signals → Position management

### Key Integration Points

1. **Feature Calculation**
   ```python
   from strategies.LorentzianStrategy.features import RSIFeature, WaveTrendFeature
   
   # Initialize on GPU
   rsi = RSIFeature(period=14)
   wave_trend = WaveTrendFeature(channel_length=10)
   
   # Calculate features
   features = {
       'rsi': rsi.forward(close_prices),
       'wave_trend': wave_trend.forward(high, low, close)
   }
   ```

2. **Signal Generation**
   ```python
   from strategies.LorentzianStrategy.models.primary.lorentzian_classifier import LorentzianClassifier
   
   # Initialize classifier
   classifier = LorentzianClassifier()
   
   # Generate signals with multi-timeframe analysis
   signals = classifier.generate_signals(features)
   ```

3. **Position Management**
   ```python
   # Calculate position size based on volatility
   position_size = classifier.calculate_position_size(
       signal=signals[-1],
       volatility=current_volatility,
       balance=account_balance
   )
   
   # Get adaptive stop levels
   stop_loss, take_profit = classifier.manage_risk(
       position=position_size,
       current_price=price,
       volatility=current_volatility
   )
   ```

## 🛠️ Common Usage Patterns

### 1. Feature Development
```python
from strategies.LorentzianStrategy.features.base import BaseFeature
import torch

class NewFeature(BaseFeature):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # GPU-accelerated calculations
        return processed_data
```

### 2. Model Integration
```python
# In your strategy file
def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
    # Convert to tensor
    tensor_data = torch.tensor(dataframe['close'].values, device=self.device)
    
    # Calculate features
    features = self.calculate_features(tensor_data)
    
    # Generate signals
    signals = self.classifier.generate_signals(features)
    
    return self.prepare_dataframe(dataframe, signals)
```

## 🔧 Configuration

Key configuration areas:

1. **Feature Parameters**
   - Lookback periods
   - Smoothing factors
   - Calculation methods

2. **Signal Generation**
   - Timeframe weights
   - Threshold levels
   - Regime detection

3. **Risk Management**
   - Position sizing
   - Stop-loss calculation
   - Take-profit levels

## 📈 Development Workflow

1. **Feature Development**
   - Implement in `features/` directory
   - Ensure GPU optimization
   - Add unit tests

2. **Testing**
   - Use `test_model_comparison.py`
   - Verify GPU utilization
   - Check performance metrics

3. **Integration**
   - Update configuration
   - Test with live data
   - Monitor performance

## 🔍 Performance Optimization

1. **GPU Utilization**
   - Batch processing
   - In-place operations
   - Memory management

2. **Feature Optimization**
   - Vectorized calculations
   - Efficient algorithms
   - Minimal CPU-GPU transfers

3. **Memory Management**
   - Tensor cleanup
   - Cache optimization
   - Resource monitoring

---

*For detailed implementation information, refer to individual component documentation and docstrings within each file.* 