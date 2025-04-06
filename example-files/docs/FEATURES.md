# Feature Engineering System

This document details the feature engineering system used in our Lorentzian Trading Strategy.

## Overview

Our feature engineering system is built on modular, GPU-accelerated technical indicators implemented in PyTorch. Each indicator is designed to capture specific market characteristics while maintaining computational efficiency.

## Available Features

### 1. Relative Strength Index (RSI)
- **Implementation**: `features/rsi.py`
- **Parameters**:
  - `period`: Lookback period (default: 14)
  - `smooth_factor`: Smoothing factor for EMA calculation
- **Characteristics**:
  - Momentum oscillator
  - Range: 0-100
  - GPU-accelerated calculations
  - Optimized for real-time updates

### 2. WaveTrend Oscillator
- **Implementation**: `features/wave_trend.py`
- **Parameters**:
  - `channel_length`: Main period (default: 10)
  - `average_length`: Smoothing period (default: 21)
- **Characteristics**:
  - Combines moving average and momentum
  - Enhanced trend detection
  - Reduced noise through smoothing

### 3. Commodity Channel Index (CCI)
- **Implementation**: `features/cci.py`
- **Parameters**:
  - `period`: Lookback period (default: 20)
  - `constant`: Scaling factor (default: 0.015)
- **Characteristics**:
  - Measures price deviation from average
  - Identifies overbought/oversold conditions
  - Volatility-aware scaling

### 4. Average Directional Index (ADX)
- **Implementation**: `features/adx.py`
- **Parameters**:
  - `period`: DI calculation period (default: 14)
  - `smooth_period`: ADX smoothing period (default: 14)
- **Characteristics**:
  - Trend strength measurement
  - Directional movement analysis
  - Smoothed calculations for stability

## Feature Combination

Features are combined in the Lorentzian Classifier using:
1. Multi-timeframe analysis
2. Dynamic weighting based on market regime
3. Cross-feature correlation analysis

### Example Usage

```python
from features.rsi import RSIFeature
from features.wave_trend import WaveTrendFeature

# Initialize features
rsi = RSIFeature(period=14)
wave_trend = WaveTrendFeature(channel_length=10)

# Calculate features
rsi_values = rsi.forward(close_prices)
wt_values = wave_trend.forward(high_prices, low_prices, close_prices)
```

## Performance Considerations

1. **Memory Efficiency**
   - Optimized tensor operations
   - In-place calculations where possible
   - Efficient memory management for historical data

2. **Computational Speed**
   - GPU acceleration for all indicators
   - Vectorized operations
   - Parallel processing support

3. **Numerical Stability**
   - Robust handling of edge cases
   - NaN and infinity checks
   - Proper numerical precision

## Future Enhancements

1. **Planned Features**
   - Volume Profile analysis
   - Order flow indicators
   - Market regime detection
   - Volatility-based adaptations

2. **Optimization Goals**
   - Further GPU optimization
   - Reduced memory footprint
   - Enhanced real-time performance

3. **Integration Plans**
   - Feature importance analysis
   - Automated feature selection
   - Dynamic feature weighting 