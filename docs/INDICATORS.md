# 📊 Custom Technical Indicators

## Table of Contents
1. [Lorentzian Classifier](#lorentzian-classifier)
2. [Wave Trend Enhanced](#wave-trend-enhanced)
3. [Custom RSI Implementation](#custom-rsi-implementation)
4. [Custom CCI Implementation](#custom-cci-implementation)

## Lorentzian Classifier

The core of our signal generation system, using a unique distance metric for pattern recognition.

### Mathematical Foundation
```python
def lorentzian_distance(x1, x2):
    return np.log(1 + np.abs(x1 - x2))
```

### Key Components
1. **Distance Calculation**
   ```python
   distances = np.array([
       lorentzian_distance(current_point, historical_point)
       for historical_point in historical_data
   ])
   ```

2. **Neighbor Selection**
   - Uses k=8 nearest neighbors
   - Chronological sampling (4 bar spacing)
   - Dynamic threshold adjustment

3. **Signal Generation**
   ```python
   signal_strength = np.mean(np.sort(distances)[:n_neighbors])
   signal = 1 if signal_strength > upper_threshold else -1 if signal_strength < lower_threshold else 0
   ```

### Optimization Parameters
| Parameter | Default | Range | Description |
|-----------|---------|--------|-------------|
| n_neighbors | 8 | 4-12 | Number of neighbors |
| bar_spacing | 4 | 2-8 | Chronological spacing |
| threshold | 0.1 | 0.05-0.2 | Signal threshold |

## Wave Trend Enhanced

Custom implementation of the WaveTrend indicator with enhanced sensitivity.

### Calculation
```python
def calculate_wave_trend(close, channel_length=10, avg_length=11):
    # Step 1: Calculate ESA (Exponential Moving Average)
    esa = ema(close, channel_length)
    
    # Step 2: Calculate absolute distance
    d = ema(abs(close - esa), channel_length)
    
    # Step 3: Calculate CI (Raw Wave Trend)
    ci = (close - esa) / (0.015 * d)
    
    # Step 4: Calculate Wave Trend Lines
    wt1 = ema(ci, avg_length)      # Wave Trend Line 1
    wt2 = sma(wt1, 4)             # Wave Trend Line 2
    
    return wt1, wt2
```

### Signal Zones
- **Overbought**: > 60
- **Oversold**: < -60
- **Extreme Overbought**: > 80
- **Extreme Oversold**: < -80

### Cross Signals
```python
def wave_trend_cross(wt1, wt2):
    # Bullish Cross
    bullish = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    
    # Bearish Cross
    bearish = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    
    return bullish, bearish
```

## Custom RSI Implementation

Enhanced RSI with smoothing and multiple timeframe analysis.

### Features
1. **Dual Smoothing**
   ```python
   def smooth_rsi(close, length=14, smooth=1):
       # Calculate base RSI
       rsi = talib.RSI(close, timeperiod=length)
       
       # Apply smoothing if specified
       if smooth > 1:
           rsi = talib.EMA(rsi, timeperiod=smooth)
       
       return rsi
   ```

2. **Multi-timeframe Integration**
   ```python
   def multi_timeframe_rsi(data, timeframes=[14, 28, 56]):
       signals = []
       for tf in timeframes:
           rsi = smooth_rsi(data, length=tf)
           signals.append(rsi)
       return np.mean(signals, axis=0)
   ```

### Signal Generation
- **Strong Buy**: RSI < 30 with positive divergence
- **Strong Sell**: RSI > 70 with negative divergence
- **Neutral**: 30 ≤ RSI ≤ 70

## Custom CCI Implementation

Modified CCI with enhanced sensitivity and noise reduction.

### Calculation
```python
def enhanced_cci(high, low, close, length=20, smooth=1):
    # Calculate Typical Price
    tp = (high + low + close) / 3
    
    # Calculate SMA of Typical Price
    sma_tp = talib.SMA(tp, timeperiod=length)
    
    # Calculate Mean Deviation
    mean_dev = mean_deviation(tp, sma_tp, length)
    
    # Calculate Raw CCI
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    
    # Apply smoothing
    if smooth > 1:
        cci = talib.EMA(cci, timeperiod=smooth)
    
    return cci
```

### Signal Thresholds
| Zone | Range | Signal Strength |
|------|-------|----------------|
| Extreme Overbought | > 200 | Strong Sell |
| Overbought | 100 to 200 | Weak Sell |
| Neutral | -100 to 100 | No Signal |
| Oversold | -200 to -100 | Weak Buy |
| Extreme Oversold | < -200 | Strong Buy |

### Divergence Detection
```python
def detect_cci_divergence(price, cci, window=14):
    # Price making higher highs
    price_hh = price > price.shift(window)
    
    # CCI making lower highs
    cci_lh = cci < cci.shift(window)
    
    # Bearish divergence
    bearish_div = price_hh & cci_lh
    
    # Similar for bullish divergence
    return bearish_div
```

## Integration with ML Pipeline

### Feature Engineering
```python
def calculate_all_features(ohlcv_data):
    features = {}
    
    # Add Wave Trend
    features['wt1'], features['wt2'] = calculate_wave_trend(
        ohlcv_data['close']
    )
    
    # Add RSI
    features['rsi'] = smooth_rsi(
        ohlcv_data['close']
    )
    
    # Add CCI
    features['cci'] = enhanced_cci(
        ohlcv_data['high'],
        ohlcv_data['low'],
        ohlcv_data['close']
    )
    
    return features
```

### Signal Combination
```python
def combine_indicator_signals(features):
    signals = {
        'wave_trend': get_wt_signal(features['wt1'], features['wt2']),
        'rsi': get_rsi_signal(features['rsi']),
        'cci': get_cci_signal(features['cci'])
    }
    
    # Weight and combine signals
    final_signal = weighted_signal_combination(signals)
    
    return final_signal
```

---

*This documentation provides detailed insights into our custom technical indicators and their implementation. Each indicator has been optimized for our specific trading strategy and market conditions.* 