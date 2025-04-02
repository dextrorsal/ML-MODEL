# ML Trading Model - Core Strategy

## Project Structure
```
ML-MODEL/
├── data/                      # Market data storage
│   ├── raw/                  # Raw data from Binance
│   │   └── solana/          # Solana specific data
│   └── processed/           # Preprocessed data
├── models/                   # Model checkpoints
│   └── checkpoints/         # Saved model files
│       └── pattern_recognition/
├── src/                     # Source code
│   ├── data/               # Data handling
│   │   ├── collectors/    # Data collection scripts
│   │   └── processors/   # Data processing scripts
│   ├── features/          # Feature engineering
│   │   ├── technical/    # Technical indicators
│   │   └── patterns/     # Pattern detection
│   ├── models/           # Model definitions
│   │   ├── architecture/ # Model structure
│   │   ├── training/     # Training scripts
│   │   └── archived/     # Previous model versions
│   └── utils/           # Utility functions
└── configs/            # Configuration files
    └── model_configs/  # Model parameters
```

## 1. Price Action & Pattern Recognition
### Primary Patterns
- Smart Money Concepts (SMC):
  * Change of Character (CHoCH)
  * Break of Structure (BOS)
  * Range identification and trading

- Fractal Patterns:
  * Bullish Fractals (Bottom Formation)
  * Bearish Fractals (Top Formation)
  * Fractal Trend Lines
  * Multi-timeframe Fractal Alignment

- ZigZag Patterns:
  * W Bottoms (Strong Support)
  * M Tops (Strong Resistance)
  * Swing Point Analysis
  * Trend Channel Detection

- Traditional Patterns:
  * Double Bottom/Top formations
  * Head and Shoulders (Regular & Inverse)
  * Cup and Handle patterns
  * Bull/Bear Flags
  * Ascending/Descending triangles

### Pattern Validation
- Multiple Timeframe Rules:
  * 1d: Major trend and key levels
  * 4h: Intermediate trend and structure
  * 1h: Trend direction and continuation
  * 15m: Pattern formation and validation
  * 5m: Entry timing and execution

- Pattern Confirmation Criteria:
  * Double Bottom:
    - Clear support level touch twice
    - Higher low on second touch preferred
    - Volume should decrease on second bottom
    - Minimum 2% price difference between bottoms
  
  * Fractal Patterns:
    - Minimum 5 candle formation
    - Volume confirmation on pivot points
    - Clear higher highs/lower lows structure
    - Multi-timeframe alignment preferred

  * ZigZag Patterns:
    - Minimum swing size (varies by timeframe):
      * 1d: 3% minimum swing
      * 4h: 2.5% minimum swing
      * 1h: 2% minimum swing
      * 15m: 1.5% minimum swing
      * 5m: 1% minimum swing
    - Clear pivot point formation
    - Volume confirmation at turns
    - Trend channel respect
  
  * Range Trading Rules:
    - Minimum 3% range size for valid range
    - At least 2 touches on support/resistance
    - Clear rejection wicks at boundaries
    - Volume increases at range boundaries

  * CHoCH/BOS Validation:
    - Previous structure break
    - Clear higher high/lower low formation
    - Volume increase on breakout
    - No immediate price rejection

- Volume Analysis:
  * Higher than 3-period average for breakouts
  * Lower volume acceptable in range consolidation
  * Volume trend alignment with price action
  * Relative volume comparison across timeframes

## 2. Model Architecture
### Pattern Recognition Model v2
- **Core Components:**
  * Bidirectional LSTM with Enhanced Attention
  * Multi-pattern Classification
  * Hierarchical Timeframe Learning

- **Input Features:**
  * 60-period window of OHLCV data
  * Normalized price data
  * Log-transformed volume
  * Technical indicators

- **Pattern Types:**
  * Fractal Patterns (Bullish/Bearish)
  * ZigZag Formations (W-Bottoms, M-Tops)
  * Traditional Patterns
  * SMC Components

### Training Configuration
- **Hierarchical Learning:**
  * Daily (1d): 20 epochs, batch_size=16
  * 4-Hour (4h): 15 epochs, batch_size=24
  * 1-Hour (1h): 10 epochs, batch_size=32
  * 15-Minute (15m): 8 epochs, batch_size=48
  * 5-Minute (5m): 5 epochs, batch_size=64

- **Optimization:**
  * Early stopping patience: 10 epochs
  * Learning rate reduction: 50% on plateau
  * Gradient clipping: 1.0
  * Mixed precision training

### Database Integration
- **Neon PostgreSQL:**
  * Real-time training progress tracking
  * Model checkpointing
  * Pattern detection results
  * Performance metrics

## 3. Usage Instructions
1. **Environment Setup:**
   ```bash
   conda activate ML-torch
   pip install -r requirements.txt
   ```

2. **Data Collection:**
   ```bash
   python src/data/collectors/sol_data_collector.py
   ```

3. **Model Training:**
   ```bash
   python src/models/training/train_pattern_recognition.py
   ```

4. **GPU Monitoring:**
   ```bash
   python src/utils/monitor_gpu.py
   ```

## 4. Performance Metrics
- Pattern Detection Accuracy: >94%
- Validation Loss: ~0.4915
- Real-time Processing Capability
- GPU Optimization for AMD RX 6750 XT

## 5. Future Improvements
- [ ] Add Elliott Wave Pattern Recognition
- [ ] Implement Real-time Pattern Alerts
- [ ] Enhance Multi-timeframe Correlation
- [ ] Add Position Sizing Optimization
- [ ] Implement Pattern Strength Scoring