# System Architecture

This document outlines the architectural organization of the Lorentzian Trading System.

## Directory Structure

```
.
├── strategies/
│   └── LorentzianStrategy/
│       ├── features/          # Technical indicators and feature calculations
│       │   ├── rsi.py
│       │   ├── wave_trend.py
│       │   ├── cci.py
│       │   └── adx.py
│       └── models/
│           └── primary/      # Primary signal generation models
│               └── lorentzian_classifier.py
├── tests/                    # Test suite
├── docs/                    # Documentation
└── example_data/           # Sample data for testing
```

## Key Components

### 1. Feature Engineering (`strategies/LorentzianStrategy/features/`)

The features directory contains modular implementations of technical indicators:

- `rsi.py`: Relative Strength Index implementation
- `wave_trend.py`: WaveTrend oscillator
- `cci.py`: Commodity Channel Index
- `adx.py`: Average Directional Index

Each indicator is implemented as a standalone class with PyTorch acceleration support.

### 2. Model Implementation (`strategies/LorentzianStrategy/models/`)

The models directory contains our core trading models:

#### Primary Signal Generator
- Located in `models/primary/lorentzian_classifier.py`
- Implements the enhanced Lorentzian Classifier
- Features:
  - GPU acceleration
  - Multi-timeframe analysis
  - Dynamic feature weighting
  - Adaptive signal thresholds

### 3. Testing Framework

Multiple testing approaches are supported:

- Unit tests in `tests/`
- Integration testing via `test_model_comparison.py`
- Live simulation through `run_live_trading.py`

## Implementation Details

### Feature Engineering

Each feature module follows a consistent pattern:
```python
class FeatureIndicator:
    def __init__(self, **params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, *inputs):
        # GPU-accelerated calculations
        pass
```

### Model Architecture

The Lorentzian Classifier combines:
1. Traditional technical analysis
2. Machine learning pattern recognition
3. Multi-timeframe consensus voting

Key improvements:
- Enhanced signal generation with balanced timeframe weights
- Adaptive volatility filtering
- Regime-aware position sizing

## Performance Optimization

The system includes several optimizations:
1. GPU acceleration for indicator calculations
2. Vectorized operations using PyTorch
3. Efficient memory management for historical data
4. Parallel processing for multi-timeframe analysis

## Future Improvements

Planned architectural improvements:
1. Microservices architecture for distributed computing
2. Real-time data streaming integration
3. Enhanced monitoring and logging system
4. Automated model retraining pipeline 