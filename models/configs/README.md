# ⚙️ Model Configuration

This directory contains configuration files for the ML models and their components.

## Files

### `model_config.py`
Contains configuration classes for:
- Base model parameters
- Pattern recognition model
- Lorentzian classifier
- Trading parameters
- Data pipeline

## Usage

```python
from models.configs import (
    PatternRecognitionConfig,
    LorentzianConfig,
    TradingConfig,
    DataConfig
)

# Use default configuration
config = PatternRecognitionConfig()

# Or customize configuration
custom_config = PatternRecognitionConfig(
    batch_size=128,
    learning_rate=0.0005,
    num_epochs=200,
    hidden_size=256
)

# Access configuration in your model
model = PatternRecognitionModel(config=custom_config)
```

## Configuration Guidelines

1. **Model Parameters**:
   - Adjust batch size based on GPU memory
   - Tune learning rate for stability
   - Set appropriate early stopping patience

2. **Trading Parameters**:
   - Risk management settings
   - Position sizing rules
   - Entry/exit conditions

3. **Data Pipeline**:
   - Timeframe selection
   - Feature engineering settings
   - Train/validation splits

## Best Practices

1. **Version Control**:
   - Track configuration changes
   - Document parameter updates
   - Keep backup of working configs

2. **Testing**:
   - Validate parameter ranges
   - Test extreme values
   - Ensure backward compatibility

3. **Documentation**:
   - Comment non-obvious parameters
   - Explain parameter relationships
   - Document optimal ranges 