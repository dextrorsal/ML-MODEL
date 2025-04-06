# Lorentzian ANN Trading System

This repository contains a machine learning-based trading system that uses a Lorentzian Approximate Nearest Neighbors classifier to generate trading signals for cryptocurrency markets.

## Key Features

- GPU-accelerated technical indicators
- Multi-timeframe analysis
- Dynamic feature weighting
- Regime-aware position sizing
- Modular architecture for easy extension

## Setup

1. Ensure you have Freqtrade installed. If not, follow the [Freqtrade installation guide](https://www.freqtrade.io/en/stable/installation/).

2. Clone this repository or copy the files to your Freqtrade `user_data` directory.

3. Install required Python packages:
   ```
   pip install torch pandas numpy matplotlib scikit-learn
   ```

## Project Structure

```
.
├── strategies/              # Trading strategies
│   └── LorentzianStrategy/ # Main strategy implementation
│       ├── features/       # Technical indicators
│       └── models/        # ML models
├── tests/                 # Test suite
├── docs/                 # Documentation
└── example_data/        # Sample data
```

For detailed documentation, see:
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Feature Engineering System](docs/FEATURES.md)
- [Technical Strategy Details](docs/TECHNICAL_STRATEGY.md)

## Running Tests

To compare different implementations:

```bash
python test_model_comparison.py
```

This will:
1. Load historical data
2. Test Modern, Standalone, and Analysis implementations
3. Generate performance metrics
4. Plot comparison results

## Live Trading

### Setup

1. Edit `config_live.json` to:
   - Set your exchange API keys (if not using dry-run mode)
   - Configure trading pairs
   - Adjust risk parameters

2. Start the trading bot in dry-run mode:
   ```bash
   python run_live_trading.py
   ```

3. Monitor trading performance:
   ```bash
   python monitor_trades.py
   ```

### Going Live

When ready to trade with real money:

1. Start with a small amount
2. Run:
   ```bash
   python run_live_trading.py --live
   ```

**WARNING**: Live trading uses real money. Use at your own risk!

## Strategy Settings

The Lorentzian Strategy has these key parameters:

- **Timeframe Weights**:
  - Short-term: 0.4
  - Medium-term: 0.35
  - Long-term: 0.25

- **Signal Thresholds**:
  - Buy: > 0.10
  - Sell: < -0.10

You can adjust these in `strategies/LorentzianStrategy/models/primary/lorentzian_classifier.py`.

## Performance Monitoring

The monitoring script provides:
- Win rate and total profit
- Performance by trading pair
- Open trade status
- Visual performance chart

## Advanced Usage

### Model Updates

The model supports continuous learning:

```python
# Load existing model
model = LorentzianClassifier()
model.load_model("models/lorentzian_model.pt")

# Update with new data
model.update(new_features, new_prices)
```

### Parameter Optimization

To optimize strategy parameters:

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy LorentzianStrategy
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Files

- `analyze_lorentzian_ann.py`: The core implementation of the Lorentzian ANN model
- `test_lorentzian_save.py`: Script to test the model's save/load functionality
- `test_freqtrade_lorentzian.py`: Script to test the model with Freqtrade's backtesting data
- `run_live_trading.py`: Script to start Freqtrade with live trading
- `monitor_trades.py`: Script to monitor trading performance
- `config_live.json`: Configuration file for live trading

## Running Backtests

To run a backtest:

```bash
python test_freqtrade_lorentzian.py --pair SOL/USDT --timeframe 5m
```

This will:
1. Load historical data
2. Either load a pre-trained model or train a new one
3. Generate predictions
4. Calculate performance metrics
5. Plot results

## Archived Files

Previous model implementations and test scripts are stored in the `archived` folder for reference. 