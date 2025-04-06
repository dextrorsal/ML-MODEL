# Model Evaluation Framework

This directory contains a consolidated framework for evaluating trading models and indicators.

## Structure

- `model_evaluator.py` - The main consolidated evaluation framework
- Implementation files:
  - `example_logistic_regression.py` - Example implementation of logistic regression
  - `src_logistic_regression.py` - Source implementation of logistic regression
  - `example_chandelier_exit.py` - Example implementation of Chandelier Exit
  - `src_chandelier_exit.py` - Source implementation of Chandelier Exit
- `data/` - Directory for data storage
- `config_samples/` - Sample configuration files
- `output/` - Output directory for results

## Usage

```bash
# Basic usage with default configuration
python model_evaluator.py

# Use a specific configuration file
python model_evaluator.py --config config_samples/default_config.json

# Compare logistic regression implementations
python model_evaluator.py --compare_logistic

# Compare Chandelier Exit implementations
python model_evaluator.py --compare_chandelier

# Compare both types of implementations
python model_evaluator.py --compare_logistic --compare_chandelier
```

## Configuration

Configuration can be provided via:
1. Command-line arguments
2. Configuration JSON file

Example configuration file:
```json
{
  "exchange": {
    "name": "binance",
    "testnet": true
  },
  "data": {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2023-01-01",
    "end_date": "2023-08-01"
  },
  "backtest": {
    "initial_balance": 10000.0,
    "risk_per_trade": 2.0,
    "fee_rate": 0.001
  },
  "output": {
    "output_dir": "output",
    "show_plots": true
  },
  "compare_logistic_regression": true,
  "compare_chandelier_exit": true
}
```

## Available Models

### Logistic Regression
- Binary classification model for BUY/SELL signals
- Both plain PyTorch and advanced deep learning versions available

### Chandelier Exit
- Traditional indicator for trend following
- Uses ATR for volatility-based stop loss placement

## Features

- **Configuration Management**: Easily configure and save model and backtest parameters
- **Data Fetching**: Automatically fetch historical data or use cached data
- **Multiple Model Support**: Compare various model implementations
  - Lorentzian Classifier implementations
  - Logistic Regression implementations
  - Chandelier Exit implementations
- **Backtesting Engine**: Test models on historical data with realistic trading rules
- **Performance Metrics**: Calculate comprehensive trading statistics
- **Visualization**: Generate performance comparison charts

## Model Implementations

The framework supports multiple model implementations:

### Lorentzian Classifier

- `your_implementation.py` - Your custom implementation
- `standalone_implementation.py` - Standalone implementation
- `analysis_implementation.py` - Analysis-focused implementation
- `modern_pytorch_implementation.py` - Modern PyTorch implementation

### Logistic Regression

- `example_logistic_regression.py` - Example implementation from strategy folder
- `src_logistic_regression.py` - Your source implementation

### Chandelier Exit

- `example_chandelier_exit.py` - Example implementation from strategy folder
- `src_chandelier_exit.py` - Your source implementation

## Output and Metrics

The evaluator generates several metrics for each model:

- **Total Profit**: Total profit/loss
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return
- **Total Trades**: Number of trades executed
- **Average Hold Time**: Average trade duration

## Visualization

The framework generates several plots for visual comparison:

- **Equity Curves**: Compare the balance growth of different models
- **Drawdowns**: Compare the drawdowns of different models
- **Trade Distribution**: Histogram of trade profit/loss for each model

## How to Add New Models

To add a new model implementation:

1. Create a new file with your model class
2. Import the model in `model_evaluator.py`
3. Add import handling in the imports section
4. Create a wrapper class if needed
5. Update the `create_model` factory function
6. Add the model to your configuration

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- PyTorch
- Matplotlib
- ccxt (for data fetching)
- tabulate (for pretty tables)

## License

MIT 