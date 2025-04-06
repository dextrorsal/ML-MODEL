# Lorentzian Model Comparison System

This tool compares the performance of different Lorentzian classifier implementations using real or synthetic market data.

## Features

- Compare 4 different Lorentzian classifier implementations:
  - Your Implementation: Your custom implementation
  - Modern PyTorch Implementation: GPU-accelerated implementation
  - Standalone Implementation: Simplified standalone version
  - Analysis Implementation: Analysis-oriented implementation
- Test with real market data from Bitget or Binance
- Backtest with configurable parameters including leverage and position sizing
- Support for both spot and futures trading with configurable order types
- Paper trading simulation with customizable starting balance
- Comprehensive performance metrics and visualizations
- Save results to customizable output directories

## Configuration Options

You can configure the comparison system using:

1. **Command-line arguments**: For quick changes to specific parameters
2. **Configuration files**: For saving and reusing complete test configurations
3. **A combination of both**: Load a config file and override specific settings

### Using Configuration Files

Sample configuration files are provided in the `config_samples` directory at the project root:

- `default_btc_config.json`: BTC futures trading with 3x leverage
- `eth_spot_config.json`: ETH spot trading
- `sol_scalping_config.json`: SOL scalping setup with higher leverage and position size

To use a configuration file:

```bash
python compare_all_implementations.py --config ../../config_samples/default_btc_config.json
```

Or when running from the project root:

```bash
python src/comparison/compare_all_implementations.py --config config_samples/default_btc_config.json
```

### Command-line Arguments

You can override any configuration parameter using command-line arguments:

```bash
python compare_all_implementations.py --symbol SOL/USDT --starting_balance 5000 --position_size 0.1
```

Or combine with a configuration file:

```bash
python compare_all_implementations.py --config ../../config_samples/eth_spot_config.json --position_size 0.2 --leverage 2
```

### Saving Configurations

You can save your current configuration settings to a file:

```bash
python compare_all_implementations.py --symbol BTC/USDT --starting_balance 10000 --save_config ../../config_samples/my_custom_config.json
```

## Available Parameters

| Parameter           | Description                                       | Default Value        |
|---------------------|---------------------------------------------------|----------------------|
| `--exchange`        | Exchange to use (binance or bitget)               | bitget               |
| `--market_type`     | Market type (spot or futures)                     | spot                 |
| `--symbol`          | Trading pair (BTC/USDT, ETH/USDT, SOL/USDT)       | SOL/USDT             |
| `--timeframe`       | Chart timeframe (1m, 5m, 15m, 1h, 4h, 1d)         | 5m                   |
| `--data_limit`      | Number of candles to fetch                        | 1000                 |
| `--order_type`      | Order type (FOK, GTC, IOC)                        | GTC                  |
| `--starting_balance`| Starting balance in USDT                          | 10000                |
| `--position_size`   | Percentage of balance to use per trade (0.1 = 10%)| 0.1                  |
| `--leverage`        | Leverage for futures trading                      | 1                    |
| `--fee_rate`        | Trading fee percentage                            | 0.001                |
| `--slippage`        | Slippage percentage                               | 0.0005               |
| `--output_dir`      | Directory to save output files                    | results              |

## Output Files

The comparison generates the following output files in the specified output directory:

- `model_comparison.png`: Visual comparison of model signals and price data
- `model_performance_comparison.png`: Bar chart comparing key performance metrics
- `equity_curves.png`: Account balance over time for each model
- `model_comparison_results.csv`: Detailed performance metrics in CSV format

## Performance Metrics

The system calculates and compares the following metrics for each implementation:

- Win rate percentage
- Total return percentage
- Maximum drawdown
- Sharpe ratio
- Signal activity (percentage of non-hold signals)
- Total number of trades
- Distribution of buy/sell/hold signals
- Average profit per trade
- Profit factor (ratio of gross profits to gross losses)
- Total fees paid

## Trade Simulation

The paper trading simulation provides detailed statistics including:
- Win/loss counts and percentages
- Profit factor
- Win/loss streaks
- Largest wins and losses
- Average win and loss amounts
- Portfolio growth and final balance
- Total fees paid

## Running Comparisons

Examples:

1. **Basic comparison with default settings**:
   ```bash
   python compare_all_implementations.py
   ```

2. **Using a configuration file**:
   ```bash
   python compare_all_implementations.py --config ../../config_samples/default_btc_config.json
   ```

3. **Custom settings via command line**:
   ```bash
   python compare_all_implementations.py --symbol ETH/USDT --timeframe 15m --starting_balance 5000 --leverage 3
   ```

4. **Save your custom configuration**:
   ```bash
   python compare_all_implementations.py --symbol BTC/USDT --leverage 5 --save_config ../../config_samples/my_btc_config.json
   ```

5. **Using a specific output directory**:
   ```bash
   python compare_all_implementations.py --output_dir ../../results/my_test_results
   ``` 