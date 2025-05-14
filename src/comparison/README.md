# Lorentzian Model Comparison System

*What is this doc?*  
This guide explains how to use the Lorentzian model comparison tools. It's for anyone benchmarking, testing, or analyzing different model implementations.

[ML Model Architecture](../../docs/ML_MODEL.md) | [Technical Strategy](../../docs/TECHNICAL_STRATEGY.md) | [Project README](../../README.md)

This tool compares the performance of different Lorentzian classifier implementations using real or synthetic market data.

## Features

- Compare 4 different Lorentzian classifier implementations
- Test with real market data from Bitget or Binance
- Backtest with configurable parameters
- Paper trading simulation with starting balance and position sizing
- Supports both spot and futures trading
- Multiple output formats including charts and CSV reports

## Configuration Options

You can configure the comparison system using:

1. **Command-line arguments**: For quick changes to specific parameters
2. **Configuration files**: For saving and reusing complete test configurations
3. **A combination of both**: Load a config file and override specific settings

### Using Configuration Files

Sample configuration files are provided in the `config_samples` directory:

- `default_btc_config.json`: BTC futures trading with 3x leverage
- `eth_spot_config.json`: ETH spot trading
- `sol_scalping_config.json`: SOL scalping setup with higher leverage

To use a configuration file:

```bash
python compare_all_implementations.py --config config_samples/default_btc_config.json
```

### Command-line Arguments

You can override any configuration parameter using command-line arguments:

```bash
python compare_all_implementations.py --symbol SOL/USDT --starting_balance 5000 --position_size 0.1
```

Or combine with a configuration file:

```bash
python compare_all_implementations.py --config config_samples/eth_spot_config.json --position_size 0.2 --leverage 2
```

### Saving Configurations

You can save your current configuration settings to a file:

```bash
python compare_all_implementations.py --symbol BTC/USDT --starting_balance 10000 --save_config my_custom_config.json
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

- `model_comparison.png`: Visual comparison of model signals
- `model_performance_comparison.png`: Bar chart comparing key metrics
- `equity_curves.png`: Account balance over time for each model
- `model_comparison_results.csv`: Detailed performance metrics in CSV format

## Running Comparisons

Examples:

1. **Basic comparison with default settings**:
   ```bash
   python compare_all_implementations.py
   ```

2. **Using a configuration file**:
   ```bash
   python compare_all_implementations.py --config config_samples/default_btc_config.json
   ```

3. **Custom settings via command line**:
   ```bash
   python compare_all_implementations.py --symbol ETH/USDT --timeframe 15m --starting_balance 5000 --leverage 3
   ```

4. **Save your custom configuration**:
   ```bash
   python compare_all_implementations.py --symbol BTC/USDT --leverage 5 --save_config my_btc_config.json
   ``` 

## See Also
- [Project README](../../README.md) — Project overview and structure
- [ML Model Architecture](../../docs/ML_MODEL.md) — Model details and integration
- [Technical Strategy](../../docs/TECHNICAL_STRATEGY.md) — How comparison fits into the workflow
- [Model Training Guide](../../docs/MODEL_TRAINING.md) — How to train and evaluate models
- [src/comparison/](./) — All comparison scripts and configs
- [config_samples/](../../model-evaluation/config_samples/) — Sample configuration files 