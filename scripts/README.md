# Scripts Directory

*What is this doc?*  
This guide explains the purpose and usage of all scripts in the project. It's for anyone running, extending, or debugging the trading system—whether for training, deployment, or monitoring.

[Model Training Guide](../docs/MODEL_TRAINING.md) | [Neon Data Pipeline](../docs/NEON_PIPELINE.md) | [Project README](../README.md)

This directory contains executable scripts for training, deployment, database setup, dashboards, and maintenance of the ML-MODEL trading system.

## Main Scripts

- **combined_model_trader.py**  
  Loads both 5-minute and 15-minute models, combines their signals for robust trading decisions. Supports live trading and backtesting.

- **extended_training.py**  
  Runs extended training sessions with walk-forward optimization and hyperparameter tuning for the SOL trading model.

- **migrate_configs.py**  
  Migrates configuration files from old locations to the new model-evaluation directory, updating paths as needed.

- **setup_database.py**  
  Sets up and initializes the Neon/PostgreSQL database for storing trading data and signals.

- **start_trading_system.py**  
  Entry point for starting the live trading system, integrating data collection, model inference, and dashboard visualization.

- **test_db_connection.py**  
  Tests the database connection and verifies that the Neon/PostgreSQL setup is working as expected.

- **train_model.py**  
  Trains the trading model on historical data using the specified configuration and features.

- **train_model_walkforward.py**  
  Trains the model using a walk-forward approach for more robust out-of-sample evaluation.

## Subdirectories

### dashboard/
- **trader_dashboard.py**: Flask-based web dashboard for real-time monitoring of trading signals and price data.
- **model_dashboard.py**: (Presumed) Dashboard for visualizing model performance and metrics.
- **requirements-dashboard.txt**: Python dependencies for running the dashboards.
- **data/**: Temporary or cached data for the dashboard.
- **templates/**: HTML/Jinja templates for rendering the dashboard UI.

### deployment/
- **check_gpu.py**: Checks PyTorch and TensorFlow installations, GPU support, and runs quick GPU performance tests.
- **monitor_gpu.py**: Monitors GPU memory usage and status in real time.

### training/
- **data_pipeline_example.py**: Example of using the Neon data pipeline for collecting, processing, and batching trading data for ML.
- **train_pattern_recognition.py**: Trains a pattern recognition model for trading, using synthetic or real data and hierarchical timeframes.

## Usage

Run any script with:
```bash
python scripts/<script_name>.py [--options]
```

Refer to each script's internal documentation or use `--help` for details on available options.

## Notes
- All scripts are tightly integrated with the ML-MODEL data pipeline, feature engineering, and model code.
- Dashboards require the dependencies listed in `requirements-dashboard.txt`.
- For environment setup and GPU monitoring, use the scripts in `deployment/`.
- For training and experimentation, see the scripts in `training/`.

## See Also
- [Project README](../README.md) — Project overview and structure
- [Model Training Guide](../docs/MODEL_TRAINING.md) — How to train and evaluate models
- [Neon Data Pipeline](../docs/NEON_PIPELINE.md) — Data ingestion and processing
- [Technical Strategy](../docs/TECHNICAL_STRATEGY.md) — How scripts fit into the trading workflow
- [tests/README.md](../tests/README.md) — Test suite and integration
- [src/](../src/) — Main source code for features, models, and data pipeline
