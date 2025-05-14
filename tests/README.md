# Tests Directory

*What is this doc?*  
This guide explains the structure and purpose of all tests in the project. It's for anyone who wants to verify, extend, or debug the system—whether you're a developer, researcher, or contributor.

[Model Training Guide](../docs/MODEL_TRAINING.md) | [Technical Strategy](../docs/TECHNICAL_STRATEGY.md) | [Project README](../README.md)

This directory contains all unit, integration, and data pipeline tests for the ML-MODEL trading system. Tests are organized by component and use `pytest` for execution and fixtures.

---

## Top-Level Files
- **conftest.py**: Common pytest fixtures for sample data, device selection, and test configuration.
- **__init__.py**: Marks the directory as a package and documents the test suite structure.

---

## Subdirectories

### features/
- **test_indicators.py**: Unit tests for custom feature indicators (WaveTrend, RSI, ADX, CCI). Checks output shape, value ranges, and integration.

### indicators/
- **run_indicator_tests.py**: Downloads test data from Binance and runs indicator validation.
- **indicator_validation.py**: Validates indicators for look-ahead bias, overfitting, and signal distribution. Integrates with custom indicator classes.

### models/
- **test_logistic_regression.py**: Tests initialization, signal generation, filters, metrics, and plotting for the custom logistic regression model.

### strategy/
- **test_integration.py**: Integration tests for the full trading strategy pipeline, including backtesting and performance metrics.
- **test_components.py**: Unit tests for strategy components (LorentzianANN, LogisticRegression, ChandelierExit) and their integration.

### utils/
- *(Empty or only contains cache)*

### data/
- **test_collector.py, test_data_collection.py, test_data_integration.py, test_integration_simplified.py**: Tests for data collection, integration, and pipeline.
- **test_database.py**: Tests database operations and integration.
- **setup_test_environment.py**: Sets up a test environment for data and database tests.
- **README.md, data_testing_summary.md**: Documentation and summary of data testing.
- **requirements-data-tests.txt**: Dependencies for data-related tests.
- **test_data.db**: Test database file.

### risk/
- **test_risk_management.py**: Tests for risk management logic and modules.

### archived/
- **test_pattern_recognition.py, test_model.py**: Archived tests for pattern recognition and model evaluation.

---

## Notes
- All test modules use pytest and are designed for modular, component-based testing.
- Fixtures and sample data are provided for reproducibility.
- Data and indicator tests are comprehensive, including integration and edge cases.
- Archived tests are kept for reference but not actively maintained.

---

## See Also
- [Project README](../README.md) — Project overview and structure
- [Model Training Guide](../docs/MODEL_TRAINING.md) — How to train and evaluate models
- [Technical Strategy](../docs/TECHNICAL_STRATEGY.md) — How tests fit into the trading workflow
- [Neon Data Pipeline](../docs/NEON_PIPELINE.md) — Data pipeline and integration tests
- [src/](../src/) — Main source code for features, models, and data pipeline
- [scripts/README.md](../scripts/README.md) — Scripts for training, deployment, and testing
