# 🤖 ML-Powered Crypto Trading Bot

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
<!-- [![Docs Status](https://img.shields.io/badge/docs-passing-brightgreen)](docs/) -->
<!-- [![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/) -->

> A sophisticated machine learning trading bot leveraging PyTorch and Neon for algorithmic trading on Solana and other cryptocurrencies.

---

## 🚦 Quick Links
- [Getting Started](#-quick-start)
- [Project Map](#-project-map)
- [Documentation Home](docs/INDEX.md) *(see all docs)*
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Model Training Guide](docs/MODEL_TRAINING.md)
- [Technical Indicators](docs/INDICATORS.md)
- [ML Model Architecture](docs/ML_MODEL.md)
- [Technical Strategy](docs/TECHNICAL_STRATEGY.md)
- [Scripts & Automation](scripts/README.md)
- [Tests & Validation](tests/README.md)

---

## 🗺️ Project Map
```
ML-MODEL/
├── src/                # Main source code (features, models, data, utils)
│   ├── features/       # Custom technical indicators (RSI, CCI, ADX, WaveTrend)
│   ├── indicators/     # Base indicator classes
│   ├── models/         # Model strategies and training
│   ├── data/           # Data collection and processing
│   └── ...
├── scripts/            # Training, deployment, dashboard, and utility scripts
├── tests/              # Unit, integration, and data pipeline tests
├── docs/               # All documentation (see docs/INDEX.md)
├── archive/            # Legacy/experimental code (see archive/README.md)
├── model-evaluation/   # Lorentzian model comparison tools
├── config_samples/     # Sample configs for strategies
└── ...
```

---

## 👋 New Here? (Onboarding & Contributing)
- **Start with [Getting Started](#-quick-start) below.**
- **Explore the [Project Map](#-project-map)** to see how everything fits together.
- **Every doc ends with a 'See Also' section** for easy navigation—use it to jump between related docs and code.
- **Want to contribute?**
  - Fork the repo, make your changes, and open a pull request!
  - See [tests/README.md](tests/README.md) for how to run and add tests.
  - Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.
- **Questions?** Open an issue or check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

---

## 🖼️ Visuals & Screenshots
*Add screenshots of the dashboard, sample plots, or architecture diagrams here!*

---

## 📚 Documentation

Dive deeper into the project with our comprehensive documentation:

### Core Concepts
- [📈 Trading Philosophy](docs/TRADING_PHILOSOPHY.md) - Understanding the core trading approach
- [🔬 Technical Strategy](docs/TECHNICAL_STRATEGY.md) - Detailed technical implementation

### Technical Documentation
- [📊 Technical Indicators](docs/INDICATORS.md) - Custom indicator implementations
- [🔄 Data Pipeline](docs/NEON_PIPELINE.md) - Neon database integration
- [🧠 ML Architecture](docs/ML_MODEL.md) - Model design and training
- [🛠️ Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [📈 Model Training Guide](docs/MODEL_TRAINING.md) - How to train and evaluate ML models

---

## 📝 Project Structure

Key directories:
- `src/features/` - Core technical indicators (RSI, CCI, ADX, WaveTrend)
- `src/indicators/` - Base indicator foundations
- `src/models/strategy/` - Trading strategies (Lorentzian, Logistic Regression, Chandelier)
- `src/models/training/` - Model training utilities
- `src/pattern-recognition/` - Pattern detection algorithms
- `src/data/` - Data collection and processing
- `model-evaluation/` - Lorentzian model implementation comparison tools
- `tests/` - Testing infrastructure
- `docs/` - Detailed documentation
- `scripts/dashboard/` - Web-based trading dashboard
- `config_samples/` - Sample configurations for testing different trading strategies

---

## 🏃 Example Workflow: End-to-End Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML-MODEL.git
   cd ML-MODEL
   ```
2. **Install dependencies**
   ```bash
   conda create -n ML-torch python=3.8
   conda activate ML-torch
   pip install -r requirements.txt
   ```
3. **Setup database**
   ```bash
   python scripts/setup_database.py --connection-string "your_neon_connection_string"
   ```
4. **Collect historical data**
   ```bash
   python src/data/collectors/sol_data_collector.py --historical --days 60
   ```
5. **Train your model**
   ```bash
   python scripts/train_model.py --data-days 30 --epochs 10
   ```
6. **Compare model implementations**
   ```bash
   python model-evaluation/compare_all_implementations.py --config model-evaluation/config_samples/default_btc_config.json
   ```
7. **Start the trading system with dashboard**
   ```bash
   python scripts/start_trading_system.py --confidence-threshold 0.3 --neon-connection "your_neon_connection_string"
   # Access the dashboard at http://127.0.0.1:5000
   ```

---

## 📈 Performance Monitoring

The integrated dashboard provides real-time visualization of:
- Current price and historical chart
- ML model signals with confidence levels
- Trading performance metrics
- All backed by the Neon database for persistent storage

---

*Every documentation file ends with a 'See Also' section for easy navigation!*