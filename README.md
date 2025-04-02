# 🤖 ML-Powered Crypto Trading Bot

> A sophisticated machine learning trading bot leveraging PyTorch and Neon for algorithmic trading on Solana and other cryptocurrencies.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

## 🌟 Features

### 🎯 Trading Strategy Components
- **Primary Signal Generation**: Lorentzian Classifier with PyTorch acceleration
- **Signal Confirmation**: Enhanced Logistic Regression with deep learning
- **Risk Management**: Advanced Chandelier Exit system
- **Position Sizing**: Dynamic position sizing based on volatility

### 📊 Technical Indicators
- **Base PyTorch Indicators**: GPU-accelerated technical analysis
- **Wave Trend Analysis**: Advanced momentum detection
- **Custom Implementations**: RSI, CCI, ADX with Pine Script accuracy
- **Trend Level Analysis**: Multi-timeframe support

### 🧠 Machine Learning Infrastructure
- **PyTorch Integration**: GPU-accelerated model training
- **Automatic Mixed Precision**: Optimized performance
- **LSTM & Attention**: Deep learning for pattern recognition
- **Real-time Inference**: Live trading capabilities

### 💾 Data Management
- **Neon Database**: Efficient data storage and retrieval
- **Real-time Pipeline**: Live market data processing
- **Batch Processing**: Optimized data loading
- **Visualization Tools**: Advanced charting and analysis

## 📝 Project Structure

Key directories:
- `src/models/` - Trading strategy models and ML architecture
- `src/features/` - Technical analysis and indicators
- `src/data/` - Data collection and processing
- `tests/` - Testing infrastructure
- `docs/` - Detailed documentation

## 🚀 Quick Start

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
   python scripts/setup_database.py
   ```

4. **Collect historical data**
   ```bash
   python src/data/collectors/sol_data_collector.py --historical --days 60
   ```

5. **Train your model**
   ```bash
   python scripts/train_model.py --data-days 30 --epochs 10
   ```

6. **Run the dashboard**
   ```bash
   pip install -r scripts/dashboard/requirements-dashboard.txt
   python scripts/dashboard/model_dashboard.py
   ```

7. **Start trading (paper mode)**
   ```bash
   python scripts/run_trading.py --paper --model models/trained/your_model.pt
   ```