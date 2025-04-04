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
- **Neon Database**: Efficient data storage and retrieval with PostgreSQL
- **Real-time Pipeline**: Live market data processing
- **Batch Processing**: Optimized data loading
- **Visualization Tools**: Advanced charting and analysis

## 📊 Neon Database Integration

Our trading bot now features full integration with Neon PostgreSQL for:

- **Price Data Storage**: All OHLCV data is automatically stored in the database
- **Trading Signals**: ML model trading signals are saved and timestamped
- **Model Predictions**: Track model performance over time
- **Real-time Dashboard**: Web-based monitoring of signals and performance
- **Historical Analysis**: Query and analyze past trading decisions

**Database Schema:**
- `price_data`: Historical price data (OHLCV)
- `trading_signals`: ML model generated signals with confidence levels
- `model_predictions`: Raw model predictions for performance tracking
- `model_checkpoints`: Model version tracking

## 📝 Project Structure

Key directories:
- `src/models/` - Trading strategy models and ML architecture
- `src/features/` - Technical analysis and indicators
- `src/data/` - Data collection and processing
- `tests/` - Testing infrastructure
- `docs/` - Detailed documentation
- `scripts/dashboard/` - Web-based trading dashboard

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
   The project uses Neon PostgreSQL as the database backend. Ensure you have:
   - A Neon account (https://neon.tech)
   - A project created with the required tables
   - Your connection string ready

   ```bash
   # Set up your database with the schema
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

6. **Start the complete trading system with dashboard**
   ```bash
   python scripts/start_trading_system.py --confidence-threshold 0.3 --neon-connection "your_neon_connection_string"
   ```
   This will start the ML trader and web dashboard simultaneously. Access the dashboard at http://127.0.0.1:5000

7. **Or run components separately:**
   ```bash
   # Start just the dashboard
   python scripts/dashboard/trader_dashboard.py
   
   # Start just the trader (live mode)
   python scripts/combined_model_trader.py --live --confidence-threshold 0.3
   ```

## 🔄 Updating the Trading System

The Neon database integration provides:

1. **Persistent Data Storage**: Your trading signals and price data are now stored permanently in the cloud
2. **Portable Analysis**: Connect visualization tools directly to Neon for deeper analysis
3. **Signal Tracking**: Every ML prediction is tracked and timestamped for performance analysis
4. **Cloud Backup**: Data is automatically backed up through Neon's managed service

To view your data manually, connect to your Neon database using any PostgreSQL client with your connection string:
```
postgresql://neondb_owner:password@endpoint-id.us-east-2.aws.neon.tech/neondb
```

## 📈 Performance Monitoring

The integrated dashboard provides real-time visualization of:
- Current price and historical chart
- ML model signals with confidence levels
- Trading performance metrics
- All backed by the Neon database for persistent storage