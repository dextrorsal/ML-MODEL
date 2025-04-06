"""
PACKAGE DEFINITION: Integrated ML Trading System

This package provides a complete trading system that combines:
1. Lorentzian ANN (Primary Signal Generator)
2. Logistic Regression (Signal Confirmation)
3. Chandelier Exit (Risk Management)

The system is organized into modules for indicators, models, and integration components.
This init file provides access to the LorentzianStrategy class for use with Freqtrade.

Usage:
```python
from strategies.LorentzianStrategy import LorentzianStrategy
```

For standalone backtesting without Freqtrade, use the IntegratedMLTrader class:
```python
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader
```
"""

from .lorentzian_strategy import LorentzianStrategy

__version__ = "1.0.0" 