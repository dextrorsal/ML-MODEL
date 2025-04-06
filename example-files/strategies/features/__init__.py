"""
Feature Generators Package

This package contains various feature generators used by the trading strategies.
Each feature is implemented as a PyTorch module for GPU acceleration.

Features:
- RSI (Relative Strength Index)
- WaveTrend
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)
"""

from .rsi import RSIIndicator, RSIConfig
from .wave_trend import WaveTrendIndicator, WaveTrendConfig
from .adx import ADXIndicator, ADXConfig
from .cci import CCIIndicator, CCIConfig

__all__ = [
    'RSIIndicator',
    'RSIConfig',
    'WaveTrendIndicator',
    'WaveTrendConfig',
    'ADXIndicator',
    'ADXConfig',
    'CCIIndicator',
    'CCIConfig'
]