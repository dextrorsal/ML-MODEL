"""Common test fixtures for the ML-Powered Crypto Trading Bot."""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    
    data = {
        'timestamp': dates,
        'open': np.random.normal(100, 10, 1000),
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.lognormal(0, 1, 1000) * 1000
    }
    
    # Ensure high/low/close are consistent
    data['close'] = data['open'] + np.random.normal(0, 2, 1000)
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.normal(0, 1, 1000))
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.normal(0, 1, 1000))
    
    return pd.DataFrame(data)

@pytest.fixture
def device():
    """Get the appropriate device for PyTorch testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def test_data_dir(project_root):
    """Get the test data directory."""
    data_dir = project_root / 'tests' / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir 