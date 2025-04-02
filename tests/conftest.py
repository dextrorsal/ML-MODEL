"""Common test fixtures for the ML-Powered Crypto Trading Bot."""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

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

@pytest.fixture(scope="session")
def device():
    """Get appropriate device for testing"""
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

@pytest.fixture
def sample_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate random walk prices
    returns = np.random.normal(0, 0.02, n_samples)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.random.uniform(0, 0.01, n_samples))
    low = close * (1 - np.random.uniform(0, 0.01, n_samples))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1h'),
        'close': close,
        'high': high,
        'low': low,
        'volume': np.random.uniform(1000, 5000, n_samples)
    })
    return df

@pytest.fixture
def trending_data():
    """Generate trending price data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate trending price data
    trend = np.linspace(0, 1, n_samples)  # Uptrend
    noise = np.random.normal(0, 0.02, n_samples)
    close = 100 * (1 + trend + noise)
    high = close * (1 + np.random.uniform(0, 0.01, n_samples))
    low = close * (1 - np.random.uniform(0, 0.01, n_samples))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1h'),
        'close': close,
        'high': high,
        'low': low,
        'volume': np.random.uniform(1000, 5000, n_samples)
    })
    return df

@pytest.fixture
def test_neon_connection():
    """Get test database connection string"""
    return "postgresql://[YOUR_TEST_DB]/test_db" 