import pytest
import torch
import pandas as pd
import numpy as np
from src.features.technical.indicators.wave_trend import WaveTrendIndicator
from src.features.technical.indicators.rsi import RSIIndicator
from src.features.technical.indicators.adx import ADXIndicator
from src.features.technical.indicators.cci import CCIIndicator

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
        'close': close,
        'high': high,
        'low': low,
        'volume': np.random.uniform(1000, 5000, n_samples)
    })
    return df

@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_wavetrend(sample_data, device):
    """Test WaveTrend indicator calculations"""
    wt = WaveTrendIndicator(device=device)
    signals = wt.calculate_signals(sample_data)
    
    # Check signal properties
    assert 'wt1' in signals, "Missing WT1 signal"
    assert 'wt2' in signals, "Missing WT2 signal"
    assert isinstance(signals['wt1'], torch.Tensor), "WT1 should be a tensor"
    
    # Verify tensor shapes
    assert len(signals['wt1']) == len(sample_data), "WT1 length mismatch"
    assert len(signals['wt2']) == len(sample_data), "WT2 length mismatch"
    
    # Check value ranges
    assert torch.all(signals['wt1'] >= -100), "WT1 below minimum"
    assert torch.all(signals['wt1'] <= 100), "WT1 above maximum"

def test_rsi(sample_data, device):
    """Test RSI indicator calculations"""
    rsi = RSIIndicator(device=device)
    signals = rsi.calculate_signals(sample_data)
    
    # Check signal properties
    assert 'rsi' in signals, "Missing RSI signal"
    assert isinstance(signals['rsi'], torch.Tensor), "RSI should be a tensor"
    
    # Verify tensor shape
    assert len(signals['rsi']) == len(sample_data), "RSI length mismatch"
    
    # Check value ranges (RSI is always between 0 and 100)
    assert torch.all(signals['rsi'] >= 0), "RSI below 0"
    assert torch.all(signals['rsi'] <= 100), "RSI above 100"

def test_adx(sample_data, device):
    """Test ADX indicator calculations"""
    adx = ADXIndicator(device=device)
    signals = adx.calculate_signals(sample_data)
    
    # Check signal properties
    assert 'adx' in signals, "Missing ADX signal"
    assert '+di' in signals, "Missing +DI signal"
    assert '-di' in signals, "Missing -DI signal"
    
    # Verify tensor shapes
    assert len(signals['adx']) == len(sample_data), "ADX length mismatch"
    
    # Check value ranges (ADX and DI are between 0 and 100)
    assert torch.all(signals['adx'] >= 0), "ADX below 0"
    assert torch.all(signals['adx'] <= 100), "ADX above 100"
    assert torch.all(signals['+di'] >= 0), "+DI below 0"
    assert torch.all(signals['-di'] >= 0), "-DI below 0"

def test_cci(sample_data, device):
    """Test CCI indicator calculations"""
    cci = CCIIndicator(device=device)
    signals = cci.calculate_signals(sample_data)
    
    # Check signal properties
    assert 'cci' in signals, "Missing CCI signal"
    assert isinstance(signals['cci'], torch.Tensor), "CCI should be a tensor"
    
    # Verify tensor shape
    assert len(signals['cci']) == len(sample_data), "CCI length mismatch"
    
    # Check if CCI responds to price changes
    price_changes = torch.diff(torch.tensor(sample_data['close'].values, device=device))
    cci_changes = torch.diff(signals['cci'])
    
    # Filter out NaN values
    valid_mask = ~torch.isnan(cci_changes)
    if valid_mask.sum() > 1:  # Need at least 2 points for correlation
        price_filtered = price_changes[valid_mask]
        cci_filtered = cci_changes[valid_mask]
        
        if len(price_filtered) > 1:
            stacked = torch.stack([price_filtered, cci_filtered])
            correlation = torch.corrcoef(stacked)[0, 1]
            
            # There should be some correlation between price changes and CCI changes
            assert not torch.isnan(correlation), "CCI correlation is NaN"
            assert correlation != 0, "CCI shows no correlation with price changes"

def test_indicator_integration(sample_data, device):
    """Test all indicators working together"""
    # Initialize all indicators
    wt = WaveTrendIndicator(device=device)
    rsi = RSIIndicator(device=device)
    adx = ADXIndicator(device=device)
    cci = CCIIndicator(device=device)
    
    # Calculate all signals
    wt_signals = wt.calculate_signals(sample_data)
    rsi_signals = rsi.calculate_signals(sample_data)
    adx_signals = adx.calculate_signals(sample_data)
    cci_signals = cci.calculate_signals(sample_data)
    
    # Verify all tensors have the same length
    signal_lengths = [
        len(wt_signals['wt1']),
        len(rsi_signals['rsi']),
        len(adx_signals['adx']),
        len(cci_signals['cci'])
    ]
    
    assert len(set(signal_lengths)) == 1, "Indicator outputs have inconsistent lengths"
    
    # Check for NaN values
    all_signals = [
        wt_signals['wt1'], wt_signals['wt2'],
        rsi_signals['rsi'],
        adx_signals['adx'],
        cci_signals['cci']
    ]
    
    for signal in all_signals:
        assert not torch.isnan(signal).any(), "NaN values in indicator output" 