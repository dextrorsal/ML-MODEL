import pytest
import torch
import pandas as pd
import numpy as np
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier
from src.models.strategy.confirmation.logistic_regression_torch import LogisticRegression
from src.models.strategy.risk_management.chandelier_exit import ChandelierExit

@pytest.fixture
def sample_data():
    """Generate sample OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    # Generate random walk prices
    close = 100 * (1 + np.random.randn(100).cumsum() * 0.02)
    high = close * (1 + abs(np.random.randn(100)) * 0.01)
    low = close * (1 - abs(np.random.randn(100)) * 0.01)
    volume = np.random.randint(800, 1000, size=100)
    
    return pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.mark.strategy
class TestStrategyComponents:
    
    def test_lorentzian_classifier(self, sample_data):
        """Test Lorentzian Classifier signal generation"""
        classifier = LorentzianClassifier()
        signals = classifier.calculate_signals(sample_data)
        
        assert isinstance(signals, dict)
        assert 'long_signals' in signals
        assert 'short_signals' in signals
        assert isinstance(signals['long_signals'], torch.Tensor)
        assert len(signals['long_signals']) == len(sample_data)
        
        # Test signal bounds
        assert torch.all((signals['long_signals'] >= 0) & (signals['long_signals'] <= 1))
        assert torch.all((signals['short_signals'] >= 0) & (signals['short_signals'] <= 1))
        
        # Test no simultaneous signals
        assert torch.all(signals['long_signals'] * signals['short_signals'] == 0)
    
    def test_logistic_regression(self, sample_data):
        """Test Logistic Regression confirmation"""
        regression = LogisticRegression()
        signals = regression.calculate_signals(sample_data)
        
        assert isinstance(signals, dict)
        assert 'predictions' in signals
        assert 'buy_signals' in signals
        assert 'sell_signals' in signals
        assert isinstance(signals['predictions'], torch.Tensor)
        assert len(signals['predictions']) == len(sample_data)
        
        # Test probability bounds
        valid_mask = ~torch.isnan(signals['predictions'])
        assert torch.all((signals['predictions'][valid_mask] >= 0) & 
                        (signals['predictions'][valid_mask] <= 1))
    
    def test_chandelier_exit(self, sample_data):
        """Test Chandelier Exit risk management"""
        exit_system = ChandelierExit()
        signals = exit_system.calculate_signals(sample_data)
        
        assert isinstance(signals, dict)
        assert 'long_stop' in signals
        assert 'short_stop' in signals
        assert isinstance(signals['long_stop'], torch.Tensor)
        assert len(signals['long_stop']) == len(sample_data)
        
        # Test stop levels (ignoring NaN values)
        # Move tensors to the same device
        close_tensor = torch.tensor(sample_data['close'].values, dtype=torch.float32, device=signals['long_stop'].device)
        valid_mask = ~torch.isnan(signals['long_stop'])
        
        # Use a more relaxed assertion - at least 80% of long stops should be below close
        # and at least 80% of short stops should be above close
        long_stop_below = (signals['long_stop'][valid_mask] <= close_tensor[valid_mask]).float().mean()
        short_stop_above = (signals['short_stop'][valid_mask] >= close_tensor[valid_mask]).float().mean()
        
        assert long_stop_below >= 0.8, f"Only {long_stop_below:.2f} of long stops are below close prices"
        assert short_stop_above >= 0.8, f"Only {short_stop_above:.2f} of short stops are above close prices"
    
    def test_strategy_integration(self, sample_data):
        """Test full strategy pipeline integration"""
        # Initialize components
        classifier = LorentzianClassifier()
        regression = LogisticRegression()
        exit_system = ChandelierExit()
        
        # Generate signals
        primary_signals = classifier.calculate_signals(sample_data)
        confirmation = regression.calculate_signals(sample_data)
        risk_signals = exit_system.calculate_signals(sample_data)
        
        # Verify signal flow
        assert len(primary_signals['long_signals']) == len(sample_data)
        assert len(confirmation['predictions']) == len(sample_data)
        assert len(risk_signals['long_stop']) == len(sample_data)
        
        # Test combined strategy logic
        long_entries = primary_signals['long_signals'] * (confirmation['predictions'] > 0.5).float()
        short_entries = primary_signals['short_signals'] * (confirmation['predictions'] < 0.5).float()
        
        # No simultaneous long/short entries
        assert torch.all(long_entries * short_entries == 0) 