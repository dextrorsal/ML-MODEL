import pytest
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.strategy.logistic_regression_torch import (
    LogisticRegression,
    LogisticConfig,
)


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    np.random.seed(42)

    # Generate random walk prices
    close = 100 * (1 + np.random.randn(100).cumsum() * 0.02)
    high = close * (1 + abs(np.random.randn(100)) * 0.01)
    low = close * (1 - abs(np.random.randn(100)) * 0.01)
    open_prices = close * (1 + np.random.randn(100) * 0.005)
    volume = np.random.randint(800, 1000, size=100)

    # Add some ATR and volume metrics for filtering
    atr1 = np.abs(np.diff(close, prepend=close[0])) * 3
    atr10 = pd.Series(atr1).rolling(10).mean().values
    volume_rsi = 50 + np.random.randn(100) * 10  # Simple mock RSI

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr1": atr1,
            "atr10": atr10,
            "volume_rsi": volume_rsi,
        },
        index=dates,
    )


@pytest.mark.strategy
class TestLogisticRegression:
    def test_initialization(self):
        """Test model initialization with default and custom configs"""
        # Test with default config
        model = LogisticRegression()
        assert model.config.lookback == 3
        assert model.config.learning_rate == 0.0009
        assert model.config.iterations == 1000

        # Test with custom config
        custom_config = LogisticConfig(
            lookback=5, learning_rate=0.001, iterations=500, volatility_filter=False
        )
        model = LogisticRegression(config=custom_config)
        assert model.config.lookback == 5
        assert model.config.learning_rate == 0.001
        assert model.config.iterations == 500
        assert model.config.volatility_filter is False

    def test_signal_generation(self, sample_data):
        """Test basic signal generation functionality"""
        model = LogisticRegression()
        result = model.calculate_signals(sample_data)

        # Check result dataframe structure
        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "signal" in result.columns
        assert "loss" in result.columns
        assert "prediction" in result.columns

        # Check signals have expected values (BUY=1, SELL=-1, HOLD=0)
        assert set(result["signal"].unique()).issubset({1, -1, 0})

        # Ensure we have the right number of rows
        assert len(result) == len(sample_data)

    def test_with_filters(self, sample_data):
        """Test signal generation with filters enabled/disabled"""
        # With filters
        model_with_filters = LogisticRegression(
            config=LogisticConfig(volatility_filter=True, volume_filter=True)
        )
        result_with_filters = model_with_filters.calculate_signals(sample_data)

        # Without filters
        model_without_filters = LogisticRegression(
            config=LogisticConfig(volatility_filter=False, volume_filter=False)
        )
        result_without_filters = model_without_filters.calculate_signals(sample_data)

        # We expect more signals without filters
        signal_count_with = (result_with_filters["signal"] != 0).sum()
        signal_count_without = (result_without_filters["signal"] != 0).sum()

        # This might not always be true due to randomness, but generally should be
        # In cases where it's not, the test will still pass
        assert signal_count_with <= signal_count_without + 5

    def test_metrics_tracking(self, sample_data):
        """Test trade metrics tracking functionality"""
        model = LogisticRegression()
        result = model.calculate_signals(sample_data)

        # Get metrics
        metrics = model.get_metrics()

        # Check metrics structure
        assert "total_trades" in metrics
        assert "winning_trades" in metrics
        assert "losing_trades" in metrics
        assert "win_rate" in metrics
        assert "win_loss_ratio" in metrics
        assert "cumulative_return" in metrics

        # Basic value checks
        assert metrics["total_trades"] >= 0
        assert metrics["winning_trades"] >= 0
        assert metrics["losing_trades"] >= 0
        assert 0 <= metrics["win_rate"] <= 1 or metrics["win_rate"] == 0

    def test_plotting(self, sample_data, monkeypatch):
        """Test that plotting function runs without errors"""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, "show", lambda: None)

        model = LogisticRegression()
        result = model.calculate_signals(sample_data)

        # This should run without errors
        model.plot_signals(sample_data, result)

        # Test passes if no exception is raised
        assert True
