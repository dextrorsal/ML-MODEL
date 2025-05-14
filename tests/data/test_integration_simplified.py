import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
import torch
from pathlib import Path

# Add root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.features.wave_trend import WaveTrendIndicator
from src.features.rsi import RSIIndicator
from src.features.adx import ADXIndicator
from src.features.cci import CCIIndicator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_price_data():
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")

    # Generate random walk price data
    returns = np.random.normal(0, 0.01, 500)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.random.uniform(0, 0.005, 500))
    low = close * (1 - np.random.uniform(0, 0.005, 500))
    open_price = close * (1 + np.random.normal(0, 0.002, 500))
    volume = np.random.uniform(1000, 5000, 500)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return df


def test_indicator_calculation(sample_price_data):
    """Test that technical indicators can process data correctly"""
    df = sample_price_data

    # Initialize indicators
    wt = WaveTrendIndicator()
    rsi = RSIIndicator()
    adx = ADXIndicator()
    cci = CCIIndicator()

    # Calculate signals
    logger.info("Calculating WaveTrend signals...")
    wt_signals = wt.calculate_signals(df)

    logger.info("Calculating RSI signals...")
    rsi_signals = rsi.calculate_signals(df)

    logger.info("Calculating ADX signals...")
    adx_signals = adx.calculate_signals(df)

    logger.info("Calculating CCI signals...")
    cci_signals = cci.calculate_signals(df)

    # Verify dimensions match input data - note our indicators return tensors
    assert len(wt_signals["wt1"]) == len(df), "WaveTrend signal length mismatch"
    assert len(rsi_signals["rsi"]) == len(df), "RSI signal length mismatch"

    # Check if ADX returns 'adx' or 'ADX' key
    adx_key = "adx" if "adx" in adx_signals else "ADX" if "ADX" in adx_signals else None
    assert adx_key is not None, "ADX signal not found in output"
    assert len(adx_signals[adx_key]) == len(df), "ADX signal length mismatch"

    assert len(cci_signals["cci"]) == len(df), "CCI signal length mismatch"

    # Verify signals are calculated after warmup period
    warmup = 50  # Allow 50 periods for warmup

    # Convert tensors to numpy for easier handling if needed
    wt1 = (
        wt_signals["wt1"].cpu().numpy()
        if isinstance(wt_signals["wt1"], torch.Tensor)
        else wt_signals["wt1"].values
    )
    rsi_vals = (
        rsi_signals["rsi"].cpu().numpy()
        if isinstance(rsi_signals["rsi"], torch.Tensor)
        else rsi_signals["rsi"].values
    )

    # Check for non-NaN values after warmup
    assert not np.isnan(
        wt1[warmup:]
    ).all(), "WaveTrend signals are all NaN after warmup"
    assert not np.isnan(rsi_vals[warmup:]).all(), "RSI signals are all NaN after warmup"

    logger.info("All indicator tests passed!")


def test_timeframe_resampling(sample_price_data):
    """Test data resampling between timeframes"""
    # Start with hourly data
    df_1h = sample_price_data

    # Resample to 4-hour data
    df_4h = (
        df_1h.set_index("timestamp")
        .resample("4h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .reset_index()
    )

    # Verify dimensions
    assert len(df_4h) == len(df_1h) // 4 + (
        1 if len(df_1h) % 4 > 0 else 0
    ), "Incorrect number of rows after resampling"

    # Verify that high is the max of the component candles
    for i in range(min(5, len(df_4h))):  # Check first 5 candles
        start_idx = i * 4
        end_idx = min(start_idx + 4, len(df_1h))

        component_high = df_1h["high"][start_idx:end_idx].max()
        resampled_high = df_4h["high"].iloc[i]

        assert (
            abs(component_high - resampled_high) < 0.0001
        ), f"High value mismatch for candle {i}"

        # Similarly check volume is summed correctly
        component_volume = df_1h["volume"][start_idx:end_idx].sum()
        resampled_volume = df_4h["volume"].iloc[i]

        assert (
            abs(component_volume - resampled_volume) < 0.0001
        ), f"Volume mismatch for candle {i}"

    logger.info("Timeframe resampling tests passed!")


def test_indicator_integration(sample_price_data):
    """Test multiple indicators working together"""
    df = sample_price_data

    # Initialize indicators
    wt = WaveTrendIndicator()
    rsi = RSIIndicator()

    # Calculate signals
    wt_signals = wt.calculate_signals(df)
    rsi_signals = rsi.calculate_signals(df)

    # Create a trading strategy combining both indicators
    # Convert tensors to numpy arrays for processing
    wt1 = (
        wt_signals["wt1"].cpu().numpy()
        if isinstance(wt_signals["wt1"], torch.Tensor)
        else wt_signals["wt1"].values
    )
    rsi_vals = (
        rsi_signals["rsi"].cpu().numpy()
        if isinstance(rsi_signals["rsi"], torch.Tensor)
        else rsi_signals["rsi"].values
    )

    # Create numpy array for signals
    trading_signals = np.zeros(len(df))

    # Simple strategy: Buy when WaveTrend crosses above -60 AND RSI > 30
    # Sell when WaveTrend crosses below 60 OR RSI < 70
    warmup = 50

    for i in range(warmup + 1, len(df)):
        # Buy signal
        if wt1[i - 1] < -60 and wt1[i] > -60 and rsi_vals[i] > 30:
            trading_signals[i] = 1

        # Sell signal
        elif wt1[i - 1] > 60 and wt1[i] < 60 or rsi_vals[i] > 70:
            trading_signals[i] = -1

    # Verify signals are generated
    buy_signals = (trading_signals == 1).sum()
    sell_signals = (trading_signals == -1).sum()

    logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")

    assert buy_signals > 0, "No buy signals generated"
    assert sell_signals > 0, "No sell signals generated"

    logger.info("Indicator integration tests passed!")


if __name__ == "__main__":
    # For manual testing, run the test functions directly
    df = sample_price_data()
    test_indicator_calculation(df)
    test_timeframe_resampling(df)
    test_indicator_integration(df)
    print("All tests passed!")
