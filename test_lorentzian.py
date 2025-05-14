"""
Quick test script for the Lorentzian Classifier
"""

import pandas as pd
import numpy as np
import ccxt
import json
from datetime import datetime, timedelta
from src.models.strategy.lorentzian_classifier import LorentzianANN


def fetch_market_data(symbol="SOL/USDT", timeframe="5m", limit=500):
    """Fetch market data from Bitget"""
    try:
        # Initialize exchange
        exchange = ccxt.bitget()

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return generate_test_data(limit)


def generate_test_data(length=500):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5 * length)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=length)

    # Generate price data
    close = np.zeros(length)
    close[0] = 100  # Starting price

    # Random walk with momentum
    for i in range(1, length):
        momentum = 0.1 * (close[i - 1] - close[max(0, i - 5)]) / close[max(0, i - 5)]
        random_change = np.random.normal(0, 0.02)
        close[i] = close[i - 1] * (1 + momentum + random_change)

    # Generate OHLC around close
    high = close * (1 + abs(np.random.normal(0, 0.01, length)))
    low = close * (1 - abs(np.random.normal(0, 0.01, length)))
    open_price = low + (high - low) * np.random.random(length)

    # Generate volume
    volume = np.random.normal(1000000, 200000, length)
    volume = np.maximum(100, volume)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=timestamps,
    )

    return df


def main():
    # Load config
    with open("config_samples/quick_test_config.json", "r") as f:
        config = json.load(f)

    print("Fetching market data...")
    df = fetch_market_data(
        symbol=config["symbol"],
        timeframe=config["timeframe"],
        limit=config["data_limit"],
    )
    print(f"Data shape: {df.shape}")

    # Initialize model
    print("\nInitializing Lorentzian Classifier...")
    model = LorentzianANN(
        lookback_bars=20,
        prediction_bars=4,
        k_neighbors=20,
        use_regime_filter=True,
        use_volatility_filter=True,
    )

    # Calculate signals
    print("\nCalculating trading signals...")
    signals = model.calculate_signals(df)

    # Print summary statistics
    print("\nSignal Statistics:")
    print(f"Total periods: {len(df)}")
    print(f"Buy signals: {int(signals['buy_signals'].sum())}")
    print(f"Sell signals: {int(signals['sell_signals'].sum())}")

    # Calculate basic metrics
    returns = df["close"].pct_change()
    signal_returns = returns * (signals["buy_signals"] - signals["sell_signals"])

    print("\nPerformance Metrics:")
    print(f"Total return: {signal_returns.sum():.2%}")
    print(f"Win rate: {(signal_returns > 0).mean():.2%}")
    print(
        f"Sharpe ratio: {signal_returns.mean() / signal_returns.std() * np.sqrt(252):.2f}"
    )

    print("\nTest completed!")


if __name__ == "__main__":
    main()
