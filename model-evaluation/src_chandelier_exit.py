"""
Source Chandelier Exit Implementation from src directory

This is a copy of the Chandelier Exit implementation from:
'src/models/strategy/chandelier_exit.py'

Features:
- Simple and efficient implementation of the classic indicator
- Pure NumPy implementation for performance
- Essential signal generation logic using ATR-based stops
- Clean code focused on the core algorithm
"""

import numpy as np
import pandas as pd


class ChandelierExit:
    """
    Implementation of the Chandelier Exit indicator for trend detection.

    The Chandelier Exit is a volatility-based system that identifies
    potential market reversals by setting trailing stops based on ATR.
    """

    def __init__(
        self, atr_period: int = 22, atr_multiplier: float = 3.0, use_close: bool = False
    ):
        """
        Initialize Chandelier Exit indicator.

        Args:
            atr_period: Period for ATR calculation, default 22
            atr_multiplier: Multiplier for ATR, default 3.0
            use_close: Whether to use close prices instead of high/low for reference points
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_close = use_close

    def calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Array of ATR values
        """
        # Calculate true range components
        high_low = high - low
        high_close_prev = np.abs(high - np.roll(close, 1))
        low_close_prev = np.abs(low - np.roll(close, 1))

        # Set first value to avoid lookback issues
        high_close_prev[0] = 0
        low_close_prev[0] = 0

        # Calculate true range as maximum of the three
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))

        # Initialize ATR array
        atr = np.zeros_like(true_range)

        # First value is just the first true range
        atr[0] = true_range[0]

        # Calculate smoothed ATR (Wilder's method)
        for i in range(1, len(true_range)):
            atr[i] = (
                (atr[i - 1] * (self.atr_period - 1)) + true_range[i]
            ) / self.atr_period

        return atr

    def calculate_signals(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate trading signals based on Chandelier Exit levels.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Array of signals: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)

        # Initialize arrays for highest and lowest values
        highest = np.zeros_like(high)
        lowest = np.zeros_like(low)

        # Calculate rolling highest and lowest values
        for i in range(len(high)):
            start = max(0, i - self.atr_period + 1)
            if self.use_close:
                highest[i] = np.max(close[start : i + 1])
                lowest[i] = np.min(close[start : i + 1])
            else:
                highest[i] = np.max(high[start : i + 1])
                lowest[i] = np.min(low[start : i + 1])

        # Calculate chandelier levels
        long_exit = highest - (atr * self.atr_multiplier)
        short_exit = lowest + (atr * self.atr_multiplier)

        # Initialize signals and position arrays
        signals = np.zeros_like(close)
        position = 0  # Current position: 1 (long), -1 (short), 0 (neutral)

        # Generate signals
        for i in range(1, len(close)):
            # Long exit triggered, go short
            if close[i] < long_exit[i] and position >= 0:
                signals[i] = -1
                position = -1
            # Short exit triggered, go long
            elif close[i] > short_exit[i] and position <= 0:
                signals[i] = 1
                position = 1
            # Maintain current position
            else:
                signals[i] = position

        return signals

    def plot(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, dates=None):
        """
        Plot Chandelier Exit levels and signals.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            dates: Optional array of dates for the x-axis
        """
        import matplotlib.pyplot as plt

        # Calculate ATR
        atr = self.calculate_atr(high, low, close)

        # Initialize arrays for highest and lowest values
        highest = np.zeros_like(high)
        lowest = np.zeros_like(low)

        # Calculate rolling highest and lowest values
        for i in range(len(high)):
            start = max(0, i - self.atr_period + 1)
            if self.use_close:
                highest[i] = np.max(close[start : i + 1])
                lowest[i] = np.min(close[start : i + 1])
            else:
                highest[i] = np.max(high[start : i + 1])
                lowest[i] = np.min(low[start : i + 1])

        # Calculate chandelier levels
        long_exit = highest - (atr * self.atr_multiplier)
        short_exit = lowest + (atr * self.atr_multiplier)

        # Calculate signals
        signals = self.calculate_signals(high, low, close)

        # Create plot
        plt.figure(figsize=(12, 6))

        # Use provided dates or create array
        if dates is None:
            dates = np.arange(len(close))

        # Plot price and exit levels
        plt.plot(dates, close, label="Close Price")
        plt.plot(dates, long_exit, label="Long Exit", color="red")
        plt.plot(dates, short_exit, label="Short Exit", color="green")

        # Add buy/sell markers
        buy_signals = dates[signals == 1]
        sell_signals = dates[signals == -1]

        if len(buy_signals) > 0:
            buy_prices = close[signals == 1]
            plt.scatter(
                buy_signals,
                buy_prices,
                color="green",
                marker="^",
                s=100,
                label="Buy Signal",
            )

        if len(sell_signals) > 0:
            sell_prices = close[signals == -1]
            plt.scatter(
                sell_signals,
                sell_prices,
                color="red",
                marker="v",
                s=100,
                label="Sell Signal",
            )

        # Add title and labels
        plt.title("Chandelier Exit")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Create alias for backward compatibility
ChandelierExitIndicator = ChandelierExit
