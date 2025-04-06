"""
Example Chandelier Exit Implementation from example-files directory

This is a copy of the Chandelier Exit implementation from:
'example-files/strategies/LorentzianStrategy/tools/indicators/chandelier_exit.py'

Features:
- Calculates ATR-based exit levels using high/low/close prices
- Configurable ATR period and multiplier
- Option to use close prices for smoother signals
- Built-in signal generation mechanism
- Trend-following exits that adapt to market volatility
"""

import numpy as np
import pandas as pd


class ChandelierExit:
    """
    Implementation of Chandelier Exit indicator for trend following.

    The Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR).
    It's designed to keep traders in the trend while prices are moving favorably
    but exit when the trend may be reversing.
    """

    def __init__(
        self, atr_period: int = 22, atr_multiplier: float = 3.0, use_close: bool = False
    ):
        """
        Initialize Chandelier Exit calculator

        Args:
            atr_period: Period for ATR calculation, default 22
            atr_multiplier: Multiplier for ATR, default 3.0
            use_close: Use close prices instead of high/low for smoother signals, default False
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_close = use_close

    def calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Average True Range (ATR)

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Array of ATR values
        """
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        # Set the first element to high-low
        high_close[0] = 0
        low_close[0] = 0

        # Find the greatest of the three
        ranges = np.maximum(high_low, high_close)
        true_range = np.maximum(ranges, low_close)

        # Calculate ATR
        atr = np.zeros_like(true_range)
        atr[0] = true_range[0]  # First ATR is just the first TR

        # Calculate smoothed ATR
        for i in range(1, len(atr)):
            atr[i] = (
                (atr[i - 1] * (self.atr_period - 1)) + true_range[i]
            ) / self.atr_period

        return atr

    def calculate_chandelier_exit(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> tuple:
        """
        Calculate Chandelier Exit levels

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Tuple of (long_exit, short_exit) arrays
        """
        atr = self.calculate_atr(high, low, close)

        # Calculate the highest high and lowest low for the previous 22 periods
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)

        for i in range(len(high)):
            start_idx = max(0, i - self.atr_period + 1)
            highest_high[i] = np.max(high[start_idx : i + 1])
            lowest_low[i] = np.min(low[start_idx : i + 1])

        # Calculate exit levels
        long_exit = highest_high - (atr * self.atr_multiplier)
        short_exit = lowest_low + (atr * self.atr_multiplier)

        if self.use_close:
            # Use close prices for smoother signals
            long_exit = np.maximum(long_exit, close - (atr * self.atr_multiplier))
            short_exit = np.minimum(short_exit, close + (atr * self.atr_multiplier))

        return long_exit, short_exit

    def calculate_signals(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate trading signals based on Chandelier Exit

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Array of signals: 1 for long, -1 for short, 0 for no position
        """
        long_exit, short_exit = self.calculate_chandelier_exit(high, low, close)

        signals = np.zeros_like(close)
        position = 0  # 1 for long, -1 for short, 0 for no position

        for i in range(1, len(close)):
            # Current price is above the short exit, go long
            if close[i] > short_exit[i] and position <= 0:
                signals[i] = 1
                position = 1
            # Current price is below the long exit, go short
            elif close[i] < long_exit[i] and position >= 0:
                signals[i] = -1
                position = -1
            # Hold position
            else:
                signals[i] = position

        return signals

    def plot(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, dates=None):
        """
        Plot the Chandelier Exit with price data

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            dates: Optional array of dates for x-axis
        """
        import matplotlib.pyplot as plt

        long_exit, short_exit = self.calculate_chandelier_exit(high, low, close)
        signals = self.calculate_signals(high, low, close)

        plt.figure(figsize=(12, 6))

        if dates is None:
            dates = np.arange(len(close))

        plt.plot(dates, close, label="Close")
        plt.plot(dates, long_exit, label="Long Exit", color="red")
        plt.plot(dates, short_exit, label="Short Exit", color="green")

        # Plot buy/sell signals
        buy_signals = dates[signals == 1]
        sell_signals = dates[signals == -1]

        if len(buy_signals) > 0:
            plt.scatter(
                buy_signals,
                close[signals == 1],
                color="green",
                marker="^",
                s=100,
                label="Buy",
            )
        if len(sell_signals) > 0:
            plt.scatter(
                sell_signals,
                close[signals == -1],
                color="red",
                marker="v",
                s=100,
                label="Sell",
            )

        plt.title("Chandelier Exit")
        plt.legend()
        plt.grid(True)
        plt.show()


# Create alias for backward compatibility
ChandelierExitIndicator = ChandelierExit
