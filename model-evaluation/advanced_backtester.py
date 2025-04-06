"""
Advanced Backtester Module

This module provides a realistic backtesting engine for Lorentzian model comparison
with advanced features like funding rate simulation, proper fee calculation,
position management, and detailed performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import sys
import os
import asyncio
import torch
import logging
import ccxt
import math
from enum import Enum
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configuration
from model_evaluation.comparison_config import (
    ComparisonConfig,
    MarketType,
    OrderType,
    BacktestConfig,
    ClosePositionConfig,
    MarginConfig,
    OrderConfig,
    RiskConfig,
    PositionConfig,
)

# Import data collector
from src.data.collectors.sol_data_collector import SOLDataCollector


class Position:
    """Represents a trading position with entry and exit details"""

    def __init__(
        self,
        entry_time: datetime,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        entry_price: float,
        size: float,  # Size in base currency
        leverage: int = 1,
    ):
        self.entry_time = entry_time
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.leverage = leverage

        # These will be filled when position is closed
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.pnl: Optional[float] = None
        self.pnl_pct: Optional[float] = None
        self.fees_paid: float = 0.0
        self.funding_paid: float = 0.0
        self.duration: Optional[timedelta] = None
        self.exit_reason: Optional[str] = None

    def close_position(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str = "signal",
        fees: float = 0.0,
    ):
        """Close the position and calculate PnL"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.fees_paid += fees

        # Calculate duration
        self.duration = exit_time - self.entry_time

        # Calculate PnL
        price_diff = (exit_price - self.entry_price) * self.direction
        self.pnl = (
            price_diff * self.size * self.leverage - self.fees_paid - self.funding_paid
        )

        # Calculate percentage PnL
        entry_value = self.entry_price * self.size
        self.pnl_pct = (self.pnl / entry_value) * 100

        return self.pnl

    def add_funding_payment(self, amount: float):
        """Add a funding payment to the position"""
        self.funding_paid += amount

    def add_fee(self, amount: float):
        """Add a fee to the position"""
        self.fees_paid += amount

    def get_current_value(self, current_price: float) -> float:
        """Get the current value of the position"""
        price_diff = (current_price - self.entry_price) * self.direction
        unrealized_pnl = (
            price_diff * self.size * self.leverage - self.fees_paid - self.funding_paid
        )
        return unrealized_pnl

    def get_liquidation_price(self) -> float:
        """Calculate the liquidation price for the position"""
        # Simple liquidation calculation: When unrealized PnL equals -position value / leverage
        if self.direction == 1:  # Long
            return self.entry_price * (1 - 1 / self.leverage)
        else:  # Short
            return self.entry_price * (1 + 1 / self.leverage)

    def __str__(self) -> str:
        return (
            f"{self.symbol} {'LONG' if self.direction == 1 else 'SHORT'} "
            f"Entry: {self.entry_price:.2f} Size: {self.size:.4f} "
            f"PnL: {self.pnl:.2f} ({self.pnl_pct:.2f}%) "
            f"Duration: {self.duration}"
        )


class AdvancedBacktester:
    """Advanced backtester with realistic trading simulation"""

    def __init__(self, config: ComparisonConfig):
        """Initialize backtester with configuration"""
        self.config = config
        self.backtest_config = config.backtest
        self.market_config = config.market

        # Metrics
        self.equity_curve = []
        self.drawdowns = []
        self.positions = []
        self.open_positions = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.break_even_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0

        # Account state
        self.balance = self.backtest_config.initial_balance
        self.equity = self.backtest_config.initial_balance
        self.high_water_mark = self.backtest_config.initial_balance

        # Initialize data collector
        self.data_collector = SOLDataCollector()

    async def fetch_data(self, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch data using the SOLDataCollector"""
        return await self.data_collector.fetch_historical(timeframe, lookback_days)

    def generate_funding_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock funding rates for futures backtesting"""
        if self.market_config.market_type != MarketType.FUTURES:
            return pd.DataFrame()

        # If a fixed funding rate is specified, use that
        if self.backtest_config.funding_rate is not None:
            # Create a dataframe with constant funding rate
            funding_df = pd.DataFrame(index=data.index)
            funding_df["rate"] = self.backtest_config.funding_rate
            return funding_df

        # Otherwise generate realistic funding rates
        timestamps = data.index

        # Funding occurs every 8 hours on most exchanges
        # Filter to get only funding timestamps (00:00, 08:00, 16:00 UTC)
        funding_times = []
        funding_rates = []

        # Generate correlated funding rates based on price movements
        returns = data["close"].pct_change()

        for i, ts in enumerate(timestamps):
            # Check if this is a funding timestamp (every 8 hours)
            if ts.hour in [0, 8, 16] and ts.minute == 0:
                funding_times.append(ts)

                # Generate funding rate correlated with recent returns
                # Funding is often inversely related to recent price movement
                if i > 0:
                    # Calculate mean return over recent periods
                    recent_return = returns.iloc[max(0, i - 24) : i].mean()

                    # Inverse correlation with some noise
                    # Typical funding range: -0.1% to 0.1%
                    noise = np.random.normal(0, 0.0002)
                    funding_rate = -0.5 * recent_return + noise

                    # Clamp to realistic range
                    funding_rate = max(min(funding_rate, 0.001), -0.001)
                else:
                    funding_rate = np.random.normal(0, 0.0003)

                funding_rates.append(funding_rate)

        if funding_times:
            funding_df = pd.DataFrame(
                {"timestamp": funding_times, "rate": funding_rates}
            )
            funding_df.set_index("timestamp", inplace=True)
            return funding_df

        return pd.DataFrame()

    def calculate_position_size(self, price: float, direction: int) -> float:
        """Calculate position size based on configuration"""
        # Calculate position size (in base currency)
        position_value = self.balance * self.backtest_config.position_size
        position_size = position_value / price  # Convert to base currency units

        return position_size

    def apply_funding(self, position: Position, funding_rate: float, price: float):
        """Apply funding payment to position"""
        # Funding formula: position_value * funding_rate * direction
        position_value = position.size * price

        # For longs: pay positive funding, receive negative
        # For shorts: pay negative funding, receive positive
        funding_amount = position_value * funding_rate * -position.direction

        position.add_funding_payment(funding_amount)
        self.total_funding += funding_amount

        # Update equity
        self.equity -= funding_amount

    def calculate_fee(self, price: float, size: float, is_maker: bool = False) -> float:
        """Calculate trading fee for a transaction"""
        position_value = price * size

        if is_maker:
            fee_rate = self.backtest_config.maker_fee
        else:
            fee_rate = self.backtest_config.taker_fee

        fee_amount = position_value * fee_rate
        return fee_amount

    def execute_entry(
        self, timestamp: datetime, price: float, direction: int, is_maker: bool = False
    ) -> Optional[Position]:
        """Execute a trade entry"""
        if direction == 0:
            return None

        # Check if we're allowed to go short
        if direction == -1 and not self.backtest_config.allow_shorts:
            return None

        # Calculate position size
        size = self.calculate_position_size(price, direction)

        # Check available balance
        position_value = price * size
        if position_value > self.balance:
            # Adjust size to available balance
            size = self.balance / price

        # Calculate fees
        fee = self.calculate_fee(price, size, is_maker)

        # Create position
        position = Position(
            entry_time=timestamp,
            symbol=self.market_config.symbol,
            direction=direction,
            entry_price=price,
            size=size,
            leverage=self.market_config.leverage,
        )

        # Add entry fee
        position.add_fee(fee)
        self.total_fees += fee

        # Update account state
        self.balance -= fee  # Only deduct fees from balance
        self.equity = self.balance  # Equity includes unrealized PnL

        # Track position
        self.open_positions.append(position)

        return position

    def execute_exit(
        self,
        timestamp: datetime,
        position: Position,
        price: float,
        reason: str = "signal",
        is_maker: bool = False,
    ) -> float:
        """Execute a trade exit"""
        # Calculate fees
        fee = self.calculate_fee(price, position.size, is_maker)

        # Close position and calculate PnL
        pnl = position.close_position(
            exit_time=timestamp, exit_price=price, exit_reason=reason, fees=fee
        )

        # Update account state
        self.balance += pnl  # Add PnL to balance
        self.equity = self.balance  # Update equity

        # Update metrics
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_fees += fee

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        else:
            self.break_even_trades += 1

        # Update high water mark and drawdown
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        current_drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
        self.drawdowns.append(current_drawdown)
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Store closed position
        self.positions.append(position)

        # Remove from open positions
        self.open_positions.remove(position)

        return pnl

    def update_equity(self, timestamp: datetime, current_price: float):
        """Update equity value based on open positions"""
        # Start with cash balance
        new_equity = self.balance

        # Add value of open positions
        for position in self.open_positions:
            position_value = position.get_current_value(current_price)
            new_equity += position_value

        # Update equity
        self.equity = new_equity

        # Update equity curve
        self.equity_curve.append((timestamp, self.equity))

        # Update high water mark and drawdown
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        current_drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
        self.drawdowns.append(current_drawdown)
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def check_liquidations(self, timestamp: datetime, current_price: float):
        """Check for liquidations in open positions"""
        for position in list(
            self.open_positions
        ):  # Use list to safely remove during iteration
            liquidation_price = position.get_liquidation_price()

            # Check if liquidation price has been hit
            if (position.direction == 1 and current_price <= liquidation_price) or (
                position.direction == -1 and current_price >= liquidation_price
            ):
                # Execute liquidation
                self.execute_exit(
                    timestamp=timestamp,
                    position=position,
                    price=liquidation_price,  # Use liquidation price
                    reason="liquidation",
                    is_maker=False,  # Liquidations are always taker
                )

                print(
                    f"LIQUIDATION at {timestamp}: {position.symbol} {position.direction} at {liquidation_price}"
                )

    def calculate_performance_metrics(self):
        """Calculate performance metrics after backtest is complete"""
        if self.total_trades == 0:
            return

        # Win rate
        self.win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        )

        # Calculate equity curve metrics
        if self.equity_curve:
            equity_values = [e[1] for e in self.equity_curve]
            equity_returns = np.diff(equity_values) / equity_values[:-1]

            # Annualized return
            total_return = (equity_values[-1] / equity_values[0]) - 1
            days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
            self.annualized_return = ((1 + total_return) ** (365 / max(1, days))) - 1

            # Sharpe ratio (assuming risk-free rate of 0)
            self.sharpe_ratio = (
                np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(365)
                if len(equity_returns) > 0
                else 0
            )

            # Sortino ratio (using downside deviation)
            negative_returns = [r for r in equity_returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            self.sortino_ratio = (
                np.mean(equity_returns) / downside_deviation * np.sqrt(365)
                if downside_deviation > 0
                else 0
            )

            # Calmar ratio
            self.calmar_ratio = (
                self.annualized_return / self.max_drawdown
                if self.max_drawdown > 0
                else 0
            )

    def generate_report(self) -> Dict[str, Any]:
        """Generate a detailed performance report"""
        self.calculate_performance_metrics()

        return {
            "initial_balance": self.backtest_config.initial_balance,
            "final_balance": self.balance,
            "total_return_pct": (
                (self.balance / self.backtest_config.initial_balance) - 1
            )
            * 100,
            "annualized_return_pct": self.annualized_return * 100
            if hasattr(self, "annualized_return")
            else 0,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "break_even_trades": self.break_even_trades,
            "win_rate_pct": self.win_rate * 100,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "max_drawdown_pct": self.max_drawdown * 100,
            "sharpe_ratio": self.sharpe_ratio if hasattr(self, "sharpe_ratio") else 0,
            "sortino_ratio": self.sortino_ratio
            if hasattr(self, "sortino_ratio")
            else 0,
            "calmar_ratio": self.calmar_ratio if hasattr(self, "calmar_ratio") else 0,
        }

    def plot_equity_curve(self, filename: Optional[str] = None):
        """Plot equity curve"""
        if not self.equity_curve:
            print("No equity data to plot")
            return

        timestamps = [e[0] for e in self.equity_curve]
        equity_values = [e[1] for e in self.equity_curve]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity_values)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_drawdowns(self, filename: Optional[str] = None):
        """Plot drawdowns"""
        if not self.drawdowns:
            print("No drawdown data to plot")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.drawdowns)
        plt.title("Drawdowns")
        plt.xlabel("Bar")
        plt.ylabel("Drawdown %")
        plt.grid(True)

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    async def run_backtest(
        self, predictions: List[int], data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with predictions

        Args:
            predictions: List of prediction signals (1=buy, -1=sell, 0=neutral)
            data: Optional DataFrame with OHLCV data. If None, it will be fetched.

        Returns:
            Performance report
        """
        # Fetch data if not provided
        if data is None:
            data = await self.fetch_data(self.market_config.timeframe, lookback_days=30)

        if len(data) == 0:
            print("No data available for backtesting")
            return {}

        # Generate funding rates for futures markets
        funding_df = self.generate_funding_rates(data)

        # Run backtest
        print(
            f"Running backtest on {len(data)} bars of {self.market_config.symbol} data..."
        )

        # Ensure predictions match data length
        if len(predictions) > len(data):
            predictions = predictions[: len(data)]
        elif len(predictions) < len(data):
            # Pad with zeros
            predictions = predictions + [0] * (len(data) - len(predictions))

        # Iterate through each bar
        last_signal = 0

        for i, (timestamp, bar) in enumerate(data.iterrows()):
            current_price = bar["close"]
            current_signal = predictions[i] if i < len(predictions) else 0

            # Apply funding if it's a funding timestamp
            if not funding_df.empty and timestamp in funding_df.index:
                funding_rate = funding_df.loc[timestamp, "rate"]
                for position in self.open_positions:
                    self.apply_funding(position, funding_rate, current_price)

            # Check for liquidations first
            self.check_liquidations(timestamp, current_price)

            # Process signals for existing positions
            if self.open_positions and current_signal != last_signal:
                # Signal changed, close existing positions
                for position in list(self.open_positions):
                    # Only close if signal is opposite or neutral
                    if current_signal == 0 or current_signal == -position.direction:
                        self.execute_exit(
                            timestamp=timestamp,
                            position=position,
                            price=current_price,
                            reason="signal",
                            is_maker=self.backtest_config.order_type == OrderType.LIMIT,
                        )

            # Open new position if signal exists and no open positions
            if (
                current_signal != 0
                and len(self.open_positions) < self.backtest_config.max_positions
            ):
                # Check if we have the opposite direction signal
                open_directions = [p.direction for p in self.open_positions]

                # Don't open if we already have position in this direction
                if current_signal not in open_directions:
                    self.execute_entry(
                        timestamp=timestamp,
                        price=current_price,
                        direction=current_signal,
                        is_maker=self.backtest_config.order_type == OrderType.LIMIT,
                    )

            # Update equity after all position changes
            self.update_equity(timestamp, current_price)

            # Update last signal
            last_signal = current_signal

        # Close any remaining positions at the end
        final_timestamp = data.index[-1]
        final_price = data.iloc[-1]["close"]

        for position in list(self.open_positions):
            self.execute_exit(
                timestamp=final_timestamp,
                position=position,
                price=final_price,
                reason="backtest_end",
                is_maker=False,
            )

        # Generate report
        report = self.generate_report()

        # Generate plots if configured
        if self.config.show_plots:
            self.plot_equity_curve()
            self.plot_drawdowns()

        if self.config.save_plots:
            os.makedirs("results", exist_ok=True)
            self.plot_equity_curve(f"results/{self.config.plot_filename}")
            self.plot_drawdowns(f"results/drawdowns_{self.config.plot_filename}")

        return report


# Example usage
async def test_backtester():
    from model_evaluation.comparison_config import binance_btc_futures_config

    # Initialize backtester
    backtester = AdvancedBacktester(binance_btc_futures_config)

    # Get data
    data = await backtester.fetch_data("1h", 30)

    # Generate random predictions
    predictions = np.random.choice([-1, 0, 1], size=len(data))

    # Run backtest
    report = await backtester.run_backtest(predictions, data)

    # Print report
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(test_backtester())
