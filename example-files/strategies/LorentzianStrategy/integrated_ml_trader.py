"""
MAIN INTEGRATION COMPONENT: Combined ML Trading System

This is the central integration module that combines all three core components:
1. Lorentzian ANN (Primary Signal Generator) - Identifies potential trading opportunities
2. Logistic Regression (Signal Confirmation) - Validates signals to reduce false positives
3. Chandelier Exit (Risk Management) - Manages position exits and protects capital

The IntegratedMLTrader class orchestrates the interaction between these components,
creating a complete trading system with signal generation, confirmation, risk management,
backtesting capabilities, and performance analysis.

Key Functions:
- Model training and persistence
- Signal generation combining all three components
- Backtesting with realistic trade execution
- Performance metrics calculation
- Visualization of trading results

This is the recommended entry point for using the complete trading system.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os
import sys

# Add parent directory to path to allow imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

# Import components
from strategies.LorentzianStrategy.lorentzian_classifier import LorentzianANN
from strategies.LorentzianStrategy.models.confirmation.logistic_regression_torch import LogisticRegression, LogisticConfig
from strategies.LorentzianStrategy.models.risk_management.chandelier_exit import ChandelierExit, ChandelierConfig
from strategies.LorentzianStrategy.indicators.base_torch_indicator import TorchIndicatorConfig

# Set up GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class IntegratedMLTrader:
    """
    Combines Lorentzian ANN, Logistic Regression, and Chandelier Exit 
    into a complete trading system.
    """
    
    def __init__(
            self,
            lookback_bars=50,        # How many historical bars to consider
            prediction_bars=4,       # How many bars into the future to predict
            k_neighbors=20,          # Number of nearest neighbors to consider
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True,
            adx_threshold=20.0,      # Minimum ADX value for trend trades
            regime_threshold=-0.1,   # Threshold for market regime detection
            atr_period=22,           # ATR period for Chandelier Exit
            atr_multiplier=3.0,      # ATR multiplier for stop distance
            logistic_threshold=0.6,  # Threshold for logistic confirmation
            model_dir="models"       # Directory to store models
        ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration objects
        torch_config = TorchIndicatorConfig(
            device=str(device),
            use_amp=True
        )
        
        logistic_config = LogisticConfig(
            use_deep=True,
            use_amp=True,
            threshold=logistic_threshold,
            device=str(device)
        )
        
        chandelier_config = ChandelierConfig(
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            device=str(device),
            use_amp=True
        )
        
        # Initialize components
        print("Initializing Lorentzian ANN...")
        self.lorentzian = LorentzianANN(
            lookback_bars=lookback_bars,
            prediction_bars=prediction_bars,
            k_neighbors=k_neighbors,
            use_regime_filter=use_regime_filter,
            use_volatility_filter=use_volatility_filter,
            use_adx_filter=use_adx_filter,
            adx_threshold=adx_threshold,
            regime_threshold=regime_threshold
        )
        
        print("Initializing Logistic Regression...")
        self.logistic = LogisticRegression(
            config=logistic_config,
            torch_config=torch_config
        )
        
        print("Initializing Chandelier Exit...")
        self.chandelier = ChandelierExit(
            config=chandelier_config
        )
        
        self.position = 0  # Current position: 1 for long, -1 for short, 0 for flat
        self.entry_price = 0.0  # Price at which position was entered
        self.stop_loss = 0.0  # Current stop loss level
        
        self.trades = []  # List to track trades
        self.equity_curve = []  # List to track equity
        
        print("Integrated ML Trader initialized")
    
    def prepare_indicators(self, df):
        """Add technical indicators needed for the trading models"""
        return df  # Assume indicators are already calculated in input dataframe

    def train_models(self, df):
        """Train all models on the provided data"""
        # Prepare features for the Lorentzian ANN
        print("Training Lorentzian ANN...")
        features = torch.tensor(
            df[['rsi_14', 'wt1', 'wt2', 'cci', 'adx']].values, 
            dtype=torch.float32
        ).to(device)
        prices = torch.tensor(df['close'].values, dtype=torch.float32).to(device)
        
        # Train the Lorentzian ANN
        self.lorentzian.fit(features, prices)
        
        # Train the Logistic Regression model
        print("Training Logistic Regression...")
        self.logistic_results = self.logistic.calculate_signals(df)
        
        # Note: Chandelier Exit doesn't need training as it's rule-based
        print("Models trained successfully")
        
        return self
        
    def generate_signals(self, df):
        """Generate trading signals by combining all components"""
        # Get lorentzian signals
        features = torch.tensor(
            df[['rsi_14', 'wt1', 'wt2', 'cci', 'adx']].values, 
            dtype=torch.float32
        ).to(device)
        lorentzian_signals = self.lorentzian.predict(features).cpu().numpy()
        
        # Get logistic regression signals
        logistic_results = self.logistic.calculate_signals(df)
        logistic_signals = logistic_results['buy_signals'].cpu().numpy() - logistic_results['sell_signals'].cpu().numpy()
        
        # Get chandelier exit signals
        chandelier_results = self.chandelier.calculate_signals(df)
        
        # Convert torch tensors to numpy arrays
        long_stops = chandelier_results['long_stop'].cpu().numpy()
        short_stops = chandelier_results['short_stop'].cpu().numpy()
        
        # Combine signals
        combined_signals = np.zeros(len(df))
        stops = np.zeros(len(df))
        
        for i in range(len(df)):
            # Only generate signal if both models agree
            if lorentzian_signals[i] > 0 and logistic_signals[i] > 0:
                combined_signals[i] = 1  # Long signal
                stops[i] = long_stops[i]
            elif lorentzian_signals[i] < 0 and logistic_signals[i] < 0:
                combined_signals[i] = -1  # Short signal
                stops[i] = short_stops[i]
            else:
                # Use chandelier exit for possible exit signals
                if self.position > 0 and df['close'].iloc[i] < long_stops[i]:
                    combined_signals[i] = 0  # Exit long
                elif self.position < 0 and df['close'].iloc[i] > short_stops[i]:
                    combined_signals[i] = 0  # Exit short
        
        # Store results in dataframe
        df['lorentzian_signal'] = lorentzian_signals
        df['logistic_signal'] = logistic_signals
        df['combined_signal'] = combined_signals
        df['long_stop'] = long_stops
        df['short_stop'] = short_stops
        
        return df
    
    def backtest(self, df, initial_capital=10000.0, position_size=0.1):
        """
        Backtest the strategy on historical data
        
        Args:
            df: DataFrame with OHLCV data and signals
            initial_capital: Starting capital
            position_size: Percentage of capital to use per trade
        
        Returns:
            DataFrame with backtest results
        """
        # Initialize backtest variables
        equity = initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        
        # Create columns for backtest results
        df['position'] = 0
        df['equity'] = initial_capital
        df['returns'] = 0.0
        df['trade_pnl'] = 0.0
        
        # Loop through data
        for i in range(1, len(df)):
            # Get current price and signal
            close = df['close'].iloc[i]
            signal = df['combined_signal'].iloc[i]
            
            # Update stop loss
            if position > 0:
                stop_loss = df['long_stop'].iloc[i]
            elif position < 0:
                stop_loss = df['short_stop'].iloc[i]
            
            # Check for stop loss
            if position > 0 and close < stop_loss:
                # Exit long position due to stop loss
                pnl = (close - entry_price) / entry_price * position_size * equity
                equity += pnl
                df.loc[df.index[i], 'trade_pnl'] = pnl
                position = 0
                
            elif position < 0 and close > stop_loss:
                # Exit short position due to stop loss
                pnl = (entry_price - close) / entry_price * position_size * equity
                equity += pnl
                df.loc[df.index[i], 'trade_pnl'] = pnl
                position = 0
            
            # Check for signal-based entry/exit
            if signal != 0 and position != signal:
                # Exit existing position if any
                if position != 0:
                    if position > 0:
                        pnl = (close - entry_price) / entry_price * position_size * equity
                    else:
                        pnl = (entry_price - close) / entry_price * position_size * equity
                    
                    equity += pnl
                    df.loc[df.index[i], 'trade_pnl'] = pnl
                
                # Enter new position
                position = signal
                entry_price = close
                
                # Set initial stop
                if position > 0:
                    stop_loss = df['long_stop'].iloc[i]
                elif position < 0:
                    stop_loss = df['short_stop'].iloc[i]
            
            # Update position and equity columns
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'equity'] = equity
            
            # Calculate returns
            if i > 0:
                df.loc[df.index[i], 'returns'] = (df['equity'].iloc[i] / df['equity'].iloc[i-1]) - 1
        
        # Calculate performance metrics
        total_trades = (df['trade_pnl'] != 0).sum()
        winning_trades = (df['trade_pnl'] > 0).sum()
        losing_trades = (df['trade_pnl'] < 0).sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = df.loc[df['trade_pnl'] > 0, 'trade_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df.loc[df['trade_pnl'] < 0, 'trade_pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(df.loc[df['trade_pnl'] > 0, 'trade_pnl'].sum() / 
                         df.loc[df['trade_pnl'] < 0, 'trade_pnl'].sum()) if df.loc[df['trade_pnl'] < 0, 'trade_pnl'].sum() != 0 else float('inf')
        
        # Calculate max drawdown
        df['equity_peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] / df['equity_peak'] - 1) * 100
        max_drawdown = df['drawdown'].min()
        
        # Print metrics
        print("\nBacktest Results:")
        print(f"Starting Capital: ${initial_capital:,.2f}")
        print(f"Ending Capital: ${df['equity'].iloc[-1]:,.2f}")
        print(f"Total Return: {(df['equity'].iloc[-1] / initial_capital - 1) * 100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        return df
    
    def plot_results(self, df):
        """Plot backtest results"""
        plt.figure(figsize=(12, 10))
        
        # Plot price and signals
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df['close'], label='Price')
        
        # Plot buy signals
        buy_points = df.index[df['combined_signal'] == 1]
        if len(buy_points) > 0:
            ax1.scatter(buy_points, df.loc[buy_points, 'close'], marker='^', color='green', label='Buy')
        
        # Plot sell signals
        sell_points = df.index[df['combined_signal'] == -1]
        if len(sell_points) > 0:
            ax1.scatter(sell_points, df.loc[sell_points, 'close'], marker='v', color='red', label='Sell')
        
        # Plot stops
        ax1.plot(df.index, df['long_stop'], 'g--', alpha=0.5, label='Long Stop')
        ax1.plot(df.index, df['short_stop'], 'r--', alpha=0.5, label='Short Stop')
        
        ax1.set_title('Price with Signals')
        ax1.legend()
        
        # Plot equity curve
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(df.index, df['equity'], label='Equity')
        ax2.set_title('Equity Curve')
        
        # Plot drawdown
        ax3 = plt.subplot(3, 1, 3)
        ax3.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown (%)')
        
        plt.tight_layout()
        plt.savefig('integrated_ml_trader_results.png')
        print("Results plot saved to 'integrated_ml_trader_results.png'")
        plt.close()
    
    def save_models(self):
        """Save all models to disk"""
        # Save Lorentzian ANN
        lorentzian_path = self.model_dir / "lorentzian_ann.pt"
        self.lorentzian.save_model(lorentzian_path)
        
        # Note: For simplicity, we're not implementing save functionality
        # for the other models in this example
        
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load all models from disk"""
        # Load Lorentzian ANN
        lorentzian_path = self.model_dir / "lorentzian_ann.pt"
        if lorentzian_path.exists():
            self.lorentzian.load_model(lorentzian_path)
            print(f"Loaded Lorentzian ANN from {lorentzian_path}")
        else:
            print(f"Lorentzian model not found at {lorentzian_path}")
        
        # Note: For simplicity, we're not implementing load functionality
        # for the other models in this example
        
        return self

def main():
    """Main function to run the integrated ML trader"""
    import time
    start_time = time.time()
    
    # Load data
    data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
    df = pd.read_feather(data_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    
    # Reduce dataset size for faster testing
    test_size = 10000  # Adjust as needed
    print(f"Original data size: {len(df)}")
    df = df.iloc[-test_size:]
    print(f"Reduced data size: {len(df)}")
    
    # Prepare indicators (assuming analyze_lorentzian_ann.py's prepare_indicators function)
    from analyze_lorentzian_ann import prepare_indicators
    df = prepare_indicators(df)
    
    # Create and train the integrated ML trader
    trader = IntegratedMLTrader(
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20,
        atr_period=22,
        atr_multiplier=3.0,
        logistic_threshold=0.6
    )
    
    # Train models
    trader.train_models(df)
    
    # Generate signals
    df = trader.generate_signals(df)
    
    # Backtest the strategy
    results = trader.backtest(df, initial_capital=10000.0)
    
    # Plot results
    trader.plot_results(results)
    
    # Save models
    trader.save_models()
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 