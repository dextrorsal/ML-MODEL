import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict
import talib
import time
from pathlib import Path


class TVLogisticRegression:
    """
    Optimized implementation of TradingView's Logistic Regression strategy
    that aligns with the TradingView code while leveraging GPU acceleration
    """
    def __init__(
        self,
        price_type: str = 'close',
        resolution: str = '15m',
        lookback: int = 3,
        normalization_lookback: int = 2,
        learning_rate: float = 0.0009,
        iterations: int = 1000,
        filter_signals: str = 'Volatility',
        use_optional_calc: bool = True,
        use_price_data: bool = True,
        holding_period: int = 5,
        batch_size: int = 512
    ):
        self.price_type = price_type.lower()
        self.resolution = resolution
        self.lookback = lookback
        self.normalization_lookback = normalization_lookback
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.filter_signals = filter_signals
        self.use_optional_calc = use_optional_calc
        self.use_price_data = use_price_data
        self.holding_period = holding_period
        self.batch_size = batch_size
        
        # Set up device - use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
    def get_price_data(self, df: pd.DataFrame) -> pd.Series:
        """Get price data based on selected type (matches TradingView options)"""
        if self.price_type == 'open':
            return df['open']
        elif self.price_type == 'high':
            return df['high']
        elif self.price_type == 'low':
            return df['low']
        elif self.price_type == 'hl2':
            return (df['high'] + df['low']) / 2
        elif self.price_type == 'hlc3':
            return (df['high'] + df['low'] + df['close']) / 3
        elif self.price_type == 'ohlc4':
            return (df['open'] + df['high'] + df['low'] + df['close']) / 4
        else:  # default to close
            return df['close']
    
    def minimax(self, series, min_val, max_val):
        """
        Minimax normalization - matches TradingView's formula:
        (max - min) * (ds - lo)/(hi - lo) + min
        """
        hi = series.rolling(window=self.normalization_lookback).max()
        lo = series.rolling(window=self.normalization_lookback).min()
        return (max_val - min_val) * (series - lo) / (hi - lo + 1e-8) + min_val
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features and technical indicators"""
        print("Preparing data...")
        df = df.copy()
        
        # Get base price data
        df['price'] = self.get_price_data(df)
        
        if self.use_optional_calc:
            # Use time as base and price as synthetic (TradingView easteregg)
            df['base'] = df.index.astype(np.int64) // 10**9
            df['synth'] = df['price']
        else:
            # Use price as base and generate synthetic data
            df['base'] = df['price']
            # TradingView formula: log(abs(pow(base_ds, 2) - 1) + .5)
            df['synth'] = np.log(np.abs(np.power(df['price'], 2) - 1) + 0.5)
        
        # Calculate volatility filter - TradingView uses ATR
        df['atr1'] = talib.ATR(
            df['high'].values, df['low'].values, df['close'].values, 
            timeperiod=1
        )
        df['atr10'] = talib.ATR(
            df['high'].values, df['low'].values, df['close'].values, 
            timeperiod=10
        )
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.Series:
        """Apply volatility filter like TradingView"""
        if self.filter_signals == 'Volatility':
            return df['atr1'] > df['atr10']
        elif self.filter_signals == 'Volume':
            return df['volume'] > df['volume'].rolling(14).mean()
        elif self.filter_signals == 'Both':
            return (df['atr1'] > df['atr10']) & (df['volume'] > df['volume'].rolling(14).mean())
        return pd.Series(True, index=df.index)
    
    def sigmoid(self, x):
        """Simple sigmoid function (same as TradingView)"""
        return 1.0 / (1.0 + torch.exp(-x))
    
    def dot_product(self, v1, v2):
        """Dot product (matches TradingView's sum(v * w, p))"""
        return torch.sum(v1 * v2)
    
    def logistic_regression(self, X, Y):
        """
        Logistic regression implementation that follows TradingView's approach:
        
        w = 0.0
        for i=1 to iterations
          hypothesis = sigmoid(dot(X, w))
          gradient = 1.0 / p * dot(X, hypothesis - Y)
          w = w - lr * gradient
        """
        # Initialize weight to 0 (matching TradingView)
        w = torch.zeros(1, device=self.device, requires_grad=True)
        
        # Convert inputs to tensors if they aren't already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        # Use optimizer for gradient descent
        optimizer = torch.optim.SGD([w], lr=self.learning_rate)
        
        # Training loop - exactly like TradingView's iterations
        for _ in range(self.iterations):
            optimizer.zero_grad()
            
            # Forward pass - calculating hypothesis
            z = X * w
            hypothesis = self.sigmoid(z)
            
            # Calculate loss
            epsilon = 1e-10  # Prevent log(0)
            loss = -torch.mean(
                Y * torch.log(hypothesis + epsilon) + 
                (1 - Y) * torch.log(1 - hypothesis + epsilon)
            )
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update weights - equivalent to TradingView's w := w - lr * gradient
            optimizer.step()
        
        # Final prediction
        with torch.no_grad():
            final_pred = self.sigmoid(X * w)
            
        return loss.item(), final_pred.cpu().numpy(), w.item()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using TradingView's approach"""
        start_time = time.time()
        
        # Prepare data
        df_prepared = self.prepare_data(df)
        
        # Apply filter
        filter_mask = self.apply_filters(df_prepared)
        
        signals = np.zeros(len(df_prepared))
        losses = np.zeros(len(df_prepared))
        predictions = np.zeros(len(df_prepared))
        
        # Initialize progress counter
        total_windows = len(df_prepared) - self.lookback
        print(f"Processing {total_windows} windows...")
        
        window_start = time.time()
        
        # Process windows in batches for speed
        for start_idx in range(0, total_windows, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_windows)
            batch_size_actual = end_idx - start_idx
            
            print(f"Processing batch {start_idx}-{end_idx} ({batch_size_actual} windows)")
            
            # Process each window in the batch
            for i in range(start_idx, end_idx):
                window_start_idx = i
                window_end_idx = i + self.lookback
                
                if window_end_idx <= len(df_prepared):
                    # Get window data
                    X = df_prepared['base'].iloc[window_start_idx:window_end_idx].values
                    Y = df_prepared['synth'].iloc[window_start_idx:window_end_idx].values
                    
                    # Train logistic regression on this window
                    loss, prediction, _ = self.logistic_regression(X, Y)
                    
                    # Get global index (position in original DataFrame)
                    global_idx = window_end_idx - 1
                    
                    # Store loss and prediction
                    losses[global_idx] = loss
                    predictions[global_idx] = prediction[-1]  # Last value
                    
                    # Check if filter is passed
                    has_filter = filter_mask.iloc[global_idx]
                    
                    if not has_filter:
                        continue
                    
                    # Scale loss and prediction to price range
                    price_val = df_prepared['price'].iloc[global_idx]
                    price_min = df_prepared['price'].rolling(
                        self.normalization_lookback).min().iloc[global_idx]
                    price_max = df_prepared['price'].rolling(
                        self.normalization_lookback).max().iloc[global_idx]
                    
                    # Scale using minimax formula
                    scaled_loss = (price_max - price_min) * loss + price_min
                    scaled_pred = (price_max - price_min) * prediction[-1] + price_min
                    
                    # Generate signals based on TradingView's method
                    if self.use_price_data:
                        # Using: base_ds < scaled_loss ? SELL : base_ds > scaled_loss ? BUY
                        if price_val < scaled_loss and has_filter:
                            signals[global_idx] = -1  # SELL
                        elif price_val > scaled_loss and has_filter:
                            signals[global_idx] = 1   # BUY
                    else:
                        # Using crossover/crossunder logic
                        if (global_idx > 0 and 
                            scaled_loss > scaled_pred and 
                            losses[global_idx-1] < predictions[global_idx-1] and 
                            has_filter):
                            signals[global_idx] = -1  # SELL (crossunder)
                        elif (global_idx > 0 and 
                              scaled_loss < scaled_pred and 
                              losses[global_idx-1] > predictions[global_idx-1] and 
                              has_filter):
                            signals[global_idx] = 1   # BUY (crossover)
            
            # Print progress
            progress = min(100, (end_idx / total_windows) * 100)
            print(f"Progress: {progress:.1f}% ({end_idx}/{total_windows})")
        
        print(f"Window processing completed in {time.time() - window_start:.2f} seconds")
        
        # Apply holding period logic (exactly like TradingView)
        df_prepared['signal'] = signals
        hp_counter = np.zeros(len(df_prepared))
        signal_series = df_prepared['signal'].values
        
        for i in range(1, len(signal_series)):
            if signal_series[i] != signal_series[i-1]:
                hp_counter[i] = 0
            else:
                hp_counter[i] = hp_counter[i-1] + 1
            
            # Close position after holding period
            if hp_counter[i] >= self.holding_period and signal_series[i] != 0:
                signal_series[i] = 0
        
        df_prepared['signal'] = signal_series
        
        elapsed_time = time.time() - start_time
        print(f"Signal generation completed in {elapsed_time:.2f} seconds")
        
        return df_prepared
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate trading metrics"""
        df['position'] = df['signal'].shift(1).fillna(0)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Calculate metrics
        total_trades = (df['position'].diff() != 0).sum()
        winning_trades = (df['strategy_returns'] > 0).sum()
        
        # Avoid division by zero
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        final_return = df['cumulative_returns'].iloc[-1] - 1 if len(df) > 0 else 0
        
        # Calculate max drawdown
        peak = df['cumulative_returns'].cummax()
        drawdown = (df['cumulative_returns'] / peak - 1)
        max_drawdown = drawdown.min()
        
        # Print metrics
        print("\nTrading Metrics:")
        print(f"total_trades: {total_trades}")
        print(f"winning_trades: {winning_trades}")
        print(f"win_rate: {win_rate:.2%}")
        print(f"final_return: {final_return:.2%}")
        print(f"max_drawdown: {max_drawdown:.2%}")
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_return': final_return,
            'max_drawdown': max_drawdown
        }
    
    def plot_results(self, df: pd.DataFrame) -> None:
        """Plot trading signals and cumulative returns"""
        plt.figure(figsize=(15, 10))
        
        # Plot price and signals
        plt.subplot(2, 1, 1)
        plt.title('Price with Trading Signals')
        plt.plot(df.index, df['close'], label='Price', color='blue', alpha=0.5)
        
        # Plot buy signals
        buy_points = df[df['signal'] == 1].index
        plt.scatter(buy_points, df.loc[buy_points, 'close'], 
                   marker='^', color='green', label='Buy Signal')
        
        # Plot sell signals
        sell_points = df[df['signal'] == -1].index
        plt.scatter(sell_points, df.loc[sell_points, 'close'], 
                   marker='v', color='red', label='Sell Signal')
        
        plt.legend()
        
        # Plot returns
        plt.subplot(2, 1, 2)
        plt.title('Cumulative Returns')
        plt.plot(df.index, df['cumulative_returns'], 
                label='Strategy Returns', color='green')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        plt.ylabel('Cumulative Returns')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('logistic_tv_results.png')
        print("\nResults have been plotted to 'logistic_tv_results.png'")


def main():
    start_time = time.time()
    
    # Load data
    print("Loading data...")
    data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
    df = pd.read_feather(data_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    
    # Resample to 15min
    print("Resampling to 15min timeframe...")
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # For faster testing (comment out for full run)
    df_15m = df_15m.iloc[-5000:]
    
    print(f"Loaded data shape: {df_15m.shape}")
    print(f"Date range: {df_15m.index[0]} to {df_15m.index[-1]}")
    
    # Initialize the trader
    trader = TVLogisticRegression(
        price_type='close',
        resolution='15m',
        lookback=3,          # TradingView default
        normalization_lookback=2,  # TradingView recommends 2-5 for BTC
        learning_rate=0.0009,  # TradingView default
        iterations=1000,     # TradingView default
        filter_signals='Volatility',  # TradingView default
        use_optional_calc=True,  # TradingView easteregg
        use_price_data=True,    # TradingView setting
        holding_period=5,      # TradingView default
        batch_size=512         # Smaller batch for reliability
    )
    
    # Generate signals
    df_15m = trader.generate_signals(df_15m)
    
    # Calculate metrics
    trader.calculate_metrics(df_15m)
    
    # Plot results
    trader.plot_results(df_15m)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()