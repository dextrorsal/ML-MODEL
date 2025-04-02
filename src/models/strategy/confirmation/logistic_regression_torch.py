"""
Enhanced Logistic Regression with PyTorch Implementation

This module provides a PyTorch-based implementation of logistic regression for trading,
combining traditional logistic regression with modern deep learning capabilities.

Features:
- GPU acceleration support
- Optional deep neural network mode
- Pine Script compatibility
- Real-time signal generation
- Customizable filters (volatility, volume)
- Proper error handling and validation
- Built-in visualization tools
- Backtesting metrics

Example:
    >>> indicator = EnhancedLogisticRegressionIndicator()
    >>> df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    >>> signals = indicator.generate_signals(df)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from ....features.technical.indicators.base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig
from models.configs import TradingConfig
from contextlib import nullcontext

@dataclass
class LogisticConfig:
    """Configuration for logistic regression model"""
    use_deep: bool = False
    use_amp: bool = False
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    hidden_size: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    lookback: int = 20
    threshold: float = 0.5
    volatility_filter: bool = True
    volume_filter: bool = True
    input_size: int = 1
    num_epochs: int = 100

@dataclass
class BacktestMetrics:
    """Container for backtesting metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

def minimax_scale(value: float, data_window: np.ndarray) -> float:
    """
    Scale value to window's min-max range with proper error handling
    
    Args:
        value: Value to scale
        data_window: Window of data to use for scaling
        
    Returns:
        Scaled value between 0 and 1
    """
    try:
        window_min = np.min(data_window)
        window_max = np.max(data_window)
        
        if np.isclose(window_max, window_min):
            return value
            
        return (value - window_min) / (window_max - window_min)
    except Exception as e:
        print(f"Error in minimax scaling: {str(e)}")
        return value

class SingleDimLogisticRegression:
    """Fallback single-dimension logistic regression model"""
    
    def __init__(self, lookback: int = 3, learning_rate: float = 0.0009, iterations: int = 1000):
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0,1] range"""
        try:
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if np.isclose(arr_max, arr_min):
                return arr
            return (arr - arr_min) / (arr_max - arr_min)
        except Exception as e:
            print(f"Error normalizing array: {str(e)}")
            return arr
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with overflow protection"""
        try:
            return 1 / (1 + np.exp(-np.clip(z, -100, 100)))
        except Exception as e:
            print(f"Error in sigmoid calculation: {str(e)}")
            return np.zeros_like(z)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Fit model and return (loss, prediction) for last point
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (final loss, prediction for last point)
        """
        try:
            # Normalize inputs
            X_norm = self.normalize_array(X)
            y_norm = self.normalize_array(y)
            
            # Initialize weight
            w = 0.0
            
            # Gradient descent
            for _ in range(self.iterations):
                # Forward pass
                z = X_norm * w
                pred = self.sigmoid(z)
                
                # Compute gradient
                error = pred - y_norm
                gradient = np.mean(X_norm * error)
                
                # Update weight
                w -= self.learning_rate * gradient
            
            # Final prediction for last point
            final_pred = self.sigmoid(X_norm[-1] * w)
            
            # Compute final loss
            final_loss = np.mean((self.sigmoid(X_norm * w) - y_norm) ** 2)
            
            return final_loss, final_pred
            
        except Exception as e:
            print(f"Error in single-dim logistic regression: {str(e)}")
            return 0.0, 0.5

class LogisticRegressionTorch(nn.Module):
    """
    PyTorch implementation of logistic regression for trading
    """
    
    def __init__(self, input_dim: int = 2):
        """
        Initialize the logistic regression model
        
        Args:
            input_dim: Dimension of the input features
        """
        super().__init__()
        
        # Simple logistic regression
        self.logistic = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        # Deep neural network for more complex patterns
        self.deep_network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self.use_deep = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the selected network
        
        Args:
            x: Input tensor with shape (batch_size, features)
            
        Returns:
            Predicted probabilities
        """
        # Ensure input has correct shape for the model
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Ensure x has the correct number of features
        if x.shape[1] != self.logistic[0].in_features:
            # If the number of features doesn't match, reshape or pad as needed
            if x.shape[1] < self.logistic[0].in_features:
                # Pad with zeros if fewer features than expected
                padding = torch.zeros(x.shape[0], self.logistic[0].in_features - x.shape[1], 
                                     device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            else:
                # Take only the first n features if there are more than expected
                x = x[:, :self.logistic[0].in_features]
        
        return self.deep_network(x) if self.use_deep else self.logistic(x)

class LogisticRegression(BaseTorchIndicator):
    """
    Logistic Regression indicator with PyTorch backend.
    
    This indicator combines traditional logistic regression with modern deep learning,
    providing both simple and complex models for trading signal generation.
    """
    
    def __init__(
        self,
        config: Optional[LogisticConfig] = None,
        torch_config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize the indicator with configuration settings.
        
        Args:
            config: Logistic regression specific configuration
            torch_config: PyTorch configuration for GPU/CPU
        """
        super().__init__(torch_config)
        
        self.config = config or LogisticConfig()
        self.metrics = BacktestMetrics()
        
        # Initialize model
        self.model = LogisticRegressionTorch().to(self.device)
        self.model.use_deep = self.config.use_deep
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Initialize fallback model
        self.fallback_model = SingleDimLogisticRegression(
            lookback=self.config.lookback,
            learning_rate=self.config.learning_rate,
            iterations=self.config.epochs
        )
        
    def prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare data for model input"""
        # Convert to tensors and normalize
        close = self.to_tensor(data['close'].values)
        volume = self.to_tensor(data['volume'].values)
        
        # Create features
        returns = torch.diff(close) / close[:-1]
        vol_change = torch.diff(volume) / volume[:-1]
        
        # Pad to maintain length
        returns = F.pad(returns, (1, 0))
        vol_change = F.pad(vol_change, (1, 0))
        
        # Stack features
        features = torch.stack([returns, vol_change], dim=1)
        
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, n_features) or (n_features,)
            
        Returns:
            Tensor of predictions
        """
        # Add batch dimension if input is 1D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Flatten if input has more than 2 dimensions
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Get predictions from model
        predictions = self.model(x)
        
        # Generate signals based on predictions
        buy_signals = torch.zeros_like(predictions, dtype=self.dtype)
        sell_signals = torch.zeros_like(predictions, dtype=self.dtype)
        
        # Set signals based on threshold
        buy_signals[predictions > self.config.threshold] = 1
        sell_signals[predictions < -self.config.threshold] = 1
        
        return {
            'predictions': predictions.squeeze(),
            'buy_signals': buy_signals.squeeze(),
            'sell_signals': sell_signals.squeeze()
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate trading signals from data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signals and predictions
        """
        # Prepare features
        features = self.prepare_data(data)
        
        # Generate predictions
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            results = self.forward(features)
        
        return results
    
    def train_epoch(self, features: torch.Tensor, targets: torch.Tensor):
        """Train model for one epoch"""
        self.model.train()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        total_loss = 0
        for batch_features, batch_targets in loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)
    
    def update_metrics(self, current_price: float, signal: int, last_signal: int) -> None:
        """Update trading metrics"""
        if last_signal != 0 and signal != last_signal:
            self.metrics.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal
            
            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = ((self.metrics.avg_win * 
                    (self.metrics.winning_trades - 1) + pnl) / 
                    self.metrics.winning_trades)
            else:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = ((self.metrics.avg_loss * 
                    (self.metrics.losing_trades - 1) + abs(pnl)) / 
                    self.metrics.losing_trades)
            
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (self.metrics.winning_trades / 
                    self.metrics.total_trades)
                
            if self.metrics.avg_loss > 0:
                self.metrics.profit_factor = ((self.metrics.avg_win * 
                    self.metrics.winning_trades) / 
                    (self.metrics.avg_loss * self.metrics.losing_trades))
        
        self.last_price = current_price
    
    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot signals and predictions"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
            
            # Plot price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            buy_points = df.index[signals['buy_signals'] == 1]
            sell_points = df.index[signals['sell_signals'] == 1]
            
            if len(buy_points) > 0:
                ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                          color='green', marker='^', label='Buy')
            if len(sell_points) > 0:
                ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
                          color='red', marker='v', label='Sell')
            
            ax1.set_title('Price with Logistic Regression Signals')
            ax1.legend()
            
            # Plot predictions
            ax2.plot(df.index, signals['predictions'], label='Prediction', color='blue')
            ax2.axhline(y=self.config.threshold, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=1-self.config.threshold, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('Model Predictions')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting signals: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get current trading metrics"""
        return {
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'profit_factor': self.metrics.profit_factor,
            'avg_win': self.metrics.avg_win,
            'avg_loss': self.metrics.avg_loss,
            'sharpe_ratio': self.metrics.sharpe_ratio
        }

    def _generate_signals_fallback(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using fallback single-dim logistic regression"""
        signals = pd.Series(0, index=df.index)
        current_signal = 0
        bar_count = 0

        # Get price data
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)

        for i in range(self.config.lookback, len(df)):
            try:
                # Prepare data windows
                price_window = close[i-self.config.lookback:i]
                volume_window = volume[i-self.config.lookback:i]
                
                # Get prediction
                loss, pred = self.fallback_model.fit_predict(price_window, volume_window)
                
                # Scale values
                if i >= self.config.lookback + 2:
                    scaled_loss = minimax_scale(loss, close[i-2:i])
                    scaled_pred = minimax_scale(pred, volume[i-2:i])
                    
                    # Generate signal
                    new_signal = self._calculate_signal(
                        current_price=close[i],
                        scaled_loss=scaled_loss,
                        scaled_pred=scaled_pred,
                        df=df,
                        i=i
                    )
                    
                    # Apply holding period logic
                    if new_signal != current_signal:
                        bar_count = 0
                    else:
                        bar_count += 1
                        
                    if bar_count >= 5 and current_signal != 0:
                        new_signal = 0
                        bar_count = 0
                    
                    signals.iloc[i] = new_signal
                    current_signal = new_signal
                    
            except Exception as e:
                print(f"Error in fallback signal generation at bar {i}: {str(e)}")
                signals.iloc[i] = 0

        return signals

    def _calculate_signal(self, current_price: float, scaled_loss: float, 
                         scaled_pred: float, df: pd.DataFrame, i: int) -> int:
        """Calculate trading signal with proper error handling"""
        try:
            if not self._passes_filter(df, i):
                return 0
                
            if scaled_pred > scaled_loss:
                return 1   # BUY
            elif scaled_pred < scaled_loss:
                return -1  # SELL
            return 0
            
        except Exception as e:
            print(f"Error in signal calculation: {str(e)}")
            return 0

    def _passes_filter(self, df: pd.DataFrame, i: int) -> bool:
        """Check if current bar passes filtering criteria"""
        try:
            if self.config.volatility_filter:
                atr1 = df['atr1'].iloc[i] if 'atr1' in df.columns else 0
                atr10 = df['atr10'].iloc[i] if 'atr10' in df.columns else 0
                if atr1 <= atr10:
                    return False

            if self.config.volume_filter:
                if 'volume' not in df.columns:
                    return True
                rsi_vol = df['volume_rsi'].iloc[i] if 'volume_rsi' in df.columns else 0
                if rsi_vol <= 49:
                    return False

            return True
            
        except Exception as e:
            raise RuntimeError(f"Filter check failed: {str(e)}") 