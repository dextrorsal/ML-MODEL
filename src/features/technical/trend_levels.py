import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

class TrendState(Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"

@dataclass
class TrendLevel:
    high: float
    low: float
    mid: float
    trend: TrendState
    strength: float  # delta percentage between up/down counts
    timestamp: pd.Timestamp
    bars_in_trend: int
    count_up: int = 0
    count_down: int = 0
    # ML-specific features
    momentum: float = 0.0
    volatility: float = 0.0
    volume_strength: float = 0.0
    level_support_tests: int = 0
    level_resistance_tests: int = 0

class TrendLevelsAnalyzer(BaseTorchIndicator):
    def __init__(
        self, 
        lookback_period: int = 30,
        torch_config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize TrendLevels Analyzer with PyTorch backend
        
        Args:
            lookback_period: Length input to determine range of bars (default 30)
            torch_config: PyTorch configuration for GPU/CPU
        """
        super().__init__(torch_config)
        self.lookback_period = lookback_period
        self.current_trend = TrendState.NEUTRAL
        self.bars_in_trend = 0
        self.count_up = 0
        self.count_down = 0
        
        # Track previous values for trend changes
        self.prev_high = None
        self.prev_low = None
        self.prev_mid = None
        
    def calculate_trend_levels(self, df: pd.DataFrame) -> List[TrendLevel]:
        """
        Calculate trend levels using PyTorch operations
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of TrendLevel objects
        """
        # Convert data to tensors and move to device
        high = self.to_tensor(df['high'])
        low = self.to_tensor(df['low'])
        close = self.to_tensor(df['close'])
        volume = self.to_tensor(df['volume']) if 'volume' in df else None
        
        levels = []
        self.bars_in_trend = 0
        self.count_up = 0
        self.count_down = 0
        
        # Use PyTorch's unfold for rolling windows
        high_windows = high.unfold(0, self.lookback_period, 1)
        low_windows = low.unfold(0, self.lookback_period, 1)
        
        # Calculate highest and lowest efficiently
        highest_vals, _ = torch.max(high_windows, dim=1)
        lowest_vals, _ = torch.min(low_windows, dim=1)
        
        # Pre-calculate momentum and volatility
        momentum = self.calculate_momentum(close)
        volatility = self.calculate_volatility(high, low)
        
        for i in range(self.lookback_period, len(df)):
            # Get current values
            curr_high = high[i].item()
            curr_low = low[i].item()
            curr_close = close[i].item()
            
            # Check for trend shifts using tensor operations
            if curr_high == highest_vals[i-self.lookback_period]:
                if self.current_trend != TrendState.UP:
                    self.bars_in_trend = 1
                    self.count_up = 0
                    self.count_down = 0
                    self.current_trend = TrendState.UP
                else:
                    self.bars_in_trend += 1
            elif curr_low == lowest_vals[i-self.lookback_period]:
                if self.current_trend != TrendState.DOWN:
                    self.bars_in_trend = 1
                    self.count_up = 0
                    self.count_down = 0
                    self.current_trend = TrendState.DOWN
                else:
                    self.bars_in_trend += 1
            else:
                self.bars_in_trend += 1
            
            # Calculate trend levels with PyTorch
            if self.bars_in_trend > 1:
                trend_slice = slice(i-self.bars_in_trend+1, i+1)
                h1 = torch.max(high[trend_slice]).item()
                l1 = torch.min(low[trend_slice]).item()
                m1 = (h1 + l1) / 2
                
                # Update counts using tensor operations
                above_mid = (close[trend_slice] > m1).sum().item()
                below_mid = (close[trend_slice] < m1).sum().item()
                self.count_up = int(above_mid)
                self.count_down = int(below_mid)
                
                # Calculate delta percentage
                total = self.count_up + self.count_down
                delta_percent = ((self.count_up - self.count_down) / total * 100) if total > 0 else 0
                
                # Calculate ML features
                curr_momentum = momentum[i].item()
                curr_volatility = volatility[i].item()
                
                # Calculate volume strength if volume data exists
                vol_strength = 0.0
                if volume is not None:
                    vol_ma = torch.mean(volume[trend_slice])
                    vol_strength = (volume[i] / vol_ma).item() if vol_ma > 0 else 1.0
                
                # Calculate level tests
                support_tests = self.calculate_level_tests(close[trend_slice], l1, "support")
                resistance_tests = self.calculate_level_tests(close[trend_slice], h1, "resistance")
                
                levels.append(
                    TrendLevel(
                        high=h1,
                        low=l1,
                        mid=m1,
                        trend=self.current_trend,
                        strength=delta_percent,
                        timestamp=df.index[i],
                        bars_in_trend=self.bars_in_trend,
                        count_up=self.count_up,
                        count_down=self.count_down,
                        momentum=curr_momentum,
                        volatility=curr_volatility,
                        volume_strength=vol_strength,
                        level_support_tests=support_tests,
                        level_resistance_tests=resistance_tests
                    )
                )
            else:
                levels.append(
                    TrendLevel(
                        high=float('nan'),
                        low=float('nan'),
                        mid=float('nan'),
                        trend=self.current_trend,
                        strength=0,
                        timestamp=df.index[i],
                        bars_in_trend=1,
                        count_up=0,
                        count_down=0,
                        momentum=0.0,
                        volatility=0.0,
                        volume_strength=1.0,
                        level_support_tests=0,
                        level_resistance_tests=0
                    )
                )
            
            # Store previous values
            if levels:
                self.prev_high = levels[-1].high
                self.prev_low = levels[-1].low
                self.prev_mid = levels[-1].mid
        
        return levels
    
    def calculate_momentum(self, close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate price momentum using PyTorch"""
        momentum = (close - F.pad(close, (period, 0))[:len(close)]) / F.pad(close, (period, 0))[:len(close)]
        return F.pad(momentum, (0, period), value=0)
    
    def calculate_volatility(self, high: torch.Tensor, low: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate price volatility using PyTorch"""
        ranges = high - low
        volatility = self.torch_sma(ranges, period)
        return volatility
    
    def calculate_level_tests(self, prices: torch.Tensor, level: float, test_type: str) -> int:
        """Calculate how many times a level was tested"""
        if test_type == "support":
            tests = ((prices <= level * 1.001) & (prices >= level * 0.999)).sum().item()
        else:  # resistance
            tests = ((prices >= level * 0.999) & (prices <= level * 1.001)).sum().item()
        return int(tests)
    
    def get_trend_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Get trend change signals with ML features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of signal dictionaries with timestamp and type
        """
        signals = []
        levels = self.calculate_trend_levels(df)
        
        for i in range(1, len(levels)):
            # Detect trend changes
            if levels[i].trend != levels[i-1].trend:
                signals.append({
                    'timestamp': levels[i].timestamp,
                    'type': 'up' if levels[i].trend == TrendState.UP else 'down',
                    'price': df['low'].iloc[i] if levels[i].trend == TrendState.UP else df['high'].iloc[i],
                    'strength': levels[i].strength,
                    'bars_in_trend': levels[i].bars_in_trend,
                    # ML features
                    'momentum': levels[i].momentum,
                    'volatility': levels[i].volatility,
                    'volume_strength': levels[i].volume_strength,
                    'level_tests': (levels[i].level_support_tests if levels[i].trend == TrendState.UP 
                                  else levels[i].level_resistance_tests)
                })
        
        return signals
    
    def get_gradient_colors(self, strength: float) -> str:
        """Convert strength to color gradient using PyTorch"""
        # Normalize strength to 0-1 range
        normalized = (strength + 100) / 200
        
        # Create gradient from fuchsia to lime using tensor operations
        colors = torch.tensor([
            [255, 0, 255],  # fuchsia
            [0, 255, 0]     # lime
        ], device=self.device)
        
        # Interpolate colors
        t = torch.tensor([normalized], device=self.device)
        rgb = (1 - t) * colors[0] + t * colors[1]
        
        # Convert to hex
        r, g, b = map(int, rgb[0].tolist())
        return f'#{r:02x}{g:02x}{b:02x}' 