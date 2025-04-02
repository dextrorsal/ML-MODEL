"""Configuration classes for ML models."""

from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    

@dataclass
class PatternRecognitionConfig(BaseModelConfig):
    """Configuration for pattern recognition model."""
    input_size: int = 100  # Lookback window
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    patterns_to_detect: List[str] = None
    min_pattern_size: int = 5
    max_pattern_size: int = 50
    confidence_threshold: float = 0.75

    def __post_init__(self):
        if self.patterns_to_detect is None:
            self.patterns_to_detect = [
                "double_bottom",
                "double_top",
                "head_and_shoulders",
                "inverse_head_and_shoulders",
                "triangle",
                "wedge"
            ]


@dataclass
class LorentzianConfig(BaseModelConfig):
    """Configuration for Lorentzian classifier."""
    lookback: int = 60
    neighbor_count: int = 8
    dimensions: List[int] = None
    time_scaling: float = 2.0
    min_neighbors: int = 4
    use_dynamic_scaling: bool = True
    use_volume_factor: bool = True
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = [32, 64, 32]


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_open_trades: int = 3
    min_volatility: float = 0.01
    max_correlation: float = 0.7
    

@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    timeframes: List[str] = None
    train_split: float = 0.8
    validation_split: float = 0.1
    sequence_length: int = 100
    stride: int = 1
    feature_columns: List[str] = None
    target_column: str = "target"
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.feature_columns is None:
            self.feature_columns = [
                "open", "high", "low", "close", "volume",
                "rsi", "macd", "wt1", "wt2", "cci"
            ]


# Default configurations
DEFAULT_PATTERN_CONFIG = PatternRecognitionConfig()
DEFAULT_LORENTZIAN_CONFIG = LorentzianConfig()
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_DATA_CONFIG = DataConfig() 