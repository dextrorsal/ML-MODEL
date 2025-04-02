import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from ..features.pattern_feature_extractor import PatternFeatureExtractor

class PatternAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, feature_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class PatternRecognitionModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature processing layers
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_dropout = nn.Dropout(dropout)
        
        # Pattern attention mechanism
        self.pattern_attention = PatternAttention(feature_dim)
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layers
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        self.pattern_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 classes: no pattern, bullish, bearish
        )
        
        self.confidence_regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, feature_dim)
        
        # Normalize and apply dropout to features
        x = self.feature_norm(x)
        x = self.feature_dropout(x)
        
        # Apply pattern attention
        x = x.transpose(0, 1)  # (seq_len, batch, feature_dim)
        x = self.pattern_attention(x)
        x = x.transpose(0, 1)  # (batch, seq_len, feature_dim)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output for predictions
        last_hidden = lstm_out[:, -1, :]
        
        # Get pattern classification and confidence
        pattern_logits = self.pattern_classifier(last_hidden)
        confidence = self.confidence_regressor(last_hidden)
        
        return pattern_logits, confidence

class PatternTrader:
    def __init__(
        self,
        timeframes: List[str] = ["1h", "4h", "1d"],
        feature_window: int = 100,
        model_hidden_dim: int = 128,
        model_layers: int = 2,
        learning_rate: float = 0.001
    ):
        """
        Initialize Pattern Trading Model
        
        Args:
            timeframes: List of timeframes to analyze
            feature_window: Number of bars to look back for feature extraction
            model_hidden_dim: Hidden dimension of the LSTM
            model_layers: Number of LSTM layers
            learning_rate: Learning rate for optimization
        """
        self.timeframes = timeframes
        self.feature_window = feature_window
        
        # Initialize feature extractor
        self.feature_extractor = PatternFeatureExtractor(timeframes)
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, feature_names = self.feature_extractor.get_training_features(
            {tf: pd.DataFrame() for tf in timeframes},
            window_size=1
        )
        self.feature_dim = len(feature_names)
        
        self.model = PatternRecognitionModel(
            feature_dim=self.feature_dim,
            hidden_dim=model_hidden_dim,
            num_layers=model_layers
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.pattern_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
    def prepare_features(self, data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Convert raw data to model features"""
        features, _ = self.feature_extractor.get_training_features(
            data,
            window_size=self.feature_window
        )
        return torch.FloatTensor(features).to(self.device)
    
    def train_step(self, 
                  features: torch.Tensor,
                  pattern_labels: torch.Tensor,
                  confidence_labels: torch.Tensor
                  ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        pattern_logits, confidence = self.model(features)
        
        # Calculate losses
        pattern_loss = self.pattern_criterion(pattern_logits, pattern_labels)
        confidence_loss = self.confidence_criterion(confidence.squeeze(), confidence_labels)
        
        # Combined loss
        total_loss = pattern_loss + confidence_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "pattern_loss": pattern_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for new data
        
        Returns:
            Tuple of (pattern_predictions, confidence_scores)
            pattern_predictions: 0=no pattern, 1=bullish, 2=bearish
        """
        self.model.eval()
        features = self.prepare_features(data)
        
        with torch.no_grad():
            pattern_logits, confidence = self.model(features)
            pattern_preds = torch.argmax(pattern_logits, dim=1).cpu().numpy()
            confidence_scores = confidence.squeeze().cpu().numpy()
            
        return pattern_preds, confidence_scores
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 