import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class DoubleBottomDetector(nn.Module):
    def __init__(self, input_size: int = 60, hidden_size: int = 64):
        """
        Simple LSTM model to detect double bottoms.
        
        Args:
            input_size: Number of candles to look at (default 60 candles = 5 hours on 5m chart)
            hidden_size: Size of the LSTM hidden layer
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM to process price sequence
        self.lstm = nn.LSTM(
            input_size=5,  # OHLCV data
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention layer to focus on important parts of the sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Final layers for pattern detection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Make prediction
        return self.classifier(context)

def prepare_data(df: pd.DataFrame, window_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for the model.
    
    Args:
        df: DataFrame with OHLCV data
        window_size: Number of candles to include in each sample
        
    Returns:
        Features tensor and labels tensor
    """
    # Normalize price and volume data
    price_cols = ['open', 'high', 'low', 'close']
    df_norm = df.copy()
    
    # Normalize prices by the first price in each window
    for col in price_cols:
        df_norm[col] = df_norm[col] / df_norm['close'].rolling(window_size).mean()
    
    # Normalize volume
    df_norm['volume'] = df_norm['volume'] / df_norm['volume'].rolling(window_size).mean()
    
    # Create sequences
    sequences = []
    labels = []
    
    for i in range(len(df) - window_size):
        # Get window of data
        window = df_norm.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
        sequences.append(window)
        
        # Label is 1 if a double bottom forms in the next 12 candles
        future_low = df['low'].iloc[i+window_size:i+window_size+12].min()
        current_low = df['low'].iloc[i+window_size-24:i+window_size].min()
        
        # Check for double bottom pattern:
        # 1. Two similar lows (within 2%)
        # 2. Price bounces at least 3% from each low
        # 3. Second low slightly higher than first (preferred)
        is_double_bottom = (
            abs(future_low - current_low) / current_low < 0.02 and  # Similar lows
            df['high'].iloc[i+window_size-24:i+window_size].max() > current_low * 1.03 and  # First bounce
            future_low >= current_low * 0.995  # Second low slightly higher (or equal)
        )
        
        labels.append(1.0 if is_double_bottom else 0.0)
    
    # Convert to tensors and move to GPU
    return (
        torch.FloatTensor(sequences).to(device),
        torch.FloatTensor(labels).reshape(-1, 1).to(device)
    )

def train_model(model: nn.Module, 
                train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor],
                epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.001) -> List[float]:
    """
    Train the model.
    
    Args:
        model: The DoubleBottomDetector model
        train_data: Tuple of (features, labels) for training
        val_data: Tuple of (features, labels) for validation
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        
    Returns:
        List of validation losses per epoch
    """
    model = model.to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0
        
        # Training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
            
            # Calculate validation metrics
            predictions = (val_outputs > 0.5).float()
            accuracy = (predictions == y_val).float().mean()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/best_double_bottom_model.pth')
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Training Loss: {total_loss/batches:.4f}')
        logger.info(f'  Validation Loss: {val_loss:.4f}')
        logger.info(f'  Validation Accuracy: {accuracy:.4f}')
    
    return val_losses

def main():
    # Print PyTorch device and CUDA information
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Load the data
    df_5m = pd.read_csv('data/sol_perp_5m.csv')
    df_15m = pd.read_csv('data/sol_perp_15m.csv')
    
    # Prepare the data
    X_5m, y_5m = prepare_data(df_5m)
    
    # Split into train and validation sets (80/20)
    split_idx = int(len(X_5m) * 0.8)
    train_data = (X_5m[:split_idx], y_5m[:split_idx])
    val_data = (X_5m[split_idx:], y_5m[split_idx:])
    
    # Create and train the model
    model = DoubleBottomDetector()
    val_losses = train_model(model, train_data, val_data)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 