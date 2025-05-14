import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.pattern_detection import PatternDetector
from src.models.pattern_recognition_model import PatternRecognitionModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(timeframe: str = '5m', n_samples: int = 1000):
    """
    Load and prepare data for a specific timeframe
    """
    # TODO: Replace with your actual data loading logic
    # For now, we'll create some sample data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq=timeframe)
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, n_samples),
        'high': np.random.normal(102, 10, n_samples),
        'low': np.random.normal(98, 10, n_samples),
        'close': np.random.normal(101, 10, n_samples),
        'volume': np.random.normal(1000000, 100000, n_samples)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['high'] = np.maximum(data[['open', 'high', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'low', 'close']].min(axis=1), data['low'])
    
    return data

def test_pattern_detection(timeframe: str = '5m'):
    """
    Test pattern detection on a specific timeframe
    """
    logger.info(f"\n{'='*50}\nTesting Pattern Detection on {timeframe} timeframe\n{'='*50}")
    
    # 1. Load and prepare data
    data = load_and_prepare_data(timeframe)
    logger.info(f"Loaded {len(data)} samples of {timeframe} data")
    
    # 2. Initialize pattern detector
    # Adjust min_swing_size based on timeframe
    swing_sizes = {
        '1d': 0.03,  # Larger swings for daily
        '4h': 0.025, # Medium swings for 4h
        '1h': 0.02,  # Standard swings for 1h
        '15m': 0.015, # Smaller swings for 15m
        '5m': 0.01   # Smallest swings for 5m
    }
    min_swing_size = swing_sizes.get(timeframe, 0.02)
    detector = PatternDetector(min_swing_size=min_swing_size)
    
    # 3. Label the data
    labeled_data = detector.label_training_data(data)
    
    # 4. Print pattern statistics
    bullish_fractals = labeled_data['bullish_fractal'].sum()
    bearish_fractals = labeled_data['bearish_fractal'].sum()
    w_bottoms = (labeled_data['zigzag_pattern'] == 'W_bottom').sum()
    m_tops = (labeled_data['zigzag_pattern'] == 'M_top').sum()
    
    logger.info("\nPattern Statistics:")
    logger.info(f"Bullish Fractals: {bullish_fractals}")
    logger.info(f"Bearish Fractals: {bearish_fractals}")
    logger.info(f"W Bottoms: {w_bottoms}")
    logger.info(f"M Tops: {m_tops}")
    
    # 5. Prepare features for model
    sequences, labels = detector.prepare_pattern_features(data)
    logger.info(f"\nPrepared {len(sequences)} sequences with shape {sequences.shape}")
    
    return sequences, labels, labeled_data

def train_hierarchical_timeframes(timeframes: list, epochs_per_timeframe: dict):
    """
    Train the model hierarchically, starting from higher timeframes
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = PatternRecognitionModel().to(device)
    
    # Training history for all timeframes
    all_history = {}
    
    for timeframe in timeframes:
        logger.info(f"\n{'#'*70}")
        logger.info(f"Training on {timeframe} timeframe")
        logger.info(f"{'#'*70}")
        
        # Get data for this timeframe
        sequences, labels, labeled_data = test_pattern_detection(timeframe)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(labels)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Adjust batch size based on timeframe
        batch_sizes = {
            '1d': 16,   # Smaller batch for more stable learning on major patterns
            '4h': 24,
            '1h': 32,
            '15m': 48,
            '5m': 64    # Larger batch for noisy data
        }
        batch_size = batch_sizes.get(timeframe, 32)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Adjust learning rate based on timeframe
        learning_rates = {
            '1d': 0.0005,  # Slower learning for stable patterns
            '4h': 0.0007,
            '1h': 0.001,
            '15m': 0.002,
            '5m': 0.003    # Faster learning for noisy data
        }
        lr = learning_rates.get(timeframe, 0.001)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Train for specified number of epochs
        history = train_pattern_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=epochs_per_timeframe[timeframe],
            device=device
        )
        
        all_history[timeframe] = history
        
        # Save model checkpoint after each timeframe
        checkpoint_dir = os.path.join("models", "checkpoints", "pattern_recognition")
        timeframe_dir = os.path.join(checkpoint_dir, timeframe)
        os.makedirs(timeframe_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            timeframe_dir,
            f"model_{timestamp}.pt"
        )
        
        # Save metadata about the training
        metadata = {
            'timeframe': timeframe,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'training_params': {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs_per_timeframe[timeframe]
            }
        }
        
        torch.save(metadata, checkpoint_path)
        
        logger.info(f"\nCompleted training on {timeframe}")
        logger.info(f"Final training loss: {history[-1]['train_loss']:.4f}")
        logger.info(f"Final validation loss: {history[-1]['val_loss']:.4f}")
        logger.info(f"Final accuracy: {history[-1]['accuracy']*100:.2f}%")
        logger.info(f"Model saved to {checkpoint_path}")
        
        # Short pause between timeframes
        logger.info("\nWaiting 5 seconds before next timeframe...")
        import time
        time.sleep(5)
    
    return all_history, model

def main():
    """
    Main test function with hierarchical timeframe training
    """
    # Define timeframes from highest to lowest
    timeframes = ['1d', '4h', '1h', '15m', '5m']
    
    # Define epochs for each timeframe
    epochs_per_timeframe = {
        '1d': 20,   # More epochs for stable patterns
        '4h': 15,
        '1h': 10,
        '15m': 8,
        '5m': 5     # Fewer epochs for noisy data
    }
    
    # Train hierarchically
    all_history, final_model = train_hierarchical_timeframes(
        timeframes, 
        epochs_per_timeframe
    )
    
    # Plot training progress across timeframes
    plt.figure(figsize=(15, 10))
    for timeframe in timeframes:
        history = all_history[timeframe]
        epochs = range(len(history))
        plt.plot([h['val_accuracy'] for h in history], 
                label=f'{timeframe} Validation Accuracy')
    
    plt.title('Training Progress Across Timeframes')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()
    
    logger.info("\nTraining completed for all timeframes!")
    logger.info("Check training_progress.png for visualization")

if __name__ == "__main__":
    main() 