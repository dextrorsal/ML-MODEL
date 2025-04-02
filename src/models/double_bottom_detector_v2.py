import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import json
import os
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import Json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    # Get current GPU device
    current_device = torch.cuda.current_device()
    
    # Print detailed GPU information
    logger.info("\nGPU Information:")
    logger.info(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    logger.info(f"GPU Architecture: {torch.cuda.get_device_properties(current_device).major}.{torch.cuda.get_device_properties(current_device).minor}")
    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
    logger.info(f"ROCm Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    
    # Test GPU with a small tensor operation
    logger.info("\nTesting GPU Operation:")
    test_tensor = torch.rand(1000, 1000).to(device)
    result = torch.mm(test_tensor, test_tensor)
    logger.info(f"✓ GPU computation test successful")
    logger.info(f"Test tensor device: {test_tensor.device}")
    
    # Clear test tensors
    del test_tensor
    del result
    torch.cuda.empty_cache()
else:
    logger.warning("No GPU detected! Training will be slow on CPU.")

# Neon database configuration
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_SO6bvFxmj1QL@ep-shy-wave-a5ilhlh4-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(NEON_DATABASE_URL)
        
    def start_training_session(self, model_name: str, total_epochs: int, training_params: Dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_sessions 
                (model_name, total_epochs, status, training_params)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (model_name, total_epochs, 'running', Json(training_params)))
            session_id = cur.fetchone()[0]
            self.conn.commit()
            return session_id
    
    def update_training_progress(self, session_id: int, epoch: int, loss: float, 
                               accuracy: float, is_best: bool = False):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE training_sessions
                SET epochs_completed = %s,
                    current_loss = %s,
                    best_accuracy = CASE WHEN %s > best_accuracy OR best_accuracy IS NULL THEN %s ELSE best_accuracy END
                WHERE id = %s
            """, (epoch, loss, accuracy, accuracy, session_id))
            
            # Save checkpoint
            cur.execute("""
                INSERT INTO model_checkpoints 
                (session_id, epoch, loss, accuracy)
                VALUES (%s, %s, %s, %s)
            """, (session_id, epoch, loss, accuracy))
            
            self.conn.commit()
    
    def complete_training_session(self, session_id: int, final_metrics: Dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE training_sessions
                SET status = 'completed',
                    end_time = NOW(),
                    metrics = %s
                WHERE id = %s
            """, (Json(final_metrics), session_id))
            self.conn.commit()
    
    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get the best performing checkpoint from all sessions."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT mc.session_id, mc.epoch, mc.loss, mc.accuracy, ts.model_name
                FROM model_checkpoints mc
                JOIN training_sessions ts ON mc.session_id = ts.id
                WHERE ts.status = 'completed'
                ORDER BY mc.loss ASC
                LIMIT 1
            """)
            result = cur.fetchone()
            if result:
                return {
                    'session_id': result[0],
                    'epoch': result[1],
                    'loss': result[2],
                    'accuracy': result[3],
                    'model_name': result[4]
                }
            return None

class DoubleBottomDetector(nn.Module):
    def __init__(self, input_size: int = 60, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Feature extraction with proper BatchNorm
        self.feature_extractor = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),  # BatchNorm on feature dimension
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        # Reshape for feature extraction (combine batch and sequence dimensions)
        x = x.view(-1, features)  # Shape: (batch_size * seq_len, features)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)  # Restore shape: (batch_size, seq_len, hidden)
        
        # Process sequence
        lstm_out, _ = self.lstm(x)
        
        # Global pooling
        x = lstm_out.transpose(1, 2)  # Shape: (batch_size, channels, seq_len)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)  # Flatten: (batch_size, channels)
        
        # Classification
        return self.classifier(x)

def prepare_data(df: pd.DataFrame, window_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    price_cols = ['open', 'high', 'low', 'close']
    df_norm = df.copy()
    
    # Z-score normalization with rolling window
    for col in price_cols:
        rolling_mean = df_norm[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df_norm[col].rolling(window=window_size, min_periods=1).std()
        df_norm[col] = (df_norm[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Volume normalization
    df_norm['volume'] = np.log1p(df_norm['volume'])
    rolling_vol_mean = df_norm['volume'].rolling(window=window_size, min_periods=1).mean()
    rolling_vol_std = df_norm['volume'].rolling(window=window_size, min_periods=1).std()
    df_norm['volume'] = (df_norm['volume'] - rolling_vol_mean) / (rolling_vol_std + 1e-8)
    
    # Convert to numpy array first
    sequences = []
    labels = []
    
    for i in range(len(df) - window_size):
        # Get the sequence
        seq = df_norm.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
        
        # Skip sequences with any infinite or NaN values
        if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
            continue
            
        future_low = df['low'].iloc[i+window_size:i+window_size+12].min()
        current_low = df['low'].iloc[i+window_size-24:i+window_size].min()
        
        # Enhanced double bottom detection with safety checks
        is_double_bottom = (
            current_low > 0 and  # Ensure positive values
            abs(future_low - current_low) / (current_low + 1e-8) < 0.02 and  # Similar lows
            df['high'].iloc[i+window_size-24:i+window_size].max() > current_low * 1.03 and  # First bounce
            future_low >= current_low * 0.995 and  # Second low slightly higher
            df['volume'].iloc[i+window_size-24:i+window_size].mean() > df['volume'].iloc[i:i+window_size-24].mean()  # Volume confirmation
        )
        
        sequences.append(seq)
        labels.append(1.0 if is_double_bottom else 0.0)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Check class balance
    positive_samples = np.sum(labels)
    balance_ratio = positive_samples / len(labels)
    logger.info(f"Class balance - Double bottoms: {balance_ratio:.2%}")
    
    if balance_ratio < 0.01:
        logger.warning("Very few double bottom patterns detected. Consider adjusting criteria.")
    
    # Convert to tensors and ensure they're float32
    return (
        torch.FloatTensor(sequences).to(device),
        torch.FloatTensor(labels).reshape(-1, 1).to(device)
    )

def load_best_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, models_dir: str) -> Tuple[float, int, int]:
    """Load the best checkpoint if available."""
    db_manager = DatabaseManager()
    best_checkpoint_info = db_manager.get_best_checkpoint()
    
    if best_checkpoint_info:
        checkpoint_path = os.path.join(
            models_dir, 
            f"double_bottom_model_session_{best_checkpoint_info['session_id']}_epoch_{best_checkpoint_info['epoch']}.pth"
        )
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading best checkpoint from session {best_checkpoint_info['session_id']}, epoch {best_checkpoint_info['epoch']}")
            checkpoint = torch.load(checkpoint_path)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return checkpoint['loss'], checkpoint['epoch'], best_checkpoint_info['session_id']
    
    return float('inf'), 0, None

def train_model(model: nn.Module, 
                train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor],
                db_manager: DatabaseManager,
                epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.0005,
                max_time_hours: float = 8.0) -> List[float]:
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join('src', 'models')
    if not os.path.exists(models_dir):
        logger.info(f"Creating models directory at {models_dir}")
        os.makedirs(models_dir)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    # Try to load the best checkpoint
    best_val_loss, start_epoch, prev_session_id = load_best_checkpoint(model, optimizer, models_dir)
    
    # Add gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Longer cycle length for overnight training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Longer initial cycle
        T_mult=2,  # Double the cycle length each time
        eta_min=1e-6
    )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # More balanced loss weighting
    pos_weight = torch.tensor([(1 - y_train.mean()) / (y_train.mean() + 1e-8)]).clamp(1.0, 10.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    training_params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_training_hours': max_time_hours,
        'device': str(device),
        'model_architecture': 'Bidirectional LSTM with Enhanced Attention',
        'continued_from_session': prev_session_id
    }
    session_id = db_manager.start_training_session('DoubleBottomDetector', epochs, training_params)
    
    val_losses = []
    start_time = time.time()
    patience = 20  # Increased patience
    no_improve_count = 0
    restart_count = 0
    max_restarts = 5  # More restarts
    min_epochs_per_restart = 30  # Ensure minimum training time per restart
    
    try:
        while restart_count < max_restarts and (time.time() - start_time) / 3600 <= max_time_hours:
            epochs_this_restart = 0
            best_loss_this_restart = float('inf')
            
            try:
                for epoch in range(start_epoch, epochs):
                    if (time.time() - start_time) / 3600 > max_time_hours:
                        logger.info(f"Reached maximum training time of {max_time_hours} hours")
                        break
                    
                    epochs_this_restart += 1
                    
                    # Training phase
                    model.train()
                    total_loss = 0
                    batches = 0
                    
                    # Enhanced data shuffling with random augmentation
                    indices = torch.randperm(len(X_train))
                    X_train_shuffled = X_train[indices]
                    y_train_shuffled = y_train[indices]
                    
                    # Add small random noise for regularization
                    if torch.rand(1).item() < 0.5:  # 50% chance
                        X_train_shuffled = X_train_shuffled + torch.randn_like(X_train_shuffled) * 0.01
                    
                    for i in range(0, len(X_train_shuffled), batch_size):
                        batch_X = X_train_shuffled[i:i+batch_size]
                        batch_y = y_train_shuffled[i:i+batch_size]
                        
                        optimizer.zero_grad()
                        
                        try:
                            with torch.amp.autocast('cuda'):
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                            
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            
                            total_loss += loss.item()
                            batches += 1
                            
                        except RuntimeError as e:
                            if "unscale_() has already been called" in str(e):
                                continue
                            else:
                                raise e
                    
                    # Validation phase
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs, y_val)
                        val_losses.append(val_loss.item())
                        
                        predictions = torch.sigmoid(val_outputs) > 0.5
                        accuracy = (predictions == y_val).float().mean()
                        
                        # Cyclical learning rate update
                        if epochs_this_restart >= min_epochs_per_restart:
                            scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        # Track best performance for this restart
                        is_best = val_loss < best_val_loss
                        is_best_this_restart = val_loss < best_loss_this_restart
                        
                        if is_best_this_restart:
                            best_loss_this_restart = val_loss
                            
                        if is_best:
                            best_val_loss = val_loss
                            model_path = os.path.join(models_dir, f'double_bottom_model_session_{session_id}_epoch_{epoch}.pth')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': val_loss,
                                'accuracy': accuracy
                            }, model_path)
                            logger.info(f"Saved best model checkpoint to {model_path}")
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        
                        # Early stopping with minimum epochs guarantee
                        if no_improve_count >= patience and epochs_this_restart >= min_epochs_per_restart:
                            if restart_count < max_restarts:
                                restart_count += 1
                                logger.info(f"Restarting training (attempt {restart_count}/{max_restarts})")
                                
                                # More aggressive learning rate increase
                                new_lr = learning_rate * (2.0 ** restart_count)
                                optimizer = torch.optim.AdamW(
                                    model.parameters(),
                                    lr=new_lr,
                                    weight_decay=0.0001 * (0.9 ** restart_count)  # Gradually reduce regularization
                                )
                                
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                    optimizer,
                                    T_0=10 * (2 ** restart_count),  # Longer cycles each restart
                                    T_mult=2,
                                    eta_min=1e-6
                                )
                                
                                no_improve_count = 0
                                break
                            else:
                                logger.info("Maximum restarts reached. Ending training.")
                                break
                        
                        # Update database
                        db_manager.update_training_progress(
                            session_id=session_id,
                            epoch=epoch + 1,
                            loss=val_loss.item(),
                            accuracy=accuracy.item(),
                            is_best=is_best
                        )
                        
                        logger.info(f'Epoch {epoch+1}/{epochs}:')
                        logger.info(f'  Training Loss: {total_loss/batches:.4f}')
                        logger.info(f'  Validation Loss: {val_loss:.4f}')
                        logger.info(f'  Validation Accuracy: {accuracy:.4f}')
                        logger.info(f'  Learning Rate: {current_lr:.6f}')
                    
                    # Prevent GPU overheating
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                if restart_count < max_restarts:
                    restart_count += 1
                    continue
                else:
                    raise
        
        final_metrics = {
            'final_loss': float(val_losses[-1]),
            'best_loss': float(best_val_loss),
            'total_epochs_completed': len(val_losses),
            'training_time_hours': (time.time() - start_time) / 3600,
            'restarts': restart_count
        }
        db_manager.complete_training_session(session_id, final_metrics)
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        db_manager.complete_training_session(session_id, {'error': str(e)})
        raise
    
    return val_losses

def main():
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Print system information
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
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
        val_losses = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            db_manager=db_manager,
            epochs=1000,  # High number of epochs, will stop based on time/early stopping
            batch_size=32,  # Back to original batch size
            learning_rate=0.0005,  # Slightly higher learning rate
            max_time_hours=8.0
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 