import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from strategies.LorentzianStrategy.models.primary.lorentzian_classifier import (
    LorentzianClassifier
)
from strategies.LorentzianStrategy.indicators.rsi import RSIIndicator
from strategies.LorentzianStrategy.indicators.wave_trend import (
    WaveTrendIndicator
)
from strategies.LorentzianStrategy.indicators.cci import CCIIndicator
from strategies.LorentzianStrategy.indicators.adx import ADXIndicator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split

# Set up GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Add strategy path to system path
strategy_path = Path(__file__).parent / 'strategies'
sys.path.append(str(strategy_path))

class TradingDataset(Dataset):
    """Custom Dataset for trading data"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TradingModel(nn.Module):
    """Enhanced Neural Network for trading signals"""
    def __init__(self, input_size):
        super(TradingModel, self).__init__()
        
        # Increase network capacity
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)
        
        # Add batch normalization
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        
        # Use LeakyReLU for better gradient flow
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)  # Increase dropout
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)
        
        x = self.layer5(x)
        return torch.sigmoid(x)

def prepare_indicators(df):
    """Add technical indicators using custom PyTorch implementations"""
    df = df.copy()
    print(f"\nInitial data shape: {df.shape}")
    
    # Initialize indicators with GPU
    rsi_14 = RSIIndicator(period=14, device=device)
    rsi_9 = RSIIndicator(period=9, device=device)
    wavetrend = WaveTrendIndicator(channel_length=10, average_length=11, device=device)
    cci = CCIIndicator(period=20, device=device)
    adx = ADXIndicator(period=20, device=device)

    # Convert data to tensors and move to GPU
    close = torch.tensor(df['close'].values, dtype=torch.float32).to(device)
    high = torch.tensor(df['high'].values, dtype=torch.float32).to(device)
    low = torch.tensor(df['low'].values, dtype=torch.float32).to(device)
    
    print("\nCalculating indicators...")
    
    try:
        # Calculate indicators
        rsi_14_values = rsi_14.forward(close)['rsi']
        print("RSI-14 calculated successfully")
        
        rsi_9_values = rsi_9.forward(close)['rsi']
        print("RSI-9 calculated successfully")
        
        wt_values = wavetrend.forward(high, low, close)
        print("WaveTrend calculated successfully")
        
        cci_values = cci.forward(high, low, close)['cci']
        print("CCI calculated successfully")
        
        adx_values = adx.forward(high, low, close)['adx']
        print("ADX calculated successfully")
        
        print("\nMoving indicators to CPU...")
        
        # Move tensors back to CPU
        df['rsi_14'] = rsi_14_values.cpu().numpy()
        df['rsi_9'] = rsi_9_values.cpu().numpy()
        df['wt1'] = wt_values['wt1'].cpu().numpy()
        df['wt2'] = wt_values['wt2'].cpu().numpy()
        df['cci'] = cci_values.cpu().numpy()
        df['adx'] = adx_values.cpu().numpy()
        
        print("\nCalculating additional features...")
        
        # Price action features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['regime'] = df['returns'].rolling(window=50).mean()
        
        # Momentum features
        df['rsi_diff'] = df['rsi_14'] - df['rsi_9']  # RSI divergence
        df['wt_diff'] = df['wt1'] - df['wt2']  # WaveTrend divergence
        
        # Trend features
        df['price_trend'] = df['close'].rolling(window=20).mean().pct_change()
        df['volume_trend'] = df['volume'].rolling(window=20).mean().pct_change()
        
        # Volatility features
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift()),
            'lc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Generate labels (1 for profitable trades, 0 for unprofitable)
        print("\nGenerating training labels...")
        df['future_returns'] = df['returns'].shift(-1)
        df['label'] = (df['future_returns'] > 0).astype(float)
        
        # Print label distribution
        total_samples = len(df)
        positive_samples = df['label'].sum()
        print(f"\nLabel Distribution:")
        print(f"Positive (Profitable) Samples: {positive_samples:,} ({positive_samples/total_samples:.2%})")
        print(f"Negative (Unprofitable) Samples: {total_samples-positive_samples:,} ({1-positive_samples/total_samples:.2%})")
        
        # Drop NaN values
        print("\nRemoving NaN values...")
        df_before = len(df)
        df = df.dropna()
        df_after = len(df)
        print(f"Rows before dropping NaN: {df_before:,}")
        print(f"Rows after dropping NaN: {df_after:,}")
        print(f"Dropped {df_before - df_after:,} rows")
        
    except Exception as e:
        print(f"Error in prepare_indicators: {str(e)}")
        raise
        
    return df

def apply_filters(df):
    """Apply filters to identify tradeable conditions"""
    print(f"\nApplying filters...")
    print(f"Initial filter data shape: {df.shape}")
    
    # Volatility filter - price movement should be significant
    volatility_mask = df['volatility'] > df['volatility'].quantile(0.1)
    print(f"Volatility filter passes: {volatility_mask.sum()}")
    
    # Regime filter - trend should be established
    regime_mask = df['regime'] > -0.2
    print(f"Regime filter passes: {regime_mask.sum()}")
    
    # ADX filter - trend should be strong enough
    adx_mask = df['adx'] > 10.0  # Lowered from 20.0
    print(f"ADX filter passes: {adx_mask.sum()}")
    
    # Combine all filters
    combined_mask = volatility_mask & regime_mask & adx_mask
    print(f"Combined filter passes: {combined_mask.sum()}")
    
    return df[combined_mask]

def prepare_features(df):
    """Prepare features for the model"""
    # Scale the features
    scaler = MinMaxScaler()
    
    features = np.column_stack([
        df['rsi_14'].values,
        df['rsi_9'].values,
        df['wt1'].values,
        df['wt2'].values,
        df['cci'].values,
        df['adx'].values
    ])
    
    # Scale features
    features = scaler.fit_transform(features)
    return features

def train_model(df, epochs=100, batch_size=64):
    """Train the trading model"""
    try:
        # Prepare features and labels
        feature_cols = [
            'rsi_14', 'rsi_9', 'wt1', 'wt2', 'cci', 'adx',
            'volatility', 'regime', 'rsi_diff', 'wt_diff',
            'price_trend', 'volume_trend', 'atr_ratio'
        ]
        
        # Scale features
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[feature_cols])
        labels = df['label'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Create datasets and dataloaders
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Initialize model
        model = TradingModel(len(feature_cols)).to(device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\nStarting model training...")
        print(f"Training on {len(X_train):,} samples")
        print(f"Validating on {len(X_val):,} samples")
        print(f"Using {len(feature_cols)} features: {', '.join(feature_cols)}\n")
        
        # Training loop
        best_val_accuracy = 0
        for epoch in range(epochs):
            model.train()
            train_losses = []
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(
                    outputs.squeeze(), batch_labels
                )
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(
                        outputs.squeeze(), batch_labels
                    )
                    
                    val_losses.append(loss.item())
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_accuracy = 100 * correct / total
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {avg_train_loss:.4f}')
                print(f'Val Loss: {avg_val_loss:.4f}')
                print(f'Val Accuracy: {val_accuracy:.2f}%\n')
            
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_cols': feature_cols,
                    'scaler': scaler
                }, 'best_model.pth')
                print(f'New best model saved! Accuracy: {val_accuracy:.2f}%\n')
        
        print(f'Training completed!')
        print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
        
        # Return model, scaler and feature columns
        return model, scaler, feature_cols
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def analyze_signals(df, model=None, scaler=None, feature_cols=None):
    """Analyze trading signals and calculate performance metrics"""
    try:
        # Prepare data
        df = prepare_indicators(df)
        
        if model is None or scaler is None or feature_cols is None:
            print("Training new model...")
            model, scaler, feature_cols = train_model(df)
            if model is None:
                raise ValueError("Model training failed")
        
        # Prepare features for prediction
        features = scaler.transform(df[feature_cols])
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(device)
            predictions = model(features_tensor)
            predictions = predictions.cpu().numpy()
        
        # Generate trading signals (more aggressive thresholds)
        df['signal'] = 0
        df.loc[predictions.squeeze() > 0.6, 'signal'] = 1  # Long threshold
        df.loc[predictions.squeeze() < 0.4, 'signal'] = -1  # Short threshold
        
        # Calculate returns
        df['position'] = df['signal'].shift(1)
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Calculate metrics
        total_trades = (df['position'].diff() != 0).sum()
        winning_trades = (df['strategy_returns'] > 0).sum()
        total_trades_with_returns = (df['strategy_returns'] != 0).sum()
        
        win_rate = winning_trades / total_trades_with_returns if total_trades_with_returns > 0 else 0
        avg_win = df.loc[df['strategy_returns'] > 0, 'strategy_returns'].mean()
        avg_loss = df.loc[df['strategy_returns'] < 0, 'strategy_returns'].mean()
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: {avg_win:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        plt.plot(df.index, df['cumulative_returns'])
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.grid(True)
        plt.savefig('cumulative_returns.png')
        plt.close()
        
        final_return = (df['cumulative_returns'].iloc[-1]-1)*100
        print(f"\nFinal Return: {final_return:.2f}%")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Load data
    data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
    df = pd.read_feather(data_path)
    
    # Run analysis
    analyze_signals(df)