import sys
import pandas as pd
import numpy as np
import torch
import talib
from pathlib import Path

# Add strategy path to system path
strategy_path = Path(__file__).parent / 'strategies'
sys.path.append(str(strategy_path))

def prepare_indicators(df):
    """Add basic indicators needed for testing"""
    df = df.copy()
    df['rsi'] = talib.RSI(df['close'].values)
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=8)
    df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=21)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
    df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
    df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
    
    # Add momentum and volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['momentum'] = df['close'].pct_change(periods=10)
    
    # Add price ratios
    df['price_to_ema_fast'] = df['close'] / df['ema_fast']
    df['price_to_ema_slow'] = df['close'] / df['ema_slow']
    df['ema_ratio'] = df['ema_fast'] / df['ema_slow']
    
    # Add volume features
    df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def prepare_features(df):
    """Prepare features for the model"""
    feature_columns = [
        'rsi', 'atr', 'ema_fast', 'ema_slow', 'macd', 'macd_signal',
        'macd_hist', 'adx', 'cci', 'returns', 'volatility', 'momentum',
        'price_to_ema_fast', 'price_to_ema_slow', 'ema_ratio',
        'volume_ratio', 'close', 'volume'
    ]
    
    # Normalize features
    features = df[feature_columns].values
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    return torch.tensor(features, dtype=torch.float32)

def test_signals():
    try:
        # Load the data
        data_path = Path('/home/dex/user_data/data/bitget/futures/SOL_USDT_USDT-5m-futures.feather')
        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            return
            
        df = pd.read_feather(data_path)
        print(f"\nLoaded data shape: {df.shape}")
        
        # Prepare indicators
        df = prepare_indicators(df)
        print("\nIndicators added successfully")
        
        # Get last 100 rows for testing
        test_df = df.tail(100).copy()
        
        # Test Lorentzian Classifier
        try:
            from LorentzianStrategy.models.primary.lorentzian_classifier import LorentzianClassifier
            
            # Initialize classifier with default parameters
            classifier = LorentzianClassifier(
                input_size=18,  # Number of features
                hidden_size=64,  # Size of hidden layers
                dropout_rate=0.2,  # Dropout rate for regularization
                sigma=1.0  # Sigma parameter for Lorentzian distance
            )
            
            # Prepare features
            features = prepare_features(test_df)
            
            # Get predictions
            with torch.no_grad():
                direction_logits, confidence = classifier(features)
                direction_probs = torch.softmax(direction_logits, dim=1)
            
            print("\nLorentzian Classifier Results (last 5 rows):")
            print("Direction Probabilities:")
            print("  Long:", direction_probs[-5:, 0].tolist())
            print("  Short:", direction_probs[-5:, 1].tolist())
            print("  Neutral:", direction_probs[-5:, 2].tolist())
            print("Confidence:", confidence[-5:].tolist())
            
        except Exception as e:
            print(f"Error testing Lorentzian Classifier: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Test Logistic Model
        try:
            from LorentzianStrategy.models.confirmation.logistic_model import LogisticModel
            logistic = LogisticModel()
            
            # Calculate signals
            log_signals = logistic.calculate(test_df)
            
            print("\nLogistic Model Results (last 5 rows):")
            for key, value in log_signals.items():
                print(f"{key}:", value.tail().tolist())
            
        except Exception as e:
            print(f"Error testing Logistic Model: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"General error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_signals()