import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from strategies.LorentzianStrategy.models.primary.lorentzian_classifier import LorentzianClassifier
from strategies.LorentzianStrategy.indicators.rsi import RSIIndicator
from strategies.LorentzianStrategy.indicators.wave_trend import WaveTrendIndicator
from strategies.LorentzianStrategy.indicators.cci import CCIIndicator
from strategies.LorentzianStrategy.indicators.adx import ADXIndicator

print("Testing Lorentzian Classifier...")

# Add strategy path to system path
strategy_path = Path(__file__).parent / 'strategies'
sys.path.append(str(strategy_path))

# Set up GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
print("Loading data...")
data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
df = pd.read_feather(data_path)
print(f"Loaded {len(df)} bars")

# Initialize indicators
print("\nInitializing indicators...")
rsi_14 = RSIIndicator(period=14, device=device)
rsi_9 = RSIIndicator(period=9, device=device)
wavetrend = WaveTrendIndicator(channel_length=10, average_length=11, device=device)
cci = CCIIndicator(period=20, device=device)
adx = ADXIndicator(period=20, device=device)

# Convert data to tensors and move to GPU
print("\nPreparing data...")
close = torch.tensor(df['close'].values, dtype=torch.float32).to(device)
high = torch.tensor(df['high'].values, dtype=torch.float32).to(device)
low = torch.tensor(df['low'].values, dtype=torch.float32).to(device)

# Calculate indicators
print("\nCalculating indicators...")
try:
    # Calculate indicators
    rsi_14_values = rsi_14.forward(close)
    print("RSI-14 calculated successfully")
    
    rsi_9_values = rsi_9.forward(close)
    print("RSI-9 calculated successfully")
    
    wt_values = wavetrend.forward(high, low, close)
    print("WaveTrend calculated successfully")
    
    cci_values = cci.forward(high, low, close)
    print("CCI calculated successfully")
    
    adx_values = adx.forward(high, low, close)
    print("ADX calculated successfully")
    
    # Convert indicators to numpy for the classifier
    features = np.column_stack([
        rsi_14_values['rsi'].cpu().numpy(),
        rsi_9_values['rsi'].cpu().numpy(),
        wt_values['wt1'].cpu().numpy(),
        wt_values['wt2'].cpu().numpy(),
        cci_values['cci'].cpu().numpy(),
        adx_values['adx'].cpu().numpy(),
    ])
    
    # Ensure no NaN values
    features = features[~np.isnan(features).any(axis=1)]
    
    print(f"\nFeature shape: {features.shape}")
    
    # Initialize and run Lorentzian Classifier
    print("\nInitializing Lorentzian Classifier...")
    input_size = features.shape[1]
    
    # Create model with explicit GPU settings
    model = LorentzianClassifier(
        input_size=input_size, 
        dropout_rate=0.2,
        sigma=1.0
    )
    
    # Explicitly move the model and all its parameters to GPU
    print("Moving model to GPU...")
    model = model.to(device)
    
    # Double-check that model parameters are on GPU
    device_check = next(model.parameters()).device
    print(f"Model parameters are on: {device_check}")
    
    # Convert features to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Get predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        direction_logits, confidence = model(features_tensor)
        
        # Convert to probabilities
        direction_probs = torch.nn.functional.softmax(direction_logits, dim=1)
        
        # Get predicted class (0=long, 1=short, 2=neutral)
        predictions = torch.argmax(direction_probs, dim=1)
        
        # Count predictions by class
        long_count = (predictions == 0).sum().item()
        short_count = (predictions == 1).sum().item()
        neutral_count = (predictions == 2).sum().item()
        
        print(f"\nPrediction counts:")
        print(f"Long: {long_count} ({long_count/len(predictions)*100:.2f}%)")
        print(f"Short: {short_count} ({short_count/len(predictions)*100:.2f}%)")
        print(f"Neutral: {neutral_count} ({neutral_count/len(predictions)*100:.2f}%)")
        
        # Average confidence
        avg_confidence = confidence.mean().item()
        print(f"\nAverage confidence: {avg_confidence:.4f}")
    
    print("\nLorentzian Classifier test completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())