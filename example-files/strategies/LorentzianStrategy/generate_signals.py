#!/usr/bin/env python3
"""
UTILITY COMPONENT: Real-Time Signal Generator

This script provides real-time trading signal generation capabilities for the 
Integrated ML Trading System. It loads pre-trained models and applies them to 
new market data to produce actionable trading signals.

Key features:
- Loads pre-trained Lorentzian ANN, Logistic Regression, and Chandelier Exit models
- Processes new market data from CSV or Feather files
- Calculates necessary technical indicators
- Generates combined trading signals with stop levels
- Outputs human-readable signal summaries and JSON format for automation

This utility is ideal for:
1. Real-time trading signal generation in production
2. Generating signals for recent market data
3. Testing model performance on new data
4. Integration with trading bots and execution systems

Command-line usage:
```
python generate_signals.py --data path/to/ohlcv.feather --models models/ --output signals.json
```

Arguments:
  --data       Path to OHLCV data file (CSV or Feather format)
  --models     Directory containing pre-trained models
  --output     Output file for signals (JSON format)
  --verbose    Print detailed output
  --last-bars  Number of last bars to analyze (default: 100)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import sys
import time
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

# Import our components
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader
from strategies.LorentzianStrategy.config import default_config

# Import indicator preparation function
try:
    from analyze_lorentzian_ann import prepare_indicators
except ImportError:
    print("Warning: Could not import prepare_indicators from analyze_lorentzian_ann.py")
    
    def prepare_indicators(df):
        """Placeholder for indicator preparation function"""
        print("Using placeholder prepare_indicators function")
        print("Please implement proper indicator calculation")
        return df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate signals using the Integrated ML Trading System")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to the new data file (.feather or .csv format)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        help="Directory containing pre-trained models"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for signals (json format)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    parser.add_argument(
        "--last-bars",
        type=int,
        default=100,
        help="Number of last bars to analyze"
    )
    
    return parser.parse_args()

def load_data(data_path):
    """Load and preprocess data"""
    file_ext = data_path.suffix.lower()
    
    if file_ext == '.feather':
        df = pd.read_feather(data_path)
    elif file_ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Ensure datetime column exists and is set as index
    if 'datetime' not in df.columns and 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    return df

def generate_signals(args):
    """Generate trading signals for the input data"""
    start_time = time.time()
    
    # Load data
    print(f"Loading data from {args.data}")
    data_path = Path(args.data)
    df = load_data(data_path)
    
    # Use only last N bars if specified
    if args.last_bars and args.last_bars < len(df):
        if args.verbose:
            print(f"Using last {args.last_bars} bars out of {len(df)}")
        df = df.iloc[-args.last_bars:]
    
    # Prepare indicators
    print("Calculating technical indicators...")
    df = prepare_indicators(df)
    
    # Create trader and load models
    model_dir = Path(args.models)
    print(f"Loading models from {model_dir}")
    
    trader = IntegratedMLTrader(model_dir=model_dir)
    trader.load_models()
    
    # Generate signals
    print("Generating signals...")
    df = trader.generate_signals(df)
    
    # Prepare signal output
    latest_bar = df.iloc[-1]
    timestamp = latest_bar.name.isoformat() if hasattr(latest_bar.name, 'isoformat') else str(latest_bar.name)
    
    signal_output = {
        "timestamp": timestamp,
        "close_price": float(latest_bar['close']),
        "primary_signal": int(latest_bar['lorentzian_signal']),
        "confirmation_signal": int(latest_bar['logistic_signal']),
        "combined_signal": int(latest_bar['combined_signal']),
        "long_stop": float(latest_bar['long_stop']),
        "short_stop": float(latest_bar['short_stop']),
        "signals_history": {
            "primary": df['lorentzian_signal'].astype(int).tolist()[-5:],
            "confirmation": df['logistic_signal'].astype(int).tolist()[-5:],
            "combined": df['combined_signal'].astype(int).tolist()[-5:],
        },
        "generated_at": datetime.now().isoformat()
    }
    
    # Add position management info
    if 'position' in df.columns:
        current_position = df['position'].iloc[-1]
        signal_output["current_position"] = int(current_position)
    
    # Print results
    print("\n" + "="*50)
    print("SIGNAL SUMMARY")
    print("="*50)
    print(f"Timestamp: {signal_output['timestamp']}")
    print(f"Close Price: {signal_output['close_price']}")
    
    # Translate signal to trading action
    if signal_output['combined_signal'] == 1:
        action = "BUY/LONG"
    elif signal_output['combined_signal'] == -1:
        action = "SELL/SHORT"
    else:
        action = "NO ACTION"
    
    print(f"Signal: {action}")
    print(f"Primary Signal (Lorentzian): {signal_output['primary_signal']}")
    print(f"Confirmation Signal (Logistic): {signal_output['confirmation_signal']}")
    
    print(f"Long Stop: {signal_output['long_stop']}")
    print(f"Short Stop: {signal_output['short_stop']}")
    print("="*50)
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(signal_output, f, indent=2)
        
        print(f"Signals saved to {output_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Signal generation completed in {elapsed_time:.2f} seconds")
    
    return signal_output

if __name__ == "__main__":
    args = parse_args()
    signals = generate_signals(args) 