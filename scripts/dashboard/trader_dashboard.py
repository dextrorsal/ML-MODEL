import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import threading
import time

# Add the root directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'solana-trading-dashboard-key'

# Create data directory if it doesn't exist
DASHBOARD_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)

# Signal data file
SIGNALS_FILE = os.path.join(DASHBOARD_DATA_DIR, 'signals.json')
PRICES_FILE = os.path.join(DASHBOARD_DATA_DIR, 'prices.json')

# Initialize data files if they don't exist
if not os.path.exists(SIGNALS_FILE):
    with open(SIGNALS_FILE, 'w') as f:
        json.dump([], f)
        
if not os.path.exists(PRICES_FILE):
    with open(PRICES_FILE, 'w') as f:
        json.dump([], f)

# Data storage
signals_data = []
prices_data = []

def load_data():
    """Load data from JSON files"""
    global signals_data, prices_data
    
    try:
        with open(SIGNALS_FILE, 'r') as f:
            signals_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading signals data: {e}")
        
    try:
        with open(PRICES_FILE, 'r') as f:
            prices_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading prices data: {e}")
        
def save_signals_data():
    """Save signals data to JSON file"""
    try:
        with open(SIGNALS_FILE, 'w') as f:
            json.dump(signals_data, f)
    except Exception as e:
        logger.error(f"Error saving signals data: {e}")
        
def save_prices_data():
    """Save prices data to JSON file"""
    try:
        with open(PRICES_FILE, 'w') as f:
            json.dump(prices_data, f)
    except Exception as e:
        logger.error(f"Error saving prices data: {e}")

# Watch for new log files
def scan_log_files():
    """Scan log files for new signals and prices"""
    global signals_data, prices_data
    
    while True:
        try:
            # Get the most recent log file
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            log_files = [f for f in os.listdir(log_dir) if f.startswith('combined_trader_')]
            
            if not log_files:
                logger.info("No log files found")
                time.sleep(10)
                continue
                
            # Sort by most recent
            log_files.sort(reverse=True)
            latest_log = os.path.join(log_dir, log_files[0])
            
            # Read the latest log file
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Process log lines
            new_signals = []
            new_prices = []
            current_signal = None
            
            for line in lines:
                # Check for signal
                if "NEW TRADING SIGNAL DETECTED" in line:
                    current_signal = {
                        "timestamp": line.split(" - ")[0],
                        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    
                # Get signal details
                if current_signal and "Timestamp:" in line:
                    current_signal["signal_time"] = line.split("Timestamp: ")[1].strip()
                    
                if current_signal and "Signal type:" in line:
                    current_signal["type"] = line.split("Signal type: ")[1].strip()
                    
                if current_signal and "5m Confidence:" in line:
                    current_signal["confidence_5m"] = float(line.split("5m Confidence: ")[1].strip())
                    
                if current_signal and "15m Confidence:" in line:
                    current_signal["confidence_15m"] = float(line.split("15m Confidence: ")[1].strip())
                    
                if current_signal and "Weighted Confidence:" in line:
                    current_signal["weighted_confidence"] = float(line.split("Weighted Confidence: ")[1].strip())
                    
                if current_signal and "Current SOL Price:" in line:
                    price_part = line.split("Current SOL Price: $")[1].strip()
                    current_signal["price"] = float(price_part)
                    
                    # Add to signals
                    signal_id = f"{current_signal['signal_time']}_{current_signal['type']}"
                    
                    # Check if we already have this signal
                    existing_ids = [s.get('id') for s in signals_data]
                    
                    if signal_id not in existing_ids:
                        current_signal["id"] = signal_id
                        new_signals.append(current_signal)
                        
                    current_signal = None
                
                # Check for price updates
                if "Current price:" in line:
                    try:
                        timestamp = line.split(" - ")[0]
                        price_str = line.split("Current price: $")[1].strip()
                        price = float(price_str)
                        
                        new_prices.append({
                            "timestamp": timestamp,
                            "price": price
                        })
                    except Exception as e:
                        logger.error(f"Error parsing price: {e}")
            
            # Add new signals
            if new_signals:
                signals_data.extend(new_signals)
                save_signals_data()
                logger.info(f"Added {len(new_signals)} new signals")
                
            # Update prices
            if new_prices:
                # Keep only the last 500 prices to avoid excessive data
                prices_data.extend(new_prices)
                prices_data = prices_data[-500:]
                save_prices_data()
                logger.info(f"Updated with {len(new_prices)} new price points")
                
        except Exception as e:
            logger.error(f"Error in log scanning: {e}")
            
        # Sleep for a bit before checking again
        time.sleep(10)

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/signals')
def get_signals():
    """API endpoint to get signals data"""
    return jsonify(signals_data)

@app.route('/api/prices')
def get_prices():
    """API endpoint to get price data"""
    return jsonify(prices_data)

@app.route('/api/stats')
def get_stats():
    """API endpoint to get trading statistics"""
    if not signals_data:
        return jsonify({
            "signal_count": 0,
            "win_rate": 0,
            "avg_return": 0,
            "types": {
                "TradingView-style": 0,
                "Combined": 0,
                "Strong": 0
            }
        })
    
    # Count signals by type
    signal_types = {}
    for signal in signals_data:
        signal_type = signal.get('type', 'Unknown')
        if signal_type in signal_types:
            signal_types[signal_type] += 1
        else:
            signal_types[signal_type] = 1
    
    # Calculate win rate and avg return if we have enough data
    # This is placeholder logic - in a real system we'd track actual trade outcomes
    win_rate = 0
    avg_return = 0
    
    return jsonify({
        "signal_count": len(signals_data),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "types": signal_types
    })

def start_log_scanner():
    """Start the log scanner thread"""
    scanner_thread = threading.Thread(target=scan_log_files)
    scanner_thread.daemon = True
    scanner_thread.start()

if __name__ == '__main__':
    # Load existing data
    load_data()
    
    # Start log scanner thread
    start_log_scanner()
    
    # Run the app
    logger.info("Starting dashboard at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False) 