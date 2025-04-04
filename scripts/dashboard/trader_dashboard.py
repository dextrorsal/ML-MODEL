import os
import sys
import logging
import pandas as pd
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import threading
import time
import asyncio

# Add the root directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.db_connector import DBConnector

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'solana-trading-dashboard-key'

# Get database connection string from environment if available
NEON_CONNECTION_STRING = os.environ.get(
    "NEON_CONNECTION_STRING",
    "postgresql://neondb_owner:PgT1zO2ywrVU@ep-silent-dust-61256651.us-east-2.aws.neon.tech/neondb"
)

# Initialize database connector
db = DBConnector(NEON_CONNECTION_STRING)

# Data storage
signals_data = []
prices_data = []

async def load_data_from_db():
    """Load initial data from database"""
    global signals_data, prices_data
    
    try:
        # Load price data
        prices_df = await db.fetch_as_dataframe("""
            SELECT timestamp, close, symbol
            FROM price_data
            WHERE symbol = 'SOLUSDT'
            AND timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp DESC
            LIMIT 500
        """)
        
        prices_data = []
        
        for _, row in prices_df.iterrows():
            timestamp = row['timestamp']
            if hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp)
                
            prices_data.append({
                'timestamp': timestamp_str,
                'price': float(row['close'])
            })
        
        logger.info(f"Loaded {len(prices_data)} price points from database")
        
        # Load signals data
        signals_df = await db.fetch_as_dataframe("""
            SELECT 
                id, 
                timestamp, 
                signal_type, 
                signal_strength, 
                symbol,
                confidence_5m,
                confidence_15m,
                weighted_confidence,
                price
            FROM trading_signals
            WHERE symbol = 'SOLUSDT'
            AND timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        signals_data = []
        
        for _, row in signals_df.iterrows():
            # Default values for newer columns that might be NULL
            confidence_5m = float(row['confidence_5m']) if row['confidence_5m'] is not None else 0.0
            confidence_15m = float(row['confidence_15m']) if row['confidence_15m'] is not None else 0.0
            weighted_confidence = float(row['weighted_confidence']) if row['weighted_confidence'] is not None else 0.0
            price = float(row['price']) if row['price'] is not None else 0.0
            
            timestamp = row['timestamp']
            if hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp)
            
            signal_data = {
                'id': f"{timestamp_str}_{row['signal_type']}",
                'signal_time': timestamp_str,
                'type': row['signal_type'],
                'confidence_5m': confidence_5m,
                'confidence_15m': confidence_15m,
                'weighted_confidence': weighted_confidence,
                'price': price
            }
            
            signals_data.append(signal_data)
        
        logger.info(f"Loaded {len(signals_data)} signals from database")
        
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")

async def scan_log_files():
    """Scan log files for new signals and prices and save to database"""
    
    while True:
        try:
            # Get the most recent log file
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            log_files = [f for f in os.listdir(log_dir) if f.startswith('combined_trader_')]
            
            if not log_files:
                logger.info("No log files found")
                await asyncio.sleep(10)
                continue
                
            # Sort by most recent
            log_files.sort(reverse=True)
            latest_log = os.path.join(log_dir, log_files[0])
            
            # Read the latest log file
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Process log lines
            current_signal = None
            
            for line in lines:
                # Check for price updates
                if "Current price:" in line:
                    try:
                        timestamp_str = line.split(" - ")[0]
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                        price_str = line.split("Current price: $")[1].strip()
                        price = float(price_str)
                        
                        # Store in database
                        await db.execute("""
                            INSERT INTO price_data (timestamp, symbol, open, high, low, close, volume)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (timestamp, symbol) DO NOTHING
                        """, timestamp, 'SOLUSDT', price, price, price, price, 0)
                        
                    except Exception as e:
                        logger.error(f"Error parsing or storing price: {e}")
                
                # Check for signal
                if "NEW TRADING SIGNAL DETECTED" in line:
                    current_signal = {
                        "timestamp": datetime.now(),
                        "detected_at": datetime.now(),
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
                    
                    # Save to database
                    try:
                        # Convert signal_time to datetime if it's a string
                        if isinstance(current_signal["signal_time"], str):
                            try:
                                signal_time = datetime.fromisoformat(current_signal["signal_time"].replace('Z', '+00:00'))
                            except:
                                signal_time = datetime.now()
                        else:
                            signal_time = current_signal["signal_time"]
                            
                        # Create a signal object to pass to our db connector
                        signal = {
                            'timestamp': signal_time,
                            'symbol': 'SOLUSDT',
                            'signal_type': current_signal["type"],
                            'signal_strength': current_signal["weighted_confidence"],
                            'confidence_5m': current_signal["confidence_5m"],
                            'confidence_15m': current_signal["confidence_15m"],
                            'weighted_confidence': current_signal["weighted_confidence"],
                            'price': current_signal["price"]
                        }
                        
                        await db.insert_trading_signal(signal)
                        logger.info(f"Saved new signal to database: {current_signal['type']} at {signal_time}")
                            
                    except Exception as e:
                        logger.error(f"Error saving signal to database: {e}")
                    
                    current_signal = None
            
            # Reload from database periodically to keep our in-memory data fresh
            await load_data_from_db()
                
        except Exception as e:
            logger.error(f"Error in log scanning: {e}")
            
        # Sleep for a bit before checking again
        await asyncio.sleep(10)

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
async def get_stats():
    """API endpoint to get trading statistics"""
    try:
        # Query database for stats
        stats_result = await db.fetch("""
            WITH signal_stats AS (
                SELECT 
                    signal_type,
                    COUNT(*) as count
                FROM trading_signals
                WHERE symbol = 'SOLUSDT'
                AND timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY signal_type
            )
            SELECT 
                (SELECT COUNT(*) FROM trading_signals WHERE symbol = 'SOLUSDT') as total_count
            FROM signal_stats
        """)
        
        # Get signal types counts
        signal_types_result = await db.fetch("""
            SELECT 
                signal_type,
                COUNT(*) as count
            FROM trading_signals
            WHERE symbol = 'SOLUSDT'
            AND timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY signal_type
        """)
        
        if stats_result and len(stats_result) > 0:
            total_count = stats_result[0]['total_count']
            
            signal_types = {}
            for row in signal_types_result:
                signal_types[row['signal_type']] = row['count']
            
            # Just placeholder values until we track actual trade outcomes
            win_rate = 0.55  # 55% win rate placeholder
            avg_return = 0.32  # 0.32% average return placeholder
            
            return jsonify({
                "signal_count": total_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "types": signal_types
            })
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
    
    # Fallback to counting signals in memory
    signal_types = {}
    for signal in signals_data:
        signal_type = signal.get('type', 'Unknown')
        if signal_type in signal_types:
            signal_types[signal_type] += 1
        else:
            signal_types[signal_type] = 1
    
    return jsonify({
        "signal_count": len(signals_data),
        "win_rate": 0.5,  # placeholder
        "avg_return": 0.25,  # placeholder
        "types": signal_types
    })

async def start_async_tasks():
    """Start all async tasks"""
    # Load initial data
    await load_data_from_db()
    
    # Start log scanning
    asyncio.create_task(scan_log_files())

def start_background_tasks():
    """Start background tasks in a thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_async_tasks())
    loop.run_forever()

if __name__ == '__main__':
    # Start background tasks in a separate thread
    bg_thread = threading.Thread(target=start_background_tasks)
    bg_thread.daemon = True
    bg_thread.start()
    
    # Run the Flask app
    logger.info("Starting dashboard at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False) 