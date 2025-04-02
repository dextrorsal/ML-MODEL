import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.testing.indicator_validation import test_indicators
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv

def get_binance_test_data():
    """Get test data from Binance for different market conditions."""
    load_dotenv()
    
    # Initialize Binance client
    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET')
    )
    
    # Get SOL/USDT data for testing (1 year of hourly data)
    klines = client.get_historical_klines(
        "SOLUSDT",
        Client.KLINE_INTERVAL_1HOUR,
        str(int((datetime.now() - timedelta(days=365)).timestamp() * 1000))
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    
    # Clean up data
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Save to CSV for testing
    test_data_path = "test_data.csv"
    df.to_csv(test_data_path)
    return test_data_path

if __name__ == "__main__":
    print("Getting Binance test data...")
    data_path = get_binance_test_data()
    
    print("\nRunning indicator validation tests...")
    print("This will check for:")
    print("1. Look-ahead bias")
    print("2. Overfitting to noise")
    print("3. Signal distribution issues")
    print("\nResults will include plots for visual inspection.")
    
    test_indicators(data_path) 