from datetime import datetime, timedelta, timezone
import asyncio
import pandas as pd
from binance.spot import Spot
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOLDataCollector:
    def __init__(self):
        self.client = Spot()
        # For perpetual futures on Binance, we use the USDT-margined contract
        self.symbol = "SOLUSDT"  
        self.timeframes = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"  # Added daily timeframe
        }
    
    async def fetch_all_timeframes(self, lookback_days: int = 365) -> Dict[str, pd.DataFrame]:  # Increased default lookback for better pattern recognition
        """
        Fetch SOL-PERP data for all specified timeframes.
        
        Args:
            lookback_days: Number of days of historical data to fetch
            
        Returns:
            Dictionary of DataFrames for each timeframe
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        
        data = {}
        for tf_name, tf_interval in self.timeframes.items():
            logger.info(f"Fetching {tf_name} data for {self.symbol}...")
            
            try:
                klines = self.client.klines(
                    symbol=self.symbol,
                    interval=tf_interval,
                    startTime=int(start_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=1000
                )
                
                # Convert to DataFrame with proper column names
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Add timeframe info
                df['timeframe'] = tf_name
                
                data[tf_name] = df
                logger.info(f"Successfully fetched {len(df)} {tf_name} candles")
                
            except Exception as e:
                logger.error(f"Error fetching {tf_name} data: {e}")
                continue
            
            # Add a small delay between requests to avoid rate limits
            await asyncio.sleep(1)
            
        return data

async def main():
    # Initialize collector
    collector = SOLDataCollector()
    
    # Fetch data for all timeframes
    data = await collector.fetch_all_timeframes(lookback_days=30)
    
    # Save each timeframe to a separate CSV file
    for timeframe, df in data.items():
        filename = f'data/sol_perp_{timeframe}.csv'
        df.to_csv(filename, index=False)
        logger.info(f"Saved {filename} with {len(df)} rows")
        
        # Print first few rows as a preview
        print(f"\nPreview of {timeframe} data:")
        print(df.head(3))
        print("\nShape:", df.shape)

if __name__ == "__main__":
    asyncio.run(main()) 