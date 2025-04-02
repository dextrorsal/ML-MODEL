from datetime import datetime, timedelta, timezone
import asyncio
import pandas as pd
from binance.spot import Spot
from binance.websocket.spot.websocket_client import SpotWebsocketClient
from typing import Dict, Optional
import logging
import json
import asyncpg
from queue import Queue
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOLDataCollector:
    def __init__(self, neon_connection_string: Optional[str] = None):
        self.client = Spot()
        self.ws_client = SpotWebsocketClient()
        self.symbol = "SOLUSDT"
        self.timeframes = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        # Initialize database connection
        self.neon_conn_str = neon_connection_string
        self.data_queue = Queue()
        self.is_running = False
        self._db_pool = None
        
        # In-memory cache for latest data
        self.cache = {
            'last_price': None,
            'last_update': None,
            'candles': {}
        }
    
    async def init_db(self):
        """Initialize database connection pool"""
        if not self._db_pool and self.neon_conn_str:
            self._db_pool = await asyncpg.create_pool(self.neon_conn_str)
    
    def message_handler(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'e' in data:  # Kline/Candlestick event
                if data['e'] == 'kline':
                    k = data['k']
                    candle_data = {
                        'symbol': self.symbol,
                        'timestamp': datetime.fromtimestamp(k['t'] / 1000),
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v'])
                    }
                    
                    # Update cache
                    self.cache['last_price'] = float(k['c'])
                    self.cache['last_update'] = datetime.now()
                    
                    # Add to queue for database insertion
                    self.data_queue.put(candle_data)
                    
                    logger.debug(f"Received candle: {candle_data}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def db_worker(self):
        """Background worker to handle database insertions"""
        await self.init_db()
        
        while self.is_running:
            try:
                if self._db_pool and not self.data_queue.empty():
                    async with self._db_pool.acquire() as conn:
                        batch_data = []
                        while not self.data_queue.empty() and len(batch_data) < 100:
                            data = self.data_queue.get()
                            batch_data.append((
                                data['timestamp'],
                                data['symbol'],
                                data['open'],
                                data['high'],
                                data['low'],
                                data['close'],
                                data['volume']
                            ))
                        
                        if batch_data:
                            await conn.executemany("""
                                INSERT INTO price_data (timestamp, symbol, open, high, low, close, volume)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                ON CONFLICT (timestamp, symbol) DO UPDATE
                                SET open = EXCLUDED.open,
                                    high = EXCLUDED.high,
                                    low = EXCLUDED.low,
                                    close = EXCLUDED.close,
                                    volume = EXCLUDED.volume
                            """, batch_data)
                
                await asyncio.sleep(1)  # Prevent CPU overload
            
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        self.is_running = True
        
        try:
            self.ws_client.start()
            
            # Start database worker task
            asyncio.create_task(self.db_worker())
            
            # Subscribe to kline/candlestick streams for different timeframes
            for tf in self.timeframes.values():
                stream_name = f"{self.symbol.lower()}@kline_{tf}"
                self.ws_client.kline(
                    symbol=self.symbol,
                    interval=tf,
                    callback=self.message_handler
                )
            
            logger.info(f"WebSocket started for {self.symbol}")
            
            while self.is_running:
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_running = False
            raise
    
    async def stop_websocket(self):
        """Stop WebSocket connection"""
        self.is_running = False
        self.ws_client.stop()
        logger.info("WebSocket stopped")
    
    def get_cached_price(self) -> Optional[float]:
        """Get latest cached price"""
        return self.cache['last_price']
    
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
    # Get Neon connection string (you'll need to provide this)
    neon_conn_str = "YOUR_NEON_CONNECTION_STRING"
    
    # Initialize collector with database connection
    collector = SOLDataCollector(neon_conn_str)
    
    try:
        # Start WebSocket for real-time data
        await collector.start_websocket()
    except KeyboardInterrupt:
        await collector.stop_websocket()
        logger.info("Gracefully shut down")

if __name__ == "__main__":
    asyncio.run(main()) 