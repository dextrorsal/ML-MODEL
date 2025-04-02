import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from src.data.collectors.sol_data_collector import SOLDataCollector
import asyncpg
import aiosqlite
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get connection string from environment variables or use a default test one
NEON_TEST_DB = os.environ.get("NEON_TEST_DB", "postgresql://dex:testpassword@localhost:5432/solana_trading_test")
SQLITE_TEST_DB = os.path.join(os.path.dirname(__file__), "test_data.db")

# Determine which database to use - we'll prioritize PostgreSQL
USE_SQLITE = False

@pytest.fixture(scope="module")
async def db_pool():
    """Create a database connection pool for testing"""
    try:
        pool = await asyncpg.create_pool(NEON_TEST_DB)
        
        # Setup test database tables
        async with pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open NUMERIC(16, 8) NOT NULL,
                    high NUMERIC(16, 8) NOT NULL,
                    low NUMERIC(16, 8) NOT NULL,
                    close NUMERIC(16, 8) NOT NULL,
                    volume NUMERIC(24, 8) NOT NULL,
                    PRIMARY KEY (timestamp, symbol)
                )
            ''')
        
        yield pool
        
        # Clean up after tests
        async with pool.acquire() as conn:
            await conn.execute('TRUNCATE TABLE price_data')
        await pool.close()
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        pytest.skip("PostgreSQL database connection failed")

@pytest.fixture
async def data_collector(db_pool):
    """Initialize data collector with test database"""
    collector = SOLDataCollector(NEON_TEST_DB)
    collector._db_pool = db_pool  # Use the existing pool
    
    yield collector
    
    # Cleanup
    try:
        await collector.stop_websocket()
    except:
        pass

@pytest.mark.asyncio
async def test_binance_connection(data_collector):
    """Test basic Binance API connectivity"""
    try:
        # Try fetching a single candle to confirm API connection
        data = await data_collector.fetch_all_timeframes(lookback_days=1)
        assert len(data) > 0, "No data received from Binance"
        
        # Check if we got data for all timeframes
        for tf in data_collector.timeframes:
            assert tf in data, f"Missing data for timeframe {tf}"
            
            df = data[tf]
            assert not df.empty, f"Empty dataframe for timeframe {tf}"
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
                f"Missing required columns in {tf} dataframe"
            
            # Verify data types
            assert pd.api.types.is_float_dtype(df['close']), "Close prices should be float"
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp should be datetime"
            
            # Check data integrity
            assert (df['high'] >= df['low']).all(), "High price must be >= low price"
            assert (df['volume'] >= 0).all(), "Volume must be non-negative"
    
    except Exception as e:
        pytest.fail(f"Binance API connection failed: {str(e)}")

@pytest.mark.asyncio
async def test_websocket_streaming(data_collector):
    """Test WebSocket data streaming"""
    received_data = []
    
    def test_message_handler(message):
        received_data.append(message)
    
    # Override message handler for testing
    original_handler = data_collector.message_handler
    data_collector.message_handler = test_message_handler
    
    try:
        # Start WebSocket
        websocket_task = asyncio.create_task(data_collector.start_websocket())
        
        # Wait for some data (with timeout)
        timeout = 15  # seconds
        start_time = datetime.now()
        
        while len(received_data) == 0:
            if (datetime.now() - start_time).total_seconds() > timeout:
                break
            await asyncio.sleep(1)
            
        # Restore original handler
        data_collector.message_handler = original_handler
        data_collector.is_running = False
            
        assert len(received_data) > 0, "No data received from WebSocket within timeout"
        
        # Verify data structure of at least one message
        for message in received_data:
            if '"e":"kline"' in message:  # Check for kline event
                assert '"s":"SOLUSDT"' in message, "Symbol not found in message"
                assert '"i":"' in message, "Interval not found in message"
                assert '"o":"' in message, "Open price not found in message"
                assert '"c":"' in message, "Close price not found in message"
                return  # Found at least one valid kline message
                
        pytest.fail("No valid kline messages found in received data")
    
    finally:
        # Ensure WebSocket is stopped
        data_collector.is_running = False
        try:
            await websocket_task
        except:
            pass

@pytest.mark.asyncio
async def test_database_storage(data_collector, db_pool):
    """Test data storage in database"""
    # Generate some test data
    test_time = datetime.now()
    test_data = {
        'symbol': data_collector.symbol,
        'timestamp': test_time,
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 103.0,
        'volume': 1000.0
    }
    
    # Add to queue and process
    data_collector.data_queue.put(test_data)
    data_collector.is_running = True
    
    # Start db_worker task
    worker_task = asyncio.create_task(data_collector.db_worker())
    
    # Wait a moment for processing
    await asyncio.sleep(2)
    
    # Stop the worker
    data_collector.is_running = False
    
    try:
        # Check if data was stored correctly
        async with db_pool.acquire() as conn:
            query = '''
                SELECT * FROM price_data 
                WHERE symbol = $1 AND 
                timestamp BETWEEN $2 AND $3
            '''
            timestamp_start = test_time - timedelta(seconds=1)
            timestamp_end = test_time + timedelta(seconds=1)
            
            row = await conn.fetchrow(query, data_collector.symbol, timestamp_start, timestamp_end)
            
            assert row is not None, "Test data was not stored in database"
            assert float(row['open']) == test_data['open'], "Open price didn't match"
            assert float(row['high']) == test_data['high'], "High price didn't match"
            assert float(row['low']) == test_data['low'], "Low price didn't match"
            assert float(row['close']) == test_data['close'], "Close price didn't match"
            assert float(row['volume']) == test_data['volume'], "Volume didn't match"
            
            # Clean up test data
            await conn.execute(
                "DELETE FROM price_data WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3",
                data_collector.symbol, timestamp_start, timestamp_end
            )
    
    finally:
        # Clean up
        try:
            await worker_task
        except:
            pass

@pytest.mark.asyncio
async def test_timeframe_consistency(data_collector):
    """Test consistency across different timeframes"""
    # Fetch data for the last week
    data = await data_collector.fetch_all_timeframes(lookback_days=7)
    
    # Ensure we have data for at least the 1h and 1d timeframes
    assert '1h' in data, "Missing 1h timeframe data"
    assert '1d' in data, "Missing 1d timeframe data"
    
    # Check that higher timeframes have fewer candles
    assert len(data['1h']) > len(data['4h']), "4h should have fewer candles than 1h"
    assert len(data['4h']) > len(data['1d']), "1d should have fewer candles than 4h"
    
    # Verify price range consistency
    assert abs(data['1h']['high'].max() - data['1d']['high'].max()) / data['1d']['high'].max() < 0.05, \
        "Large discrepancy in max price across timeframes"
    
    assert abs(data['1h']['low'].min() - data['1d']['low'].min()) / data['1d']['low'].min() < 0.05, \
        "Large discrepancy in min price across timeframes"
    
    # Check timestamp alignment
    for tf in data:
        df = data[tf]
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), \
            f"Timestamps not in datetime format in {tf}"
        assert df['timestamp'].is_monotonic_increasing, \
            f"Timestamps not properly ordered in {tf}"

@pytest.mark.asyncio
async def test_fetch_large_dataset(data_collector):
    """Test fetching a large historical dataset"""
    # Try to fetch 30 days of 5-minute data (should be a lot of candles)
    data = await data_collector.fetch_all_timeframes(lookback_days=30)
    
    assert '5m' in data, "Missing 5m timeframe data"
    
    # 5-minute data should have approximately 12 * 24 * 30 = 8640 candles
    # Allow for some missing candles
    assert len(data['5m']) > 5000, "Not enough 5m candles fetched"
    
    # Calculate returns and check for statistical properties
    data['5m']['returns'] = data['5m']['close'].pct_change()
    
    # Remove NaN and outliers 
    returns = data['5m']['returns'].dropna()
    
    # Basic statistical checks
    assert abs(returns.mean()) < 0.01, "Unexpected mean of returns"
    assert returns.std() > 0, "Standard deviation should be positive"

@pytest.mark.asyncio
async def test_cache_functionality(data_collector):
    """Test the in-memory caching functionality"""
    # Update cache with a test value
    test_price = 123.45
    test_time = datetime.now()
    
    data_collector.cache['last_price'] = test_price
    data_collector.cache['last_update'] = test_time
    
    # Verify cached price retrieval
    assert data_collector.get_cached_price() == test_price, "Cached price doesn't match"
    
    # Test WebSocket updating the cache
    received_data = False
    original_handler = data_collector.message_handler
    
    def test_cache_update_handler(message):
        nonlocal received_data
        original_handler(message)
        received_data = True
    
    data_collector.message_handler = test_cache_update_handler
    
    try:
        # Start WebSocket
        websocket_task = asyncio.create_task(data_collector.start_websocket())
        
        # Wait for some data (with timeout)
        timeout = 15  # seconds
        start_time = datetime.now()
        
        while not received_data:
            if (datetime.now() - start_time).total_seconds() > timeout:
                break
            await asyncio.sleep(1)
            
        if received_data:
            # Cache should be updated with a new price
            assert data_collector.cache['last_update'] > test_time, "Cache was not updated"
            assert data_collector.cache['last_price'] is not None, "No cached price available"
    
    finally:
        # Ensure WebSocket is stopped
        data_collector.is_running = False
        data_collector.message_handler = original_handler
        try:
            await websocket_task
        except:
            pass 