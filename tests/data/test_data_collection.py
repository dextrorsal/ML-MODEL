import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from src.data.collectors.sol_data_collector import SOLDataCollector

@pytest.fixture
async def collector():
    """Initialize data collector with test database"""
    # Use a test database connection string
    test_conn_str = "postgresql://[YOUR_TEST_DB]/test_db"
    collector = SOLDataCollector(test_conn_str)
    yield collector
    # Cleanup
    await collector.stop_websocket()

@pytest.mark.asyncio
async def test_binance_connection(collector):
    """Test basic Binance API connectivity"""
    try:
        # Try fetching a single candle
        data = await collector.fetch_all_timeframes(lookback_days=1)
        assert len(data) > 0, "No data received from Binance"
        
        # Check if we got data for all timeframes
        for tf in collector.timeframes:
            assert tf in data, f"Missing data for timeframe {tf}"
            
            df = data[tf]
            assert not df.empty, f"Empty dataframe for timeframe {tf}"
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
                f"Missing required columns in {tf} dataframe"
            
            # Verify data types
            assert pd.api.types.is_float_dtype(df['close']), "Close prices should be float"
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp should be datetime"
    
    except Exception as e:
        pytest.fail(f"Binance API connection failed: {str(e)}")

@pytest.mark.asyncio
async def test_websocket_streaming(collector):
    """Test WebSocket data streaming"""
    received_data = []
    
    def test_message_handler(message):
        received_data.append(message)
    
    # Override message handler for testing
    collector.message_handler = test_message_handler
    
    # Start WebSocket
    await collector.start_websocket()
    
    # Wait for some data
    await asyncio.sleep(10)
    
    assert len(received_data) > 0, "No data received from WebSocket"
    
    # Verify data structure
    first_message = received_data[0]
    assert 'e' in first_message, "Missing event type in message"
    assert 'k' in first_message, "Missing kline data in message"

@pytest.mark.asyncio
async def test_database_storage(collector):
    """Test data storage in Neon database"""
    # Start WebSocket and collect some data
    await collector.start_websocket()
    await asyncio.sleep(10)
    
    # Query the database
    async with collector._db_pool.acquire() as conn:
        # Check if data was stored
        count = await conn.fetchval('SELECT COUNT(*) FROM price_data WHERE symbol = $1', collector.symbol)
        assert count > 0, "No data stored in database"
        
        # Verify data integrity
        row = await conn.fetchrow('''
            SELECT * FROM price_data 
            WHERE symbol = $1 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', collector.symbol)
        
        assert row is not None, "Failed to retrieve stored data"
        assert row['open'] > 0, "Invalid open price"
        assert row['high'] >= row['low'], "High price should be >= low price"
        assert row['volume'] >= 0, "Volume should be non-negative"

@pytest.mark.asyncio
async def test_timeframe_consistency(collector):
    """Test consistency across different timeframes"""
    data = await collector.fetch_all_timeframes(lookback_days=7)
    
    # Compare volume across timeframes
    daily_volume = data['1d']['volume'].sum()
    hourly_volume = data['1h']['volume'].sum()
    
    # Allow for small differences due to rounding
    assert abs(daily_volume - hourly_volume) / daily_volume < 0.01, \
        "Large discrepancy in volume across timeframes"
    
    # Check timestamp alignment
    for tf in data:
        df = data[tf]
        assert df['timestamp'].is_monotonic_increasing, \
            f"Timestamps not properly ordered in {tf} timeframe" 