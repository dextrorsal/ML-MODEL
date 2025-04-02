import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.features.technical.indicators.wave_trend import WaveTrendIndicator
from src.features.technical.indicators.rsi import RSIIndicator
from src.features.technical.indicators.adx import ADXIndicator
from src.features.technical.indicators.cci import CCIIndicator
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get connection string from environment variables or use a default test one
NEON_TEST_DB = os.environ.get("NEON_TEST_DB", "postgresql://postgres:test@localhost:5432/solana_trading_test")

@pytest.fixture
async def collector():
    """Initialize data collector"""
    collector = SOLDataCollector(NEON_TEST_DB)
    yield collector
    # Cleanup
    await collector.stop_websocket()

@pytest.fixture
async def sample_data(collector):
    """Fetch sample data from Binance for testing"""
    data = await collector.fetch_all_timeframes(lookback_days=3)
    return data

@pytest.mark.asyncio
async def test_data_indicator_compatibility(sample_data):
    """Test that collected data is compatible with technical indicators"""
    # Get 1h data for testing
    df = sample_data['1h']
    
    # Initialize indicators
    wt = WaveTrendIndicator()
    rsi = RSIIndicator()
    adx = ADXIndicator()
    cci = CCIIndicator()
    
    # Calculate signals for each indicator
    wt_signals = wt.calculate_signals(df)
    rsi_signals = rsi.calculate_signals(df)
    adx_signals = adx.calculate_signals(df)
    cci_signals = cci.calculate_signals(df)
    
    # Check that signals are generated with correct dimensions
    assert len(wt_signals['wt1']) == len(df), "WaveTrend signals length mismatch"
    assert len(rsi_signals['rsi']) == len(df), "RSI signals length mismatch"
    assert len(adx_signals['ADX']) == len(df), "ADX signals length mismatch"
    assert len(cci_signals['cci']) == len(df), "CCI signals length mismatch"
    
    # Check for NaN values (should be minimal after warmup period)
    # Allow for warmup period (first 20% of data)
    warmup_period = int(len(df) * 0.2)
    
    # Check WaveTrend
    assert wt_signals['wt1'][warmup_period:].isnan().sum() < len(df) * 0.05, "Too many NaN values in WaveTrend"
    
    # Check RSI
    assert rsi_signals['rsi'][warmup_period:].isnan().sum() < len(df) * 0.05, "Too many NaN values in RSI"
    
    # Check ADX
    assert adx_signals['ADX'][warmup_period:].isnan().sum() < len(df) * 0.05, "Too many NaN values in ADX"
    
    # Check CCI
    assert cci_signals['cci'][warmup_period:].isnan().sum() < len(df) * 0.05, "Too many NaN values in CCI"

@pytest.mark.asyncio
async def test_data_lorentzian_compatibility(sample_data):
    """Test that collected data works with the Lorentzian Classifier"""
    # Get 1h data for testing
    df = sample_data['1h']
    
    # Initialize classifier
    classifier = LorentzianClassifier()
    
    # Generate signals
    try:
        signals = classifier.calculate_signals(df)
        
        # Check signal dimensions
        assert len(signals['predictions']) == len(df), "Classifier predictions length mismatch"
        assert len(signals['buy_signals']) == len(df), "Classifier buy signals length mismatch"
        assert len(signals['sell_signals']) == len(df), "Classifier sell signals length mismatch"
        
        # Verify warmup period (allow for NaNs in first 20% of data)
        warmup_period = int(len(df) * 0.2)
        
        # Check for valid signal generation after warmup
        valid_signals = (signals['buy_signals'][warmup_period:] + signals['sell_signals'][warmup_period:]).sum()
        assert valid_signals > 0, "No trading signals generated"
        
    except Exception as e:
        pytest.fail(f"Lorentzian Classifier failed to process data: {str(e)}")

@pytest.mark.asyncio
async def test_timeframe_transformation(sample_data):
    """Test timeframe resampling and transformation capabilities"""
    # Get 5m data as source
    df_5m = sample_data['5m']
    
    # Resample to 15 minutes
    df_15m_resampled = df_5m.set_index('timestamp').resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Compare with actual 15m data
    df_15m_actual = sample_data['15m']
    
    # Get overlapping time range
    start_time = max(df_15m_resampled['timestamp'].min(), df_15m_actual['timestamp'].min())
    end_time = min(df_15m_resampled['timestamp'].max(), df_15m_actual['timestamp'].max())
    
    df_15m_resampled = df_15m_resampled[(df_15m_resampled['timestamp'] >= start_time) & 
                                      (df_15m_resampled['timestamp'] <= end_time)]
    
    df_15m_actual = df_15m_actual[(df_15m_actual['timestamp'] >= start_time) & 
                                 (df_15m_actual['timestamp'] <= end_time)]
    
    # Allow for small differences due to slight differences in candle boundaries
    close_diff = abs(df_15m_resampled['close'].mean() - df_15m_actual['close'].mean()) / df_15m_actual['close'].mean()
    volume_diff = abs(df_15m_resampled['volume'].sum() - df_15m_actual['volume'].sum()) / df_15m_actual['volume'].sum()
    
    assert close_diff < 0.05, "Large discrepancy in closing prices between resampled and actual data"
    assert volume_diff < 0.10, "Large discrepancy in volumes between resampled and actual data"

@pytest.mark.asyncio
async def test_data_streaming_to_indicators(collector):
    """Test real-time data streaming through indicators"""
    # Initialize indicators
    wt = WaveTrendIndicator()
    rsi = RSIIndicator()
    
    # Stream processing state
    processed_data = []
    
    # Custom handler that processes data through indicators
    def indicator_processor(message):
        try:
            import json
            data = json.loads(message)
            
            if 'e' in data and data['e'] == 'kline':
                k = data['k']
                
                # Create a small dataframe with this candle
                candle_df = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(k['t'] / 1000)],
                    'open': [float(k['o'])],
                    'high': [float(k['h'])],
                    'low': [float(k['l'])],
                    'close': [float(k['c'])],
                    'volume': [float(k['v'])]
                })
                
                # Process through indicators
                try:
                    wt_signals = wt.calculate_signals(candle_df)
                    rsi_signals = rsi.calculate_signals(candle_df)
                    
                    # Record successful processing
                    processed_data.append({
                        'timestamp': candle_df['timestamp'][0],
                        'close': float(k['c']),
                        'wt1': wt_signals['wt1'].item() if not wt_signals['wt1'].isnan()[0] else None,
                        'rsi': rsi_signals['rsi'].item() if not rsi_signals['rsi'].isnan()[0] else None
                    })
                except Exception as ind_err:
                    logger.error(f"Indicator processing error: {ind_err}")
        
        except Exception as e:
            logger.error(f"Error in indicator processor: {e}")
    
    # Override message handler
    original_handler = collector.message_handler
    collector.message_handler = indicator_processor
    
    try:
        # Start WebSocket
        websocket_task = asyncio.create_task(collector.start_websocket())
        
        # Wait for some data (with timeout)
        timeout = 20  # seconds
        start_time = datetime.now()
        
        while len(processed_data) < 3:  # Wait for at least 3 processed candles
            if (datetime.now() - start_time).total_seconds() > timeout:
                break
            await asyncio.sleep(1)
        
        # Check results
        if len(processed_data) > 0:
            logger.info(f"Processed {len(processed_data)} candles in real-time")
            for i, data in enumerate(processed_data):
                logger.info(f"Candle {i+1}: Close={data['close']}, WT1={data['wt1']}, RSI={data['rsi']}")
        
        assert len(processed_data) > 0, "No candles processed in real-time"
    
    finally:
        # Restore original handler and stop WebSocket
        collector.message_handler = original_handler
        collector.is_running = False
        try:
            await websocket_task
        except:
            pass

@pytest.mark.asyncio
async def test_data_db_query_performance(collector):
    """Test database query performance for historical data"""
    # First ensure we have some data in the database
    # Fetch a few days of data
    data = await collector.fetch_all_timeframes(lookback_days=3)
    
    # Measure query performance
    start_time = datetime.now()
    
    async def timing_test():
        last_week = datetime.now() - timedelta(days=7)
        async with collector._db_pool.acquire() as conn:
            # Test query performance
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = $1 AND timestamp > $2
                ORDER BY timestamp
            """
            
            # Measure execution time
            rows = await conn.fetch(query, collector.symbol, last_week)
            return len(rows)
    
    # Run the test if we have a database connection
    if collector._db_pool:
        row_count = await timing_test()
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query retrieved {row_count} rows in {execution_time:.2f} seconds")
        
        # Performance assertions
        assert execution_time < 5.0, "Database query taking too long"
    else:
        pytest.skip("No database connection available") 