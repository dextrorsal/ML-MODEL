#!/usr/bin/env python
"""
Simple test script to verify the SOLDataCollector functionality
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.collectors.sol_data_collector import SOLDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_binance_api():
    """Test basic Binance API connectivity"""
    try:
        collector = SOLDataCollector()
        
        # Check timeframes
        logger.info(f"Timeframes: {collector.timeframes}")
        
        # Try fetching data for one day (1d timeframe only for speed)
        logger.info("Fetching 1d data...")
        
        klines = collector.client.klines(
            symbol=collector.symbol,
            interval="1d",
            limit=7
        )
        
        logger.info(f"Successfully fetched {len(klines)} 1d candles")
        
        # Print last candle for verification
        if klines:
            last_candle = klines[-1]
            logger.info(f"Latest candle: Open={last_candle[1]}, High={last_candle[2]}, Low={last_candle[3]}, Close={last_candle[4]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Binance API test failed: {str(e)}")
        return False

async def test_websocket():
    """Test WebSocket streaming connection"""
    collector = SOLDataCollector()
    success = False
    
    try:
        # Set up a simple message counter
        message_count = 0
        
        def message_counter(ws, message):
            nonlocal message_count
            message_count += 1
            logger.info(f"Received message #{message_count}: {message[:100]}...")
        
        # Override message handler
        collector.on_message = message_counter
        
        # Start WebSocket
        logger.info("Connecting to WebSocket...")
        
        # Connect manually
        if collector._connect_websocket():
            logger.info("WebSocket connected successfully")
            
            # Subscribe to just one stream for testing
            stream = f"{collector.symbol.lower()}@kline_1m"
            if collector._subscribe_to_streams([stream]):
                logger.info(f"Subscribed to {stream}")
                
                # Wait for messages
                logger.info("Waiting for messages (10 seconds)...")
                for i in range(10):
                    await asyncio.sleep(1)
                    logger.info(f"Waiting... {i+1}/10s - Messages received: {message_count}")
                
                if message_count > 0:
                    logger.info(f"✅ Received {message_count} messages")
                    success = True
                else:
                    logger.error("❌ No messages received")
            else:
                logger.error("❌ Failed to subscribe to stream")
        else:
            logger.error("❌ Failed to connect to WebSocket")
        
        # Cleanup
        if collector.ws:
            collector.ws.close()
            logger.info("WebSocket closed")
        
        return success
        
    except Exception as e:
        logger.error(f"WebSocket test failed: {str(e)}")
        if collector.ws:
            try:
                collector.ws.close()
            except:
                pass
        return False

async def main():
    logger.info("\n======== Testing Binance API Connection ========")
    api_result = await test_binance_api()
    
    logger.info("\n======== Testing WebSocket Connection ========")
    ws_result = await test_websocket()
    
    # Print summary
    logger.info("\n======== Test Summary ========")
    logger.info(f"API Test: {'✅ PASSED' if api_result else '❌ FAILED'}")
    logger.info(f"WebSocket Test: {'✅ PASSED' if ws_result else '❌ FAILED'}")
    
    if api_result and ws_result:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed!")

if __name__ == "__main__":
    asyncio.run(main()) 