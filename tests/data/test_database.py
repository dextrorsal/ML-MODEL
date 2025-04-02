#!/usr/bin/env python
"""
Test script for database functionality with the SOLDataCollector
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import asyncpg

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.collectors.sol_data_collector import SOLDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PostgreSQL connection string - update with your values
DB_CONN_STR = "postgresql://dex:testpassword@localhost:5432/solana_trading_test"

async def setup_database():
    """Create the test database table"""
    try:
        logger.info(f"Setting up database at {DB_CONN_STR}")
        
        # Create connection
        conn = await asyncpg.connect(DB_CONN_STR)
        
        # Create price_data table
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
        
        # Create index for faster queries
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS price_data_symbol_timestamp_idx 
            ON price_data(symbol, timestamp)
        ''')
        
        logger.info("Database table created successfully")
        
        # Cleanup
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

async def test_database_insert():
    """Test data insertion into database"""
    try:
        # Create collector with database connection
        collector = SOLDataCollector(DB_CONN_STR)
        
        # Initialize database pool
        await collector.init_db()
        
        # Generate test data
        test_data = {
            'symbol': collector.symbol,
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 103.0,
            'volume': 1000.0
        }
        
        # Start collector running
        collector.is_running = True
        
        # Add data to queue
        collector.data_queue.put(test_data)
        
        # Start database worker
        db_worker_task = asyncio.create_task(collector.db_worker())
        
        # Wait a moment for processing
        logger.info("Inserted test data, waiting for processing...")
        await asyncio.sleep(2)
        
        # Stop the worker
        collector.is_running = False
        
        # Check if data was inserted
        pool = await asyncpg.create_pool(DB_CONN_STR)
        async with pool.acquire() as conn:
            # Query for the inserted data
            timestamp_start = test_data['timestamp'] - timedelta(seconds=1)
            timestamp_end = test_data['timestamp'] + timedelta(seconds=1)
            
            row = await conn.fetchrow('''
                SELECT * FROM price_data 
                WHERE symbol = $1 AND 
                timestamp BETWEEN $2 AND $3
            ''', collector.symbol, timestamp_start, timestamp_end)
            
            if row:
                logger.info("✅ Data was successfully inserted into database")
                logger.info(f"Retrieved row: {row}")
                
                # Verify values
                assert float(row['open']) == test_data['open'], "Open price mismatch"
                assert float(row['high']) == test_data['high'], "High price mismatch"
                assert float(row['low']) == test_data['low'], "Low price mismatch"
                assert float(row['close']) == test_data['close'], "Close price mismatch"
                assert float(row['volume']) == test_data['volume'], "Volume mismatch"
                
                # Clean up test data
                await conn.execute('''
                    DELETE FROM price_data 
                    WHERE symbol = $1 AND 
                    timestamp BETWEEN $2 AND $3
                ''', collector.symbol, timestamp_start, timestamp_end)
                
                logger.info("Test data cleaned up")
                
                # Close pool
                await pool.close()
                return True
            else:
                logger.error("❌ Data was not inserted into database")
                await pool.close()
                return False
    
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

async def main():
    logger.info("\n======== Setting Up Database ========")
    db_setup = await setup_database()
    
    if db_setup:
        logger.info("\n======== Testing Database Insert ========")
        db_insert = await test_database_insert()
        
        # Print summary
        logger.info("\n======== Test Summary ========")
        logger.info(f"Database Setup: {'✅ PASSED' if db_setup else '❌ FAILED'}")
        logger.info(f"Database Insert: {'✅ PASSED' if db_insert else '❌ FAILED'}")
        
        if db_setup and db_insert:
            logger.info("\n✅ All database tests passed!")
        else:
            logger.error("\n❌ Some database tests failed!")
    else:
        logger.error("\n❌ Database setup failed, skipping tests.")

if __name__ == "__main__":
    asyncio.run(main()) 