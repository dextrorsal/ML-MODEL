#!/usr/bin/env python
"""
Setup Test Environment for Data Collection and Storage Tests

This script:
1. Creates a test database for data collection tests
2. Populates it with some sample data
3. Verifies the connection to Binance API
"""

import asyncio
import asyncpg
import argparse
import os
import sys
from datetime import datetime, timedelta
import logging
import pandas as pd
import json
import sqlite3
import aiosqlite

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.collectors.sol_data_collector import SOLDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default test database connection string
DEFAULT_TEST_DB = "postgresql://dex:testpassword@localhost:5432/solana_trading_test"
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "test_data.db")

async def setup_sqlite_database():
    """Setup SQLite database as a fallback"""
    logger.info(f"Setting up SQLite database at: {SQLITE_DB_PATH}")
    
    try:
        # Create SQLite database
        async with aiosqlite.connect(SQLITE_DB_PATH) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (timestamp, symbol)
                )
            ''')
            
            # Create index for faster queries
            await db.execute('''
                CREATE INDEX IF NOT EXISTS price_data_symbol_timestamp_idx 
                ON price_data(symbol, timestamp)
            ''')
            
            await db.commit()
            logger.info("SQLite database setup completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"SQLite database setup failed: {str(e)}")
        return False

async def setup_database(conn_str):
    """Setup the test database schema"""
    logger.info(f"Setting up database schema using connection: {conn_str}")
    
    try:
        # Create connection pool
        pool = await asyncpg.create_pool(conn_str)
        
        async with pool.acquire() as conn:
            # Create tables
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
            
            # Check if the table was created successfully
            result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'price_data'")
            if result == 1:
                logger.info("Database setup completed successfully")
            else:
                logger.error("Failed to create price_data table")
                
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL database setup failed: {str(e)}")
        logger.info("Falling back to SQLite database")
        return await setup_sqlite_database()

async def populate_sqlite_sample_data(days=7):
    """Populate the SQLite database with sample data from Binance"""
    logger.info(f"Fetching sample data for last {days} days from Binance for SQLite")
    
    try:
        # Initialize data collector without a connection string
        collector = SOLDataCollector()
        
        # Fetch data
        data = await collector.fetch_all_timeframes(lookback_days=days)
        
        if not data:
            logger.error("Failed to fetch data from Binance")
            return False
            
        # Insert data into SQLite database
        async with aiosqlite.connect(SQLITE_DB_PATH) as db:
            # Get existing count
            cursor = await db.execute("SELECT COUNT(*) FROM price_data")
            count_before = (await cursor.fetchone())[0]
            
            # Process each timeframe
            for tf_name, df in data.items():
                logger.info(f"Processing {len(df)} rows from {tf_name} timeframe")
                
                # Prepare and insert data
                for _, row in df.iterrows():
                    timestamp = row['timestamp'].isoformat()
                    await db.execute("""
                        INSERT INTO price_data (timestamp, symbol, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (timestamp, symbol) DO UPDATE
                        SET open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close,
                            volume = excluded.volume
                    """, (
                        timestamp,
                        collector.symbol,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
            
            await db.commit()
            
            # Verify data was inserted
            cursor = await db.execute("SELECT COUNT(*) FROM price_data")
            count_after = (await cursor.fetchone())[0]
            logger.info(f"Inserted {count_after - count_before} new rows into the SQLite database")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate SQLite sample data: {str(e)}")
        return False

async def populate_sample_data(conn_str, days=7):
    """Populate the database with sample data from Binance"""
    logger.info(f"Fetching sample data for last {days} days from Binance")
    
    try:
        # Initialize data collector
        collector = SOLDataCollector(conn_str)
        
        # Fetch data
        data = await collector.fetch_all_timeframes(lookback_days=days)
        
        if not data:
            logger.error("Failed to fetch data from Binance")
            return False
            
        # Try to connect to PostgreSQL
        try:
            # Insert data into database
            pool = await asyncpg.create_pool(conn_str)
            
            async with pool.acquire() as conn:
                # Check existing data
                count_before = await conn.fetchval("SELECT COUNT(*) FROM price_data")
                
                # Process each timeframe
                for tf_name, df in data.items():
                    logger.info(f"Processing {len(df)} rows from {tf_name} timeframe")
                    
                    # Prepare batch data
                    batch_data = []
                    for _, row in df.iterrows():
                        batch_data.append((
                            row['timestamp'],
                            collector.symbol,
                            float(row['open']),
                            float(row['high']),
                            float(row['low']),
                            float(row['close']),
                            float(row['volume'])
                        ))
                    
                    # Insert data in batches
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
                
                # Verify data was inserted
                count_after = await conn.fetchval("SELECT COUNT(*) FROM price_data")
                logger.info(f"Inserted {count_after - count_before} new rows into the database")
                
            await pool.close()
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {str(e)}")
            logger.info("Falling back to SQLite database")
            return await populate_sqlite_sample_data(days)
        
    except Exception as e:
        logger.error(f"Failed to populate sample data: {str(e)}")
        return False

async def verify_binance_connection():
    """Verify connection to Binance API"""
    logger.info("Verifying Binance API connection...")
    
    try:
        collector = SOLDataCollector()
        
        # Try to fetch a small amount of data
        test_data = await collector.fetch_all_timeframes(lookback_days=1)
        
        if test_data and len(test_data) > 0:
            # Print some basic info about the fetched data
            logger.info("Successfully connected to Binance API")
            
            for tf, df in test_data.items():
                logger.info(f"  {tf}: {len(df)} candles, latest close: {df['close'].iloc[-1]:.2f}")
                
            return True
        else:
            logger.error("No data received from Binance API")
            return False
            
    except Exception as e:
        logger.error(f"Binance API connection failed: {str(e)}")
        return False

async def verify_websocket_connection():
    """Verify WebSocket connection to Binance"""
    logger.info("Verifying Binance WebSocket connection...")
    
    try:
        collector = SOLDataCollector()
        received_messages = []
        
        def test_handler(message):
            received_messages.append(message)
        
        # Override message handler
        collector.message_handler = test_handler
        
        # Start WebSocket
        collector.is_running = True
        websocket_task = asyncio.create_task(collector.start_websocket())
        
        # Wait for some messages
        timeout = 15  # seconds
        for i in range(timeout):
            if len(received_messages) > 0:
                break
            await asyncio.sleep(1)
            logger.info(f"Waiting for WebSocket messages... ({i+1}/{timeout}s)")
        
        # Stop WebSocket
        collector.is_running = False
        try:
            await asyncio.wait_for(websocket_task, timeout=1)
        except:
            pass
        
        if len(received_messages) > 0:
            logger.info(f"Successfully received {len(received_messages)} WebSocket messages")
            # Print the first message for debugging
            if len(received_messages) > 0:
                try:
                    first_msg = json.loads(received_messages[0])
                    logger.info(f"First message type: {first_msg.get('e', 'unknown')}")
                except:
                    logger.info("Could not parse first message as JSON")
            return True
        else:
            logger.error("No messages received from WebSocket")
            return False
            
    except Exception as e:
        logger.error(f"WebSocket connection failed: {str(e)}")
        return False

async def run_setup(conn_str, days):
    """Run all setup tasks"""
    # 1. Verify Binance connection
    binance_ok = await verify_binance_connection()
    if not binance_ok:
        logger.error("Binance API connection check failed. Setup cannot continue.")
        return False
    
    # 2. Verify WebSocket connection
    websocket_ok = await verify_websocket_connection()
    if not websocket_ok:
        logger.warning("WebSocket connection check failed, but proceeding with setup.")
    
    # 3. Setup database
    db_ok = await setup_database(conn_str)
    if not db_ok:
        logger.error("Database setup failed. Setup cannot continue.")
        return False
    
    # 4. Populate sample data
    data_ok = await populate_sample_data(conn_str, days)
    if not data_ok:
        logger.error("Sample data population failed.")
        return False
        
    logger.info("==== Test environment setup completed successfully ====")
    
    # Check if we're using SQLite
    if os.path.exists(SQLITE_DB_PATH):
        logger.info(f"Using SQLite database: {SQLITE_DB_PATH}")
        os.environ["SQLITE_TEST_DB"] = SQLITE_DB_PATH
        logger.info("\nYou can now run the tests with:")
        logger.info(f"SQLITE_TEST_DB='{SQLITE_DB_PATH}' python -m pytest tests/data/ -v")
    else:
        logger.info(f"Using PostgreSQL database: {conn_str}")
        logger.info("\nYou can now run the tests with:")
        logger.info(f"NEON_TEST_DB='{conn_str}' python -m pytest tests/data/ -v")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup test environment for data collection tests")
    parser.add_argument("--conn", type=str, default=DEFAULT_TEST_DB, 
                        help="Database connection string")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of historical data to fetch")
    args = parser.parse_args()
    
    asyncio.run(run_setup(args.conn, args.days))

if __name__ == "__main__":
    main() 