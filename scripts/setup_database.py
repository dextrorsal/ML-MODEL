#!/usr/bin/env python3
"""
Database Setup Script for ML Trading Bot

This script initializes the database schema for the ML trading bot.
It creates all necessary tables in the trading_bot schema if they don't already exist.

Usage:
    python scripts/setup_database.py --connection-string "your_neon_connection_string"
"""

import os
import argparse
import asyncio
import asyncpg
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_schema_if_not_exists(conn, schema_name: str):
    """Create schema if it doesn't exist"""
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
    logger.info(f"Schema '{schema_name}' created or already exists")


async def create_price_data_table(conn, schema_name: str):
    """Create the price_data table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.price_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        open NUMERIC(20, 8) NOT NULL,
        high NUMERIC(20, 8) NOT NULL,
        low NUMERIC(20, 8) NOT NULL,
        close NUMERIC(20, 8) NOT NULL,
        volume NUMERIC(30, 8) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_price_entry UNIQUE(timestamp, symbol)
    )
    """)
    logger.info(f"Table '{schema_name}.price_data' created or already exists")

    # Create index for faster queries
    await conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_price_data_symbol_timestamp 
    ON {schema_name}.price_data(symbol, timestamp DESC)
    """)
    logger.info(f"Index on '{schema_name}.price_data' created")


async def create_signals_table(conn, schema_name: str):
    """Create the signals table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.signals (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        signal_type VARCHAR(10) NOT NULL,  -- BUY, SELL, NEUTRAL
        signal_strength NUMERIC(5, 2) NOT NULL,
        rsi_14 NUMERIC(10, 4),
        wt_value NUMERIC(10, 4),
        cci_20 NUMERIC(10, 4),
        adx_20 NUMERIC(10, 4),
        rsi_9 NUMERIC(10, 4),
        confidence_5m NUMERIC(10, 4),
        confidence_15m NUMERIC(10, 4),
        weighted_confidence NUMERIC(10, 4),
        price NUMERIC(20, 8),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_signal UNIQUE(timestamp, symbol, signal_type)
    )
    """)
    logger.info(f"Table '{schema_name}.signals' created or already exists")

    # Create index for faster queries
    await conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
    ON {schema_name}.signals(symbol, timestamp DESC)
    """)
    logger.info(f"Index on '{schema_name}.signals' created")


async def create_models_table(conn, schema_name: str):
    """Create the models table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.models (
        id SERIAL PRIMARY KEY,
        model_id VARCHAR(100) NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        description TEXT,
        parameters JSONB,
        training_data_start TIMESTAMP,
        training_data_end TIMESTAMP,
        validation_accuracy NUMERIC(10, 4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_model_id UNIQUE(model_id)
    )
    """)
    logger.info(f"Table '{schema_name}.models' created or already exists")


async def create_model_predictions_table(conn, schema_name: str):
    """Create the model_predictions table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.model_predictions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        model_id VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        prediction_value NUMERIC(10, 4) NOT NULL,
        actual_outcome NUMERIC(10, 4),
        confidence NUMERIC(10, 4) NOT NULL,
        features_used JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_prediction UNIQUE(timestamp, model_id, symbol, timeframe)
    )
    """)
    logger.info(f"Table '{schema_name}.model_predictions' created or already exists")

    # Create index for faster queries
    await conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_timestamp 
    ON {schema_name}.model_predictions(symbol, timestamp DESC)
    """)
    logger.info(f"Index on '{schema_name}.model_predictions' created")


async def create_trades_table(conn, schema_name: str):
    """Create the trades table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.trades (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP,
        entry_price NUMERIC(20, 8) NOT NULL,
        exit_price NUMERIC(20, 8),
        quantity NUMERIC(20, 8) NOT NULL,
        direction VARCHAR(10) NOT NULL,  -- LONG or SHORT
        status VARCHAR(20) NOT NULL,     -- OPEN, CLOSED, CANCELED
        pnl NUMERIC(20, 8),
        pnl_percentage NUMERIC(10, 4),
        fees NUMERIC(20, 8),
        net_pnl NUMERIC(20, 8),
        model_id VARCHAR(100),
        signal_id INTEGER,
        confidence NUMERIC(10, 4),
        trade_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    logger.info(f"Table '{schema_name}.trades' created or already exists")

    # Create index for faster queries
    await conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_trades_status 
    ON {schema_name}.trades(status)
    """)
    logger.info(f"Index on '{schema_name}.trades' created")


async def create_backtest_results_table(conn, schema_name: str):
    """Create the backtest_results table if it doesn't exist"""
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.backtest_results (
        id SERIAL PRIMARY KEY,
        model_id VARCHAR(100) NOT NULL,
        strategy_id INTEGER NOT NULL,
        start_date TIMESTAMP NOT NULL,
        end_date TIMESTAMP NOT NULL,
        symbols VARCHAR[] NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        total_trades INTEGER NOT NULL,
        winning_trades INTEGER NOT NULL,
        losing_trades INTEGER NOT NULL,
        win_rate NUMERIC(10, 4) NOT NULL,
        profit_loss_percentage NUMERIC(10, 4) NOT NULL,
        max_drawdown NUMERIC(10, 4) NOT NULL,
        sharpe_ratio NUMERIC(10, 4),
        parameters JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )
    """)
    logger.info(f"Table '{schema_name}.backtest_results' created or already exists")


async def initialize_database(connection_string: str, schema_name: str = "trading_bot"):
    """Initialize the database schema and tables"""
    try:
        # Connect to the database
        conn = await asyncpg.connect(connection_string)
        logger.info("Connected to database")

        # Create schema if it doesn't exist
        await create_schema_if_not_exists(conn, schema_name)

        # Create tables
        await create_price_data_table(conn, schema_name)
        await create_signals_table(conn, schema_name)
        await create_models_table(conn, schema_name)
        await create_model_predictions_table(conn, schema_name)
        await create_trades_table(conn, schema_name)
        await create_backtest_results_table(conn, schema_name)

        logger.info("Database initialization completed successfully")
        await conn.close()
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Initialize database for ML trading bot"
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        default=os.environ.get(
            "NEON_CONNECTION_STRING",
            "postgresql://neondb_owner:PgT1zO2ywrVU@ep-silent-dust-61256651.us-east-2.aws.neon.tech/neondb",
        ),
        help="Database connection string",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="trading_bot",
        help="Schema name to create tables in",
    )
    return parser.parse_args()


async def main():
    """Main function"""
    args = parse_arguments()

    logger.info(f"Initializing database with schema '{args.schema}'")
    success = await initialize_database(args.connection_string, args.schema)

    if success:
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")


if __name__ == "__main__":
    asyncio.run(main())
