#!/usr/bin/env python3
"""
Database Connection Test Script

This script tests the connection to the Neon PostgreSQL database
and validates that we can perform basic operations.
"""

import os
import sys
import asyncio
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.db_connector import DBConnector

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of connection strings to try
CONNECTION_STRINGS = [
    # Connection from test_database.py
    "postgresql://dex:testpassword@localhost:5432/solana_trading_test",
    
    # Connection from db_connector.py
    "postgresql://ml_trader:solanaml2024@ep-silent-dust-61256651.us-east-2.aws.neon.tech/neondb",
    
    # Other possible variations
    "postgresql://neondb_owner:PgT1zO2ywrVU@ep-silent-dust-61256651.us-east-2.aws.neon.tech/neondb",
    
    # Local development database fallback
    "postgresql://postgres:postgres@localhost:5432/postgres"
]

async def test_connection():
    """Test the database connection and basic operations"""
    logger.info("Testing database connections...")
    
    successful_connection = None
    
    # Try each connection string
    for connection_string in CONNECTION_STRINGS:
        try:
            logger.info(f"Trying connection: {connection_string.split('@')[1]}")
            
            # Initialize the DB connector with this connection string
            db = DBConnector(connection_string)
            
            # Test 1: Connect to the database
            await db.connect()
            logger.info("✅ Database connection successful!")
            
            # Test 2: Get database info
            db_info = await db.fetch("""
                SELECT current_database() as db_name, 
                       current_user as username, 
                       version() as db_version
            """)
            
            if db_info:
                logger.info(f"✅ Connected to:")
                logger.info(f"  - Database: {db_info[0]['db_name']}")
                logger.info(f"  - Username: {db_info[0]['username']}")
                logger.info(f"  - Version: {db_info[0]['db_version']}")
                
                # This is a valid connection - save it
                successful_connection = connection_string
                
                # Close this connection before proceeding
                await db.close()
                
                # We found a working connection, break the loop
                break
        
        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            try:
                await db.close()
            except:
                pass
    
    # If we found a working connection, update our files
    if successful_connection:
        logger.info("\n✨ Found a working connection! ✨")
        logger.info(f"Connection string: {successful_connection.split('@')[1]}")
        
        # Update db_connector.py with the working connection
        update_connection_string(successful_connection)
        
        # Now proceed with the full test using the working connection
        await run_full_test(successful_connection)
    else:
        logger.error("❌ All connection attempts failed!")
        sys.exit(1)

def update_connection_string(connection_string):
    """Update the connection string in relevant files"""
    try:
        # Update db_connector.py
        db_connector_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'src', 'utils', 'db_connector.py')
        
        if os.path.exists(db_connector_path):
            logger.info(f"Updating connection string in db_connector.py")
            
            # Read the file
            with open(db_connector_path, 'r') as f:
                content = f.read()
            
            # Replace the connection string
            if 'NEON_CONNECTION_STRING' in content:
                # Find the default connection string
                import re
                match = re.search(r'"postgresql://[^"]*"', content)
                if match:
                    old_conn = match.group(0)
                    new_conn = f'"{connection_string}"'
                    
                    # Replace and write back
                    content = content.replace(old_conn, new_conn)
                    
                    with open(db_connector_path, 'w') as f:
                        f.write(content)
                    
                    logger.info("✅ Updated connection string in db_connector.py")
        
        # Update start_trading_system.py
        trading_system_path = os.path.join(os.path.dirname(__file__), 
                                          'start_trading_system.py')
        
        if os.path.exists(trading_system_path):
            logger.info(f"Updating connection string in start_trading_system.py")
            
            # Read the file
            with open(trading_system_path, 'r') as f:
                content = f.read()
            
            # Replace the connection string
            if '--neon-connection' in content:
                # Find the default connection string
                import re
                match = re.search(r'default="postgresql://[^"]*"', content)
                if match:
                    old_conn = match.group(0)
                    new_conn = f'default="{connection_string}"'
                    
                    # Replace and write back
                    content = content.replace(old_conn, new_conn)
                    
                    with open(trading_system_path, 'w') as f:
                        f.write(content)
                    
                    logger.info("✅ Updated connection string in start_trading_system.py")
    
    except Exception as e:
        logger.error(f"Error updating connection string: {str(e)}")

async def run_full_test(connection_string):
    """Run the full database test with a working connection"""
    logger.info("\n======== Running Full Database Test ========")
    
    # Initialize the DB connector with the working connection
    db = DBConnector(connection_string)
    
    try:
        # Test 1: Connect to the database
        logger.info("Test 1: Connecting to database...")
        await db.connect()
        logger.info("✅ Database connection successful!")
        
        # Test 2: Create a test table for validation
        logger.info("Test 2: Creating a test table...")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS db_test (
                id SERIAL PRIMARY KEY,
                test_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        logger.info("✅ Test table created successfully!")
        
        # Test 3: Insert a test record
        logger.info("Test 3: Inserting test data...")
        await db.execute("""
            INSERT INTO db_test (test_name) VALUES ($1)
        """, "connection_test")
        logger.info("✅ Test data inserted successfully!")
        
        # Test 4: Query the data back
        logger.info("Test 4: Querying test data...")
        results = await db.fetch("""
            SELECT * FROM db_test ORDER BY timestamp DESC LIMIT 5
        """)
        
        logger.info(f"✅ Retrieved {len(results)} test records:")
        for row in results:
            logger.info(f"  - ID: {row['id']}, Name: {row['test_name']}, Time: {row['timestamp']}")
        
        # Test 5: Check existing tables
        logger.info("Test 5: Listing all tables in database...")
        tables = await db.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        logger.info(f"✅ Found {len(tables)} tables in database:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
            
            # Get row count for each table
            try:
                count_result = await db.fetch(f"""
                    SELECT COUNT(*) as count FROM {table['table_name']}
                """)
                if count_result:
                    logger.info(f"    - Row count: {count_result[0]['count']}")
            except:
                logger.warning(f"    - Could not get row count for {table['table_name']}")
        
        logger.info("✨ All database tests completed successfully! ✨")
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {str(e)}")
        raise
    finally:
        # Close the connection
        await db.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    try:
        # Run the test function
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        sys.exit(1) 