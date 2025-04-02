# Data Collection & Storage Tests

This directory contains tests for the data collection and storage components of the ML trading system.

## Prerequisites

- Python 3.9+
- PostgreSQL database (local or Neon)
- Access to Binance API 

## Setup

Before running the tests, you need to set up the test environment:

1. Make sure PostgreSQL is running and accessible
2. Set up the test database by running:

```bash
python setup_test_environment.py --conn "postgresql://user:password@localhost:5432/test_db" --days 7
```

This script will:
- Verify the connection to Binance API
- Create the necessary database tables
- Populate the database with sample data (7 days by default)

## Running the Tests

After setting up the environment, you can run the tests with:

```bash
# Run with default database
python -m pytest -v

# Or specify a custom database connection
NEON_TEST_DB="postgresql://user:password@localhost:5432/test_db" python -m pytest -v
```

## Test Categories

The tests are organized into several categories:

1. **Basic API Tests** - Testing the connection to Binance API
   - `test_binance_connection()` - Tests basic connectivity
   - `test_websocket_streaming()` - Tests WebSocket data streaming

2. **Database Tests** - Testing the database storage functionality
   - `test_database_storage()` - Tests data insertion and retrieval
   - `test_data_db_query_performance()` - Tests database query performance

3. **Integration Tests** - Testing the integration with ML models
   - `test_data_indicator_compatibility()` - Tests data compatibility with indicators
   - `test_data_lorentzian_compatibility()` - Tests data compatibility with classifier
   - `test_timeframe_transformation()` - Tests timeframe resampling
   - `test_data_streaming_to_indicators()` - Tests real-time data streaming

## Troubleshooting

Common issues and solutions:

1. **Database Connection Issues**
   - Check if PostgreSQL is running
   - Verify connection string and credentials
   - Ensure the specified database exists
   - Make sure your database user has proper permissions (use `ALTER USER username WITH SUPERUSER;` if needed)

2. **Binance API Issues**
   - Check your internet connection
   - If using a VPN, try disabling it or changing locations
   - If rate-limited, wait a few minutes and try again
   - Verify API keys are valid and have proper permissions

3. **WebSocket Issues**
   - If WebSocket tests fail, verify your network allows WebSocket connections
   - Some proxies or firewalls may block WebSocket traffic
   - Ensure proper ping/pong handling is implemented - Binance requires responding to ping frames with the same payload
   - If connection drops frequently, implement automatic reconnection with exponential backoff
   - Check the WebSocket logs for specific error messages (configure logging.DEBUG for detailed information)

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use async/await for all database and API operations
2. Use fixtures to share resources between tests
3. Make tests independent of each other
4. Clean up any created resources after tests 