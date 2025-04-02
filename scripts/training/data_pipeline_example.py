"""
Example script showing how to use the Neon data pipeline
"""

from datetime import datetime, timedelta
from src.data.pipeline.neon_collector import NeonDataCollector
from src.data.pipeline.neon_processor import NeonDataProcessor
from src.data.pipeline.neon_batch_loader import NeonBatchLoader
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get Neon connection string from environment
    connection_string = os.getenv('NEON_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("Please set NEON_CONNECTION_STRING in .env file")
        
    try:
        # Initialize components
        collector = NeonDataCollector(connection_string)
        processor = NeonDataProcessor(connection_string)
        batch_loader = NeonBatchLoader(batch_size=32)
        
        # Set parameters
        symbol = 'BTC/USD'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # 1. Collect historical data
        logger.info(f"Collecting historical data for {symbol}...")
        collector.collect_historical(symbol, days=30)
        
        # 2. Get and process data
        logger.info("Getting and processing data...")
        df = processor.get_price_data(symbol, start_date, end_date)
        df = processor.calculate_features(df)
        
        # 3. Prepare data for ML
        logger.info("Preparing data for ML...")
        X, y = processor.prepare_ml_data(
            df,
            target_column='returns',
            lookback=10,
            prediction_horizon=1
        )
        
        # 4. Split into train/test
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = processor.create_train_test_split(
            X, y, train_size=0.8
        )
        
        # 5. Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, test_loader = batch_loader.create_dataloaders(
            X_train, X_test, y_train, y_test
        )
        
        # Print batch information
        train_info = batch_loader.get_batch_info(train_loader)
        logger.info("Training data info:")
        for key, value in train_info.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("Data pipeline setup complete!")
        
        return train_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise
        
if __name__ == "__main__":
    main() 