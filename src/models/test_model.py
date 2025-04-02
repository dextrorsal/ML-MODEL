import torch
import pandas as pd
import logging
from double_bottom_detector_v2 import DoubleBottomDetector, DatabaseManager, prepare_data
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test if data can be loaded and preprocessed correctly"""
    try:
        df_5m = pd.read_csv('data/sol_perp_5m.csv')
        logger.info("✓ 5m data loaded successfully")
        logger.info(f"  Shape: {df_5m.shape}")
        logger.info(f"  Columns: {df_5m.columns.tolist()}")
        
        # Check for missing values
        missing = df_5m.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"⚠ Found missing values:\n{missing[missing > 0]}")
        else:
            logger.info("✓ No missing values found")
            
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {str(e)}")
        return False

def test_data_preprocessing():
    """Test if data preprocessing works"""
    try:
        df_5m = pd.read_csv('data/sol_perp_5m.csv')
        X, y = prepare_data(df_5m)
        
        logger.info("✓ Data preprocessing successful")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Device: {X.device}")
        
        # Check class balance
        positive_ratio = y.mean().item()
        logger.info(f"  Positive class ratio: {positive_ratio:.2%}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data preprocessing failed: {str(e)}")
        return False

def test_model_creation():
    """Test if model can be created and moved to GPU"""
    try:
        model = DoubleBottomDetector()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test with dummy data
        batch_size = 32
        seq_len = 60
        features = 5
        X = torch.randn(batch_size, seq_len, features).to(device)
        output = model(X)
        
        logger.info("✓ Model creation and forward pass successful")
        logger.info(f"  Device: {next(model.parameters()).device}")
        logger.info(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model creation failed: {str(e)}")
        return False

def test_database_connection():
    """Test if database connection and operations work"""
    try:
        db = DatabaseManager()
        
        # Test session creation
        test_params = {
            'test': True,
            'batch_size': 32
        }
        session_id = db.start_training_session('TestModel', 10, test_params)
        logger.info(f"✓ Created test session with ID: {session_id}")
        
        # Test progress update
        db.update_training_progress(session_id, 1, 0.5, 0.75)
        logger.info("✓ Updated training progress")
        
        # Test session completion
        db.complete_training_session(session_id, {'final_loss': 0.4})
        logger.info("✓ Completed training session")
        
        return True
    except Exception as e:
        logger.error(f"✗ Database operations failed: {str(e)}")
        return False

def main():
    logger.info("Starting tests...")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Creation", test_model_creation),
        ("Database Operations", test_database_connection)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if not test_func():
                all_passed = False
                logger.error(f"✗ {test_name} test failed")
        except Exception as e:
            all_passed = False
            logger.error(f"✗ {test_name} test failed with exception: {str(e)}")
    
    if all_passed:
        logger.info("\n✓ All tests passed! You can start training.")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 