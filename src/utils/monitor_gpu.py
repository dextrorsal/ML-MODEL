import torch
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def monitor_gpu():
    if not torch.cuda.is_available():
        logger.error("No GPU available!")
        return
    
    try:
        while True:
            # Get current device
            device = torch.cuda.current_device()
            
            # Get memory information
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9
            max_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            # Print status
            logger.info(f"\nGPU Status - {torch.cuda.get_device_name(device)}")
            logger.info(f"Memory Used: {memory_allocated:.2f} GB")
            logger.info(f"Memory Reserved: {memory_reserved:.2f} GB")
            logger.info(f"Memory Available: {max_memory - memory_reserved:.2f} GB")
            logger.info("-" * 50)
            
            # Wait before next update
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Error monitoring GPU: {e}")

if __name__ == "__main__":
    monitor_gpu() 