print("Checking Deep Learning Framework Installations...\n")

# Check PyTorch
print("=== PyTorch ===")
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple M1/M2 GPU Available")
    elif hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f"ROCm/HIP Version: {torch.version.hip}")
        if torch.cuda.is_available():  # ROCm uses CUDA API
            print(f"AMD GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU Support detected")
except ImportError:
    print("PyTorch is not installed")

print("\n=== TensorFlow ===")
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    print("GPU Devices:", end=" ")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                print(f"\n - {gpu.device_type}: {gpu.name}")
        else:
            print("No GPU devices found")
    except:
        print("Error checking GPU devices")
except ImportError:
    print("TensorFlow is not installed")

# Test GPU Performance
print("\n=== Quick GPU Performance Test ===")
try:
    import torch
    if torch.cuda.is_available() or (hasattr(torch, 'version') and hasattr(torch.version, 'hip')):
        # Create random tensors
        size = (5000, 5000)
        a = torch.randn(*size, device='cuda')
        b = torch.randn(*size, device='cuda')
        
        # Warmup
        torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Test
        import time
        start = time.time()
        for _ in range(10):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        print(f"PyTorch GPU Matrix Multiplication (5000x5000) - Average time: {(end-start)/10:.4f} seconds")
    else:
        print("Skipping PyTorch GPU test - No GPU available")
except:
    print("Error during PyTorch GPU test")

try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        # Create random tensors
        size = (5000, 5000)
        a = tf.random.normal(size)
        b = tf.random.normal(size)
        
        # Warmup
        tf.matmul(a, b)
        
        # Test
        import time
        start = time.time()
        for _ in range(10):
            tf.matmul(a, b)
        end = time.time()
        print(f"TensorFlow GPU Matrix Multiplication (5000x5000) - Average time: {(end-start)/10:.4f} seconds")
    else:
        print("Skipping TensorFlow GPU test - No GPU available")
except:
    print("Error during TensorFlow GPU test") 