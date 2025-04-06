# 🔧 Troubleshooting Guide

This guide helps you diagnose and resolve common issues in the GPU-accelerated trading system.

## 📋 Table of Contents
1. [Installation Issues](#installation-issues)
2. [GPU Problems](#gpu-problems)
3. [Memory Issues](#memory-issues)
4. [Performance Problems](#performance-problems)
5. [Data Issues](#data-issues)
6. [Common Errors](#common-errors)
7. [Debugging Tools](#debugging-tools)

## 🔌 Installation Issues

### 1. CUDA Installation
```bash
# Check CUDA version
nvidia-smi

# Common error: CUDA not found
Error: CUDA driver version is insufficient for CUDA runtime version

# Solution: Install correct CUDA version
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 2. PyTorch Installation
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Common error: CUDA not available
# Solution: Install CUDA-enabled PyTorch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Dependencies
```bash
# Check installed versions
pip list | grep -E "torch|numpy|pandas"

# Common error: Version conflicts
# Solution: Install specific versions
pip install -r requirements.txt --no-cache-dir
```

## 🎮 GPU Problems

### 1. Memory Errors
```python
try:
    # Original code
    data = data.cuda()
    result = process_features(data)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Solution 1: Clear cache
        torch.cuda.empty_cache()
        
        # Solution 2: Process in batches
        batch_size = 1024
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].cuda()
            result = process_features(batch)
            results.append(result.cpu())
        result = torch.cat(results)
```

### 2. Device Errors
```python
# Common error: Tensors on different devices
# Solution: Ensure consistent device usage
class FeatureCalculator:
    def __init__(self):
        self.device = torch.device('cuda')
        
    def process(self, data: torch.Tensor) -> torch.Tensor:
        # Move input to correct device
        data = data.to(self.device)
        
        # Process data
        result = self.calculate(data)
        
        # Return result on CPU if needed
        return result.cpu() if return_cpu else result
```

### 3. Performance Issues
```python
# Problem: Slow GPU processing
# Solution: Profile and optimize

# 1. Use CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# Your GPU code here
end_event.record()

torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)

# 2. Check GPU utilization
nvidia-smi -l 1

# 3. Use NVTX ranges for profiling
with torch.cuda.nvtx.range("feature_calculation"):
    features = calculate_features(data)
```

## 💾 Memory Issues

### 1. Memory Leaks
```python
# Problem: Memory not being released
# Solution: Track and clean up references

class MemoryTracker:
    def __init__(self):
        self.start_memory = None
        
    def __enter__(self):
        # Record starting memory
        torch.cuda.empty_cache()
        self.start_memory = torch.cuda.memory_allocated()
        return self
        
    def __exit__(self, *args):
        # Check for leaks
        torch.cuda.empty_cache()
        end_memory = torch.cuda.memory_allocated()
        leaked = end_memory - self.start_memory
        if leaked > 0:
            print(f"Memory leak detected: {leaked / 1024**2:.2f}MB")
            
# Usage
with MemoryTracker():
    process_data(data)
```

### 2. Out of Memory
```python
# Problem: OOM during processing
# Solution: Implement memory-efficient processing

def process_large_dataset(data: torch.Tensor) -> torch.Tensor:
    # 1. Use gradient checkpointing
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # 2. Process in chunks
    chunk_size = determine_optimal_chunk_size()
    results = []
    
    for chunk in data.split(chunk_size):
        # Process chunk
        with torch.no_grad():  # Disable gradient tracking if not needed
            result = process_chunk(chunk)
        results.append(result.cpu())  # Move to CPU immediately
        
        # Clear cache periodically
        if len(results) % 10 == 0:
            torch.cuda.empty_cache()
            
    return torch.cat(results)

def determine_optimal_chunk_size() -> int:
    """Determine optimal chunk size based on GPU memory."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Use 80% of available memory
    available_memory = total_memory * 0.8
    return int(available_memory / (4 * 1024))  # Assuming 4 bytes per float
```

## ⚡ Performance Problems

### 1. Slow Processing
```python
# Problem: Slow feature calculation
# Solution: Optimize GPU operations

def optimize_feature_calculation():
    # 1. Use in-place operations
    def calculate_momentum(data: torch.Tensor) -> torch.Tensor:
        result = data.clone()
        result.sub_(data.roll(1, dims=1))  # In-place subtraction
        return result
    
    # 2. Minimize CPU-GPU transfers
    def process_features(data: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Move to GPU once
        data_gpu = data.cuda()
        
        results = {}
        for feature in self.features:
            results[feature.name] = feature(data_gpu)
            
        # Move results back to CPU at once
        return {k: v.cpu() for k, v in results.items()}
    
    # 3. Use batch processing
    def calculate_indicators(data: torch.Tensor) -> torch.Tensor:
        batch_size = 1024
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        results = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            
            # Process batch
            result = process_batch(batch)
            results.append(result)
            
        return torch.cat(results)
```

### 2. Bottlenecks
```python
# Problem: Processing bottlenecks
# Solution: Profile and optimize

def profile_processing():
    # 1. Use profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Your code here
        prof.step()
        
    # Print summary
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10))
    
    # 2. Use CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    # Your code here
    end.record()
    
    torch.cuda.synchronize()
    print(f"Processing time: {start.elapsed_time(end)}ms")
```

## 📊 Data Issues

### 1. Data Loading
```python
# Problem: Slow data loading
# Solution: Optimize data pipeline

class DataLoader:
    def __init__(self, batch_size: int = 1024):
        self.batch_size = batch_size
        
    def load_data(self, path: str) -> torch.Tensor:
        try:
            # 1. Memory-mapped loading for large files
            data = np.load(path, mmap_mode='r')
            
            # 2. Load in chunks
            chunks = []
            for i in range(0, len(data), self.batch_size):
                chunk = torch.from_numpy(
                    data[i:i + self.batch_size].copy()
                )
                chunks.append(chunk)
                
            return torch.cat(chunks)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
```

### 2. Data Preprocessing
```python
# Problem: Inefficient preprocessing
# Solution: Optimize preprocessing pipeline

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        # 1. Handle missing values
        if torch.isnan(data).any():
            data = self.handle_missing(data)
            
        # 2. Scale features efficiently
        data_np = data.cpu().numpy()
        data_scaled = self.scaler.fit_transform(data_np)
        
        # 3. Convert back to tensor
        return torch.from_numpy(data_scaled).cuda()
        
    def handle_missing(self, data: torch.Tensor) -> torch.Tensor:
        # Fill missing values with forward fill
        mask = torch.isnan(data)
        data[mask] = float('nan')
        return torch.nan_to_num(data, nan=0.0)
```

## ❌ Common Errors

### 1. CUDA Errors
```python
# Error: CUDA error: device-side assert triggered
# Solution: Check input validity

def validate_input(data: torch.Tensor) -> None:
    """Validate input data."""
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
        
    if data.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {data.dim()}D")
        
    if torch.isnan(data).any():
        raise ValueError("Input contains NaN values")
        
    if not data.is_cuda:
        raise ValueError("Input must be on GPU")
```

### 2. Memory Errors
```python
# Error: RuntimeError: CUDA out of memory
# Solution: Implement memory-efficient processing

class MemoryEfficientProcessor:
    def __init__(self):
        self.batch_size = self.determine_batch_size()
        
    def determine_batch_size(self) -> int:
        """Determine optimal batch size."""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_allocated()
        available = total_memory - free_memory
        
        # Use 80% of available memory
        return int(available * 0.8 / (4 * 1024))
        
    def process(self, data: torch.Tensor) -> torch.Tensor:
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size].cuda()
            result = self.process_batch(batch)
            results.append(result.cpu())
            
            # Clear cache periodically
            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()
                
        return torch.cat(results)
```

## 🔍 Debugging Tools

### 1. Memory Profiler
```python
class MemoryProfiler:
    """Profile memory usage."""
    
    def __init__(self):
        self.memory_stats = []
        
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        return self
        
    def __exit__(self, *args):
        self.memory_stats.append({
            'allocated': torch.cuda.memory_allocated(),
            'peak': torch.cuda.max_memory_allocated(),
            'cached': torch.cuda.memory_reserved()
        })
        
    def report(self):
        """Generate memory report."""
        for i, stats in enumerate(self.memory_stats):
            print(f"Checkpoint {i}:")
            print(f"  Allocated: {stats['allocated'] / 1024**2:.2f}MB")
            print(f"  Peak: {stats['peak'] / 1024**2:.2f}MB")
            print(f"  Cached: {stats['cached'] / 1024**2:.2f}MB")
```

### 2. CUDA Profiler
```python
def profile_cuda_operations():
    """Profile CUDA operations."""
    with torch.cuda.profiler.profile():
        with torch.cuda.nvtx.range("processing"):
            # Your code here
            pass
            
    # Enable autograd profiler
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        # Your code here
        pass
        
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10))
```

### 3. Error Logger
```python
class ErrorLogger:
    """Log and analyze errors."""
    
    def __init__(self, log_file: str = "error_log.txt"):
        self.log_file = log_file
        
    def log_error(self, error: Exception, context: dict):
        """Log error with context."""
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {type(error).__name__}: {str(error)}\n")
            f.write("Context:\n")
            for k, v in context.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
    def analyze_logs(self):
        """Analyze error patterns."""
        with open(self.log_file, 'r') as f:
            logs = f.readlines()
            
        error_counts = {}
        for log in logs:
            if ': ' in log:
                error_type = log.split(': ')[0].split(']')[1].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
        return error_counts
``` 