# 📊 Performance Benchmarks

This guide outlines the benchmarking procedures for our GPU-accelerated trading system.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Benchmark Types](#benchmark-types)
3. [Running Benchmarks](#running-benchmarks)
4. [Performance Targets](#performance-targets)
5. [Profiling Tools](#profiling-tools)
6. [Results Analysis](#results-analysis)

## 📈 Overview

Our benchmarking suite measures:
- Feature calculation speed
- Memory usage
- GPU utilization
- Batch processing efficiency
- End-to-end latency

## 🔍 Benchmark Types

### 1. Feature Benchmarks
```python
def benchmark_features():
    """Benchmark feature calculation performance."""
    features = [
        RSIFeature(),
        MomentumFeature(),
        VolatilityFeature()
    ]
    
    results = {}
    for feature in features:
        # Test different data sizes
        for size in [1000, 10000, 100000]:
            data = generate_test_data(size)
            
            # Measure performance
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated()
            
            feature(data)
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated()
            
            results[f"{feature.name}_{size}"] = {
                'time': end_time - start_time,
                'memory': (end_memory - start_memory) / 1024**2
            }
            
    return results
```

### 2. Memory Benchmarks
```python
def benchmark_memory():
    """Benchmark memory usage patterns."""
    results = {
        'peak_memory': [],
        'memory_timeline': [],
        'allocation_points': []
    }
    
    # Track memory usage
    def memory_callback():
        results['memory_timeline'].append(
            torch.cuda.memory_allocated() / 1024**2
        )
    
    # Register callback
    torch.cuda.memory._record_memory_history(
        enabled=True,
        callback=memory_callback
    )
    
    # Run workload
    try:
        run_trading_pipeline()
    finally:
        torch.cuda.memory._record_memory_history(enabled=False)
        
    # Analyze results
    results['peak_memory'] = torch.cuda.max_memory_allocated() / 1024**2
    
    return results
```

### 3. Throughput Benchmarks
```python
def benchmark_throughput():
    """Benchmark data processing throughput."""
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    results = {}
    
    for batch_size in batch_sizes:
        # Generate test data
        data = generate_batch_data(batch_size)
        
        # Measure processing speed
        start_time = time.time()
        samples_processed = 0
        
        while samples_processed < 1_000_000:
            process_batch(data)
            samples_processed += batch_size
            
        end_time = time.time()
        
        # Calculate throughput
        duration = end_time - start_time
        throughput = samples_processed / duration
        
        results[batch_size] = {
            'throughput': throughput,
            'latency': duration / (samples_processed / batch_size)
        }
        
    return results
```

### 4. End-to-End Benchmarks
```python
def benchmark_pipeline():
    """Benchmark complete trading pipeline."""
    results = {
        'feature_time': 0,
        'model_time': 0,
        'signal_time': 0,
        'total_time': 0,
        'memory_usage': 0
    }
    
    # Setup pipeline
    pipeline = TradingPipeline()
    data = load_test_data()
    
    # Measure feature calculation
    start = time.time()
    features = pipeline.calculate_features(data)
    results['feature_time'] = time.time() - start
    
    # Measure model inference
    start = time.time()
    predictions = pipeline.generate_predictions(features)
    results['model_time'] = time.time() - start
    
    # Measure signal generation
    start = time.time()
    signals = pipeline.generate_signals(predictions)
    results['signal_time'] = time.time() - start
    
    # Record total time and memory
    results['total_time'] = sum([
        results['feature_time'],
        results['model_time'],
        results['signal_time']
    ])
    results['memory_usage'] = torch.cuda.max_memory_allocated() / 1024**2
    
    return results
```

## 🚀 Running Benchmarks

### 1. Command Line Interface
```bash
# Run all benchmarks
python benchmarks/run_all.py

# Run specific benchmark
python benchmarks/run_benchmark.py --type features

# Run with different configurations
python benchmarks/run_benchmark.py --batch-sizes "32,64,128,256"
```

### 2. Configuration
```python
# benchmark_config.py
BENCHMARK_CONFIG = {
    'batch_sizes': [32, 64, 128, 256, 512, 1024],
    'data_sizes': [1000, 10000, 100000],
    'warmup_iterations': 5,
    'test_iterations': 20,
    'memory_threshold': 4 * 1024,  # 4GB
    'max_latency': 0.1,  # 100ms
}
```

### 3. Results Output
```python
def save_results(results: Dict[str, Any], name: str):
    """Save benchmark results."""
    # Save raw data
    with open(f"results/{name}_raw.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_results(results, f"results/{name}_plots.png")
    
    # Generate report
    generate_report(results, f"results/{name}_report.md")
```

## 🎯 Performance Targets

### 1. Latency Targets
```python
LATENCY_TARGETS = {
    'feature_calculation': 0.005,  # 5ms
    'model_inference': 0.010,    # 10ms
    'signal_generation': 0.002,  # 2ms
    'total_pipeline': 0.020      # 20ms
}

def verify_latency(results: Dict[str, float]):
    """Verify latency meets targets."""
    for key, target in LATENCY_TARGETS.items():
        actual = results.get(key, float('inf'))
        assert actual <= target, f"{key} latency {actual:.3f}s exceeds target {target:.3f}s"
```

### 2. Memory Targets
```python
MEMORY_TARGETS = {
    'peak_memory': 4 * 1024,  # 4GB
    'steady_state': 2 * 1024, # 2GB
    'feature_memory': 512,    # 512MB
    'model_memory': 1024     # 1GB
}

def verify_memory(results: Dict[str, float]):
    """Verify memory usage meets targets."""
    for key, target in MEMORY_TARGETS.items():
        actual = results.get(key, float('inf'))
        assert actual <= target, f"{key} memory {actual:.0f}MB exceeds target {target:.0f}MB"
```

### 3. Throughput Targets
```python
THROUGHPUT_TARGETS = {
    'samples_per_second': 10000,
    'features_per_second': 1000000,
    'signals_per_second': 5000
}

def verify_throughput(results: Dict[str, float]):
    """Verify throughput meets targets."""
    for key, target in THROUGHPUT_TARGETS.items():
        actual = results.get(key, 0)
        assert actual >= target, f"{key} throughput {actual:.0f} below target {target:.0f}"
```

## 🔧 Profiling Tools

### 1. Memory Profiling
```python
def profile_memory():
    """Profile memory usage patterns."""
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
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/memory'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run workload
        run_trading_pipeline()
        prof.step()
    
    # Print summary
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=10))
```

### 2. CUDA Profiling
```python
def profile_cuda():
    """Profile CUDA operations."""
    with torch.cuda.profiler.profile():
        with torch.cuda.nvtx.range("trading_pipeline"):
            # Feature calculation
            with torch.cuda.nvtx.range("features"):
                features = calculate_features()
            
            # Model inference
            with torch.cuda.nvtx.range("model"):
                predictions = run_model(features)
            
            # Signal generation
            with torch.cuda.nvtx.range("signals"):
                signals = generate_signals(predictions)
```

### 3. Timeline Profiling
```python
def profile_timeline():
    """Profile operation timeline."""
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10))
        
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
        on_trace_ready=trace_handler
    ) as prof:
        run_trading_pipeline()
        prof.step()
```

## 📊 Results Analysis

### 1. Generate Reports
```python
def generate_report(results: Dict[str, Any]):
    """Generate benchmark report."""
    report = []
    
    # Summary
    report.append("# Benchmark Results\n")
    report.append(f"Date: {datetime.now()}\n")
    
    # Performance metrics
    report.append("## Performance Metrics\n")
    report.append("### Latency\n")
    for key, value in results['latency'].items():
        report.append(f"- {key}: {value:.3f}ms\n")
    
    # Memory usage
    report.append("### Memory Usage\n")
    for key, value in results['memory'].items():
        report.append(f"- {key}: {value:.0f}MB\n")
    
    # Throughput
    report.append("### Throughput\n")
    for key, value in results['throughput'].items():
        report.append(f"- {key}: {value:.0f} samples/sec\n")
    
    # Save report
    with open("results/report.md", 'w') as f:
        f.write('\n'.join(report))
```

### 2. Plot Results
```python
def plot_results(results: Dict[str, Any]):
    """Plot benchmark results."""
    # Latency plot
    plt.figure(figsize=(10, 6))
    plt.bar(results['latency'].keys(), results['latency'].values())
    plt.title("Operation Latency")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/latency.png")
    
    # Memory plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['memory_timeline'])
    plt.title("Memory Usage Over Time")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("results/memory.png")
    
    # Throughput plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['batch_sizes'], results['throughput'])
    plt.title("Throughput vs Batch Size")
    plt.ylabel("Samples/sec")
    plt.xlabel("Batch Size")
    plt.tight_layout()
    plt.savefig("results/throughput.png")
```

### 3. Compare Results
```python
def compare_results(
    baseline: Dict[str, Any],
    current: Dict[str, Any]
) -> Dict[str, float]:
    """Compare benchmark results."""
    comparison = {}
    
    # Calculate changes
    for metric in ['latency', 'memory', 'throughput']:
        comparison[metric] = {}
        for key in baseline[metric]:
            if key in current[metric]:
                change = (current[metric][key] - baseline[metric][key]) / baseline[metric][key] * 100
                comparison[metric][key] = change
    
    # Generate report
    report = ["# Performance Changes\n"]
    for metric, changes in comparison.items():
        report.append(f"\n## {metric.title()}\n")
        for key, change in changes.items():
            direction = "improvement" if change <= 0 else "regression"
            report.append(f"- {key}: {abs(change):.1f}% {direction}\n")
    
    # Save report
    with open("results/comparison.md", 'w') as f:
        f.write('\n'.join(report))
    
    return comparison
``` 