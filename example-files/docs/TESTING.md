# 🧪 Testing Guide

This guide outlines testing procedures and guidelines for the GPU-accelerated trading system.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Test Types](#test-types)
3. [Writing Tests](#writing-tests)
4. [Running Tests](#running-tests)
5. [Performance Testing](#performance-testing)
6. [Integration Testing](#integration-testing)
7. [CI/CD Pipeline](#cicd-pipeline)

## 📈 Overview

Our testing strategy ensures:
- Code reliability
- Performance optimization
- Memory efficiency
- GPU utilization
- Feature accuracy

## 🔍 Test Types

### 1. Unit Tests
```python
# tests/features/test_rsi.py
class TestRSIFeature(unittest.TestCase):
    """Test RSI feature calculation."""
    
    def setUp(self):
        """Setup test environment."""
        self.feature = RSIFeature(window_size=14)
        self.device = torch.device('cuda')
        
    def test_basic_calculation(self):
        """Test basic RSI calculation."""
        # Generate test data
        data = torch.randn(100, 1000).to(self.device)
        
        # Calculate RSI
        result = self.feature(data)
        
        # Verify shape
        self.assertEqual(result.shape, data.shape)
        
        # Verify range
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 100))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test empty data
        with self.assertRaises(ValueError):
            self.feature(torch.tensor([]))
            
        # Test NaN values
        data = torch.randn(10, 10).to(self.device)
        data[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.feature(data)
            
        # Test constant values
        data = torch.ones(10, 10).to(self.device)
        result = self.feature(data)
        self.assertTrue(torch.allclose(result[14:], torch.tensor(50.0)))
```

### 2. Integration Tests
```python
# tests/integration/test_pipeline.py
def test_feature_pipeline():
    """Test complete feature pipeline."""
    # Setup pipeline
    pipeline = FeaturePipeline([
        RSIFeature(),
        MomentumFeature(),
        VolatilityFeature()
    ])
    
    # Load test data
    data = load_test_data()
    
    # Process features
    features = pipeline.process(data)
    
    # Verify output
    assert isinstance(features, dict)
    assert all(k in features for k in ['rsi', 'momentum', 'volatility'])
    assert all(not torch.isnan(v).any() for v in features.values())
    
def test_signal_generation():
    """Test signal generation pipeline."""
    # Setup components
    features = FeaturePipeline()
    model = TradingModel()
    signals = SignalGenerator()
    
    # Process data
    data = load_test_data()
    feature_data = features.process(data)
    predictions = model.predict(feature_data)
    trading_signals = signals.generate(predictions)
    
    # Verify signals
    assert trading_signals.shape == (len(data), 3)  # Buy, Sell, Hold
    assert torch.allclose(trading_signals.sum(dim=1), torch.ones(len(data)))
```

### 3. Performance Tests
```python
# tests/performance/test_features.py
def test_feature_performance():
    """Test feature calculation performance."""
    feature = RSIFeature()
    results = {}
    
    # Test different batch sizes
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        data = generate_test_data(batch_size)
        
        # Warmup
        for _ in range(10):
            feature(data)
        torch.cuda.synchronize()
        
        # Measure performance
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()
        
        for _ in range(100):
            feature(data)
        torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        # Record metrics
        results[batch_size] = {
            'latency': (end_time - start_time) / 100,
            'memory': (end_memory - start_memory) / 1024**2,
            'throughput': batch_size * 100 / (end_time - start_time)
        }
    
    return results

def test_memory_efficiency():
    """Test memory usage patterns."""
    feature = RSIFeature()
    
    # Track memory allocations
    torch.cuda.reset_peak_memory_stats()
    data = generate_test_data(1024)
    
    # Process data
    feature(data)
    torch.cuda.synchronize()
    
    # Check memory usage
    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    assert max_memory < 1024, f"Memory usage {max_memory:.0f}MB exceeds 1GB limit"
```

## ✍️ Writing Tests

### 1. Test Structure
```python
# tests/test_template.py
class TestFeature(unittest.TestCase):
    """Test feature implementation."""
    
    def setUp(self):
        """Setup test environment."""
        # Initialize components
        self.feature = Feature()
        self.data = generate_test_data()
        
    def tearDown(self):
        """Cleanup after tests."""
        # Free GPU memory
        torch.cuda.empty_cache()
        
    def test_functionality(self):
        """Test core functionality."""
        # Arrange
        expected = calculate_expected_result()
        
        # Act
        result = self.feature(self.data)
        
        # Assert
        self.assertTrue(torch.allclose(result, expected))
```

### 2. Test Coverage
```python
# Run tests with coverage
pytest --cov=src tests/ --cov-report=html

# Coverage requirements
COVERAGE_TARGETS = {
    'features': 90,
    'models': 85,
    'utils': 80,
    'total': 85
}

def verify_coverage(coverage_data: Dict[str, float]):
    """Verify coverage meets targets."""
    for module, target in COVERAGE_TARGETS.items():
        actual = coverage_data.get(module, 0)
        assert actual >= target, (
            f"{module} coverage {actual:.1f}% "
            f"below target {target}%"
        )
```

### 3. Test Data
```python
# tests/utils/test_data.py
def generate_test_data(
    size: int = 1000,
    features: int = 10
) -> torch.Tensor:
    """Generate test data.
    
    Args:
        size: Number of samples
        features: Number of features
        
    Returns:
        Test data tensor
    """
    return torch.randn(size, features).cuda()

def load_test_dataset() -> Dataset:
    """Load test dataset."""
    return TradingDataset(
        symbols=['BTC/USDT'],
        start_date='2024-01-01',
        end_date='2024-02-01',
        timeframe='5m'
    )
```

## 🚀 Running Tests

### 1. Test Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/features/test_rsi.py

# Run tests with specific marker
pytest -m "gpu"

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/
```

### 2. Test Configuration
```python
# pytest.ini
[pytest]
markers =
    gpu: tests requiring GPU
    slow: slow running tests
    integration: integration tests
testpaths = tests
python_files = test_*.py
addopts = --verbose

# Test settings
TEST_CONFIG = {
    'batch_sizes': [32, 64, 128, 256, 512, 1024],
    'test_iterations': 100,
    'warmup_iterations': 10,
    'memory_limit': 4 * 1024,  # 4GB
    'max_latency': 0.1        # 100ms
}
```

## 📊 Performance Testing

### 1. Latency Tests
```python
def test_latency():
    """Test processing latency."""
    feature = Feature()
    data = generate_test_data()
    
    # Warmup
    for _ in range(10):
        feature(data)
    torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    for _ in range(100):
        start = time.time()
        feature(data)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)
    
    # Analyze results
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Verify requirements
    assert avg_latency < 0.001, f"Average latency {avg_latency:.6f}s exceeds 1ms limit"
    assert p99_latency < 0.002, f"P99 latency {p99_latency:.6f}s exceeds 2ms limit"
```

### 2. Memory Tests
```python
def test_memory():
    """Test memory usage."""
    feature = Feature()
    
    # Track allocations
    torch.cuda.reset_peak_memory_stats()
    data = generate_test_data(1024)
    
    # Process data
    feature(data)
    torch.cuda.synchronize()
    
    # Check usage
    memory_used = torch.cuda.max_memory_allocated() / 1024**2
    assert memory_used < 1024, f"Memory usage {memory_used:.0f}MB exceeds 1GB limit"
```

### 3. Throughput Tests
```python
def test_throughput():
    """Test processing throughput."""
    feature = Feature()
    results = {}
    
    # Test batch sizes
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        data = generate_test_data(batch_size)
        samples_processed = 0
        start_time = time.time()
        
        # Process batches
        while samples_processed < 1_000_000:
            feature(data)
            samples_processed += batch_size
            
        # Calculate throughput
        duration = time.time() - start_time
        throughput = samples_processed / duration
        
        # Record results
        results[batch_size] = {
            'throughput': throughput,
            'latency': duration / (samples_processed / batch_size)
        }
        
        # Verify requirements
        assert throughput > 10000, f"Throughput {throughput:.0f} samples/sec below target"
```

## 🔄 Integration Testing

### 1. Pipeline Tests
```python
def test_pipeline():
    """Test complete trading pipeline."""
    # Setup components
    features = FeaturePipeline()
    model = TradingModel()
    signals = SignalGenerator()
    
    # Load data
    data = load_test_data()
    
    # Process pipeline
    feature_data = features.process(data)
    predictions = model.predict(feature_data)
    trading_signals = signals.generate(predictions)
    
    # Verify results
    verify_feature_data(feature_data)
    verify_predictions(predictions)
    verify_signals(trading_signals)
```

### 2. System Tests
```python
def test_system():
    """Test complete trading system."""
    # Initialize system
    system = TradingSystem(
        exchange='binance',
        symbols=['BTC/USDT'],
        timeframe='5m'
    )
    
    # Run simulation
    results = system.backtest(
        start_date='2024-01-01',
        end_date='2024-02-01'
    )
    
    # Verify metrics
    assert results['total_trades'] > 0
    assert results['win_rate'] > 0.5
    assert results['sharpe_ratio'] > 1.0
    assert results['max_drawdown'] < 0.1
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run tests
      run: |
        pytest tests/
        pytest --cov=src --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### 2. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 23.3.0
  hooks:
    - id: black
      language_version: python3.10

- repo: https://github.com/pycqa/flake8
  rev: 6.0.0
  hooks:
    - id: flake8
```

### 3. Test Reports
```python
def generate_test_report(results: Dict[str, Any]):
    """Generate test report."""
    report = []
    
    # Summary
    report.append("# Test Results\n")
    report.append(f"Date: {datetime.now()}\n")
    
    # Test coverage
    report.append("## Coverage\n")
    for module, coverage in results['coverage'].items():
        report.append(f"- {module}: {coverage:.1f}%\n")
    
    # Performance metrics
    report.append("## Performance\n")
    for metric, value in results['performance'].items():
        report.append(f"- {metric}: {value}\n")
    
    # Save report
    with open("test_report.md", 'w') as f:
        f.write('\n'.join(report))
``` 