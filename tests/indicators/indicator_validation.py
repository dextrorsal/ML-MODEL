import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from src.features.technical.indicators.rsi import RsiIndicator
from src.features.technical.indicators.cci import CciIndicator
from src.features.technical.indicators.wave_trend import WaveTrendIndicator
from src.features.technical.indicators.chandelier_exit import ChandelierExitIndicator

class IndicatorValidator:
    """
    Validates technical indicators before ML training to ensure:
    1. No look-ahead bias
    2. Proper signal generation
    3. Reasonable sensitivity
    4. No inherent overfitting
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.results: Dict[str, Dict] = {}
        
        # Initialize indicators
        self.indicators = {
            'RSI': RsiIndicator(),
            'CCI': CciIndicator(),
            'WaveTrend': WaveTrendIndicator(),
            'ChandelierExit': ChandelierExitIndicator()
        }
        
    def validate_look_ahead(self, indicator_name: str) -> bool:
        """
        Test for look-ahead bias by comparing signals generated with
        different data window sizes.
        """
        indicator = self.indicators[indicator_name]
        
        # Test with different window sizes
        window_sizes = [100, 200, 500]
        signals = []
        
        for size in window_sizes:
            data_window = self.data.iloc[-size:]
            with torch.no_grad():
                signal = indicator(
                    torch.tensor(data_window['close'].values, dtype=torch.float32)
                )
                if isinstance(signal, torch.Tensor):
                    signal = signal.cpu().numpy()
            signals.append(signal[-1])  # Compare last signal
            
        # Check if signals are consistent
        max_diff = max(abs(s1 - s2) for s1, s2 in zip(signals[:-1], signals[1:]))
        
        self.results[indicator_name] = {
            'look_ahead_test': max_diff < 0.1,
            'signal_variance': max_diff
        }
        
        return max_diff < 0.1

    def validate_sensitivity(self, indicator_name: str) -> Tuple[float, float]:
        """
        Test indicator sensitivity to price changes to detect potential overfitting.
        """
        indicator = self.indicators[indicator_name]
        
        # Create slightly modified price data
        noise_levels = [0.0001, 0.001, 0.01]  # 0.01% to 1% noise
        base_signals = []
        noisy_signals = []
        
        for noise in noise_levels:
            # Add random noise to prices
            noisy_data = self.data.copy()
            noisy_data['close'] = noisy_data['close'] * (1 + np.random.normal(0, noise, len(self.data)))
            
            with torch.no_grad():
                base_signal = indicator(
                    torch.tensor(self.data['close'].values, dtype=torch.float32)
                )
                noisy_signal = indicator(
                    torch.tensor(noisy_data['close'].values, dtype=torch.float32)
                )
            
            base_signals.append(base_signal.numpy())
            noisy_signals.append(noisy_signal.numpy())
        
        # Calculate sensitivity scores
        correlations = [np.corrcoef(b, n)[0,1] for b, n in zip(base_signals, noisy_signals)]
        sensitivity = 1 - np.mean(correlations)
        
        self.results[indicator_name].update({
            'sensitivity_score': sensitivity,
            'noise_correlations': correlations
        })
        
        return sensitivity, correlations

    def validate_signal_distribution(self, indicator_name: str) -> Dict:
        """
        Analyze signal distribution to detect potential biases.
        """
        indicator = self.indicators[indicator_name]
        
        with torch.no_grad():
            signals = indicator(
                torch.tensor(self.data['close'].values, dtype=torch.float32)
            ).numpy()
        
        # Calculate distribution metrics
        signal_stats = {
            'mean': np.mean(signals),
            'std': np.std(signals),
            'skew': pd.Series(signals).skew(),
            'kurtosis': pd.Series(signals).kurtosis()
        }
        
        self.results[indicator_name].update({
            'signal_distribution': signal_stats
        })
        
        return signal_stats

    def plot_validation_results(self, indicator_name: str):
        """
        Plot validation results for visual inspection.
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original vs Noisy Signals
        plt.subplot(2, 1, 1)
        indicator = self.indicators[indicator_name]
        with torch.no_grad():
            original_signal = indicator(
                torch.tensor(self.data['close'].values[-500:], dtype=torch.float32)
            ).numpy()
        
        plt.plot(original_signal, label='Original Signal')
        plt.title(f'{indicator_name} Signal Validation')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Signal Distribution
        plt.subplot(2, 1, 2)
        plt.hist(original_signal, bins=50, density=True)
        plt.title('Signal Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_full_validation(self, indicator_name: str) -> Dict:
        """
        Run all validation tests for an indicator.
        """
        print(f"\nValidating {indicator_name}...")
        
        # Run all tests
        look_ahead_passed = self.validate_look_ahead(indicator_name)
        sensitivity, correlations = self.validate_sensitivity(indicator_name)
        dist_stats = self.validate_signal_distribution(indicator_name)
        
        # Compile results
        validation_summary = {
            'look_ahead_test_passed': look_ahead_passed,
            'sensitivity_score': sensitivity,
            'noise_correlations': correlations,
            'distribution_stats': dist_stats,
            'potential_issues': []
        }
        
        # Check for potential issues
        if sensitivity > 0.3:
            validation_summary['potential_issues'].append(
                'High sensitivity to noise - might be overfitting'
            )
        if abs(dist_stats['skew']) > 1:
            validation_summary['potential_issues'].append(
                'Significant signal bias detected'
            )
        if not look_ahead_passed:
            validation_summary['potential_issues'].append(
                'Possible look-ahead bias detected'
            )
            
        return validation_summary

def test_indicators(data_path: str):
    """
    Run validation tests on all indicators.
    
    Args:
        data_path: Path to OHLCV data file
    """
    # Load data
    data = pd.read_csv(data_path)
    validator = IndicatorValidator(data)
    
    # Test each indicator
    for indicator_name in validator.indicators.keys():
        results = validator.run_full_validation(indicator_name)
        print(f"\n{indicator_name} Validation Results:")
        print("=" * 50)
        print(f"Look-ahead test passed: {results['look_ahead_test_passed']}")
        print(f"Sensitivity score: {results['sensitivity_score']:.4f}")
        print("\nPotential Issues:")
        for issue in results['potential_issues']:
            print(f"- {issue}")
        
        # Plot results
        validator.plot_validation_results(indicator_name)

if __name__ == "__main__":
    # Example usage
    data_path = "path/to/your/price_data.csv"
    test_indicators(data_path) 