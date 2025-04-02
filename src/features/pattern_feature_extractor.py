import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .patterns.smart_money_concepts import SmartMoneyAnalyzer
from .patterns.geometric_patterns import GeometricPatternDetector, PatternType
from .technical.base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

class PatternFeatureExtractor(BaseTorchIndicator):
    def __init__(
        self,
        timeframes: List[str] = ["1h", "4h", "1d"],
        use_smc: bool = True,
        use_geometric: bool = True,
        torch_config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize Pattern Feature Extractor with PyTorch backend
        
        Args:
            timeframes: List of timeframes to analyze
            use_smc: Whether to use Smart Money Concepts
            use_geometric: Whether to use Geometric Patterns
            torch_config: PyTorch configuration for GPU/CPU
        """
        super().__init__(torch_config)
        self.timeframes = timeframes
        self.use_smc = use_smc
        self.use_geometric = use_geometric
        
        # Initialize analyzers with PyTorch config
        self.smc_analyzers = {
            tf: SmartMoneyAnalyzer(torch_config=torch_config) for tf in timeframes
        } if use_smc else {}
        
        self.geometric_detectors = {
            tf: GeometricPatternDetector(torch_config=torch_config) for tf in timeframes
        } if use_geometric else {}
        
    def extract_features(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Extract all pattern features using PyTorch
        
        Args:
            df: OHLCV DataFrame
            timeframe: Current timeframe being analyzed
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Convert data to tensors
        ohlcv_tensors = {
            'open': self.to_tensor(df['open']),
            'high': self.to_tensor(df['high']),
            'low': self.to_tensor(df['low']),
            'close': self.to_tensor(df['close']),
            'volume': self.to_tensor(df['volume']) if 'volume' in df else None
        }
        
        # Extract Smart Money Concepts features with PyTorch
        if self.use_smc and timeframe in self.smc_analyzers:
            smc = self.smc_analyzers[timeframe]
            
            # Get SMC components using tensors
            order_blocks = smc.identify_order_blocks(ohlcv_tensors)
            fair_value_gaps = smc.detect_fair_value_gaps(ohlcv_tensors)
            structure_points = smc.detect_structure_points(ohlcv_tensors)
            
            # Calculate SMC features efficiently
            active_obs = torch.tensor([ob.is_valid for ob in order_blocks], device=self.device)
            valid_fvgs = torch.tensor([fvg.is_valid for fvg in fair_value_gaps], device=self.device)
            bos_points = torch.tensor([sp.type == "BOS" for sp in structure_points], device=self.device)
            choch_points = torch.tensor([sp.type == "CHoCH" for sp in structure_points], device=self.device)
            
            features.update({
                f"{timeframe}_active_ob_count": int(torch.sum(active_obs).item()),
                f"{timeframe}_recent_fvg_count": int(torch.sum(valid_fvgs).item()),
                f"{timeframe}_bos_count": int(torch.sum(bos_points).item()),
                f"{timeframe}_choch_count": int(torch.sum(choch_points).item())
            })
            
            # Add market context with PyTorch calculations
            context = smc.analyze_market_context(ohlcv_tensors)
            features.update({
                f"{timeframe}_{k}": float(v) if isinstance(v, torch.Tensor) else v 
                for k, v in context.items()
            })
            
            # Add ML-specific features
            features.update(self.calculate_ml_features(ohlcv_tensors, order_blocks, fair_value_gaps))
        
        # Extract Geometric Pattern features with PyTorch
        if self.use_geometric and timeframe in self.geometric_detectors:
            detector = self.geometric_detectors[timeframe]
            patterns = detector.detect_all_patterns(ohlcv_tensors)
            
            # Calculate pattern features using tensor operations
            pattern_tensors = {
                pattern_type: torch.tensor([
                    1 if p.pattern_type == pattern_type else 0 
                    for p in patterns
                ], device=self.device)
                for pattern_type in PatternType
            }
            
            features.update({
                f"{timeframe}_{k.value}_count": int(torch.sum(v).item())
                for k, v in pattern_tensors.items()
            })
            
            # Calculate confidence metrics with PyTorch
            if patterns:
                confidences = torch.tensor([p.confidence for p in patterns], device=self.device)
                volume_confirms = torch.tensor([p.volume_confirmation for p in patterns], device=self.device)
                
                features.update({
                    f"{timeframe}_avg_pattern_confidence": float(torch.mean(confidences).item()),
                    f"{timeframe}_volume_confirmed_patterns": int(torch.sum(volume_confirms).item())
                })
                
                # Add ML-specific pattern features
                features.update(self.calculate_pattern_ml_features(patterns, ohlcv_tensors))
        
        return features
    
    def calculate_ml_features(
        self, 
        ohlcv: Dict[str, torch.Tensor],
        order_blocks: List,
        fair_value_gaps: List
    ) -> Dict[str, float]:
        """Calculate ML-specific features for SMC analysis"""
        features = {}
        
        with torch.no_grad():
            # Calculate price momentum
            close = ohlcv['close']
            momentum = (close[-1] - close[0]) / close[0]
            
            # Calculate volatility
            high, low = ohlcv['high'], ohlcv['low']
            volatility = torch.mean((high - low) / low)
            
            # Calculate order block metrics
            if order_blocks:
                ob_sizes = torch.tensor([ob.size for ob in order_blocks], device=self.device)
                ob_ages = torch.tensor([ob.age for ob in order_blocks], device=self.device)
                
                features.update({
                    'avg_ob_size': float(torch.mean(ob_sizes).item()),
                    'max_ob_age': float(torch.max(ob_ages).item()),
                    'active_ob_ratio': len([ob for ob in order_blocks if ob.is_valid]) / len(order_blocks)
                })
            
            # Calculate FVG metrics
            if fair_value_gaps:
                fvg_sizes = torch.tensor([fvg.size for fvg in fair_value_gaps], device=self.device)
                fvg_ages = torch.tensor([fvg.age for fvg in fair_value_gaps], device=self.device)
                
                features.update({
                    'avg_fvg_size': float(torch.mean(fvg_sizes).item()),
                    'max_fvg_age': float(torch.max(fvg_ages).item()),
                    'active_fvg_ratio': len([fvg for fvg in fair_value_gaps if fvg.is_valid]) / len(fair_value_gaps)
                })
            
            # Add momentum and volatility
            features.update({
                'price_momentum': float(momentum.item()),
                'price_volatility': float(volatility.item())
            })
        
        return features
    
    def calculate_pattern_ml_features(
        self,
        patterns: List,
        ohlcv: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate ML-specific features for pattern analysis"""
        features = {}
        
        with torch.no_grad():
            if patterns:
                # Convert pattern properties to tensors
                sizes = torch.tensor([p.size for p in patterns], device=self.device)
                durations = torch.tensor([p.duration for p in patterns], device=self.device)
                confidences = torch.tensor([p.confidence for p in patterns], device=self.device)
                
                # Calculate pattern metrics
                features.update({
                    'avg_pattern_size': float(torch.mean(sizes).item()),
                    'avg_pattern_duration': float(torch.mean(durations).item()),
                    'pattern_confidence_std': float(torch.std(confidences).item())
                })
                
                # Calculate pattern density
                total_bars = len(ohlcv['close'])
                pattern_density = len(patterns) / total_bars
                features['pattern_density'] = pattern_density
                
                # Calculate pattern completion rate
                completed = torch.tensor([p.is_completed for p in patterns], device=self.device)
                completion_rate = float(torch.mean(completed.float()).item())
                features['pattern_completion_rate'] = completion_rate
        
        return features
    
    def extract_all_timeframe_features(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Extract features from all timeframes using PyTorch
        
        Args:
            data: Dictionary of DataFrames for each timeframe
            
        Returns:
            Combined features from all timeframes
        """
        all_features = {}
        
        for timeframe in self.timeframes:
            if timeframe in data:
                features = self.extract_features(data[timeframe], timeframe)
                all_features.update(features)
                
        return all_features
    
    def get_training_features(self, 
                            data: Dict[str, pd.DataFrame],
                            window_size: int = 100
                            ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate training features for ML model using PyTorch
        
        Args:
            data: Dictionary of DataFrames for each timeframe
            window_size: Number of bars to look back
            
        Returns:
            Feature tensor and feature names
        """
        feature_tensors = []
        feature_names = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe not in data:
                continue
                
            df = data[timeframe]
            all_features = []
            
            # Extract features for each window efficiently
            for i in range(window_size, len(df)):
                window_df = df.iloc[i-window_size:i]
                features = self.extract_features(window_df, timeframe)
                all_features.append(list(features.values()))
                
                # Store feature names once
                if not feature_names:
                    feature_names = list(features.keys())
            
            # Convert to tensor
            if all_features:
                feature_tensors.append(
                    torch.tensor(all_features, dtype=torch.float32, device=self.device)
                )
        
        # Combine features from all timeframes
        if feature_tensors:
            combined_features = torch.cat(feature_tensors, dim=1)
            return combined_features, feature_names
        
        return torch.tensor([], device=self.device), [] 