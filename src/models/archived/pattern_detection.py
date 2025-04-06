import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class PatternDetector:
    def __init__(self, min_swing_size: float = 0.02):
        """
        Initialize pattern detector with minimum swing size (2% default)
        """
        self.min_swing_size = min_swing_size

    def detect_bullish_fractal(self, data: pd.DataFrame, center_idx: int) -> bool:
        """
        Detect bullish (bottom) fractal pattern:
        Lower low with 2 higher lows on each side
        """
        if center_idx < 2 or center_idx >= len(data) - 2:
            return False
        
        window = data['low'].iloc[center_idx-2:center_idx+3]
        center_low = window.iloc[2]
        
        return (window.iloc[0] > center_low and
                window.iloc[1] > center_low and
                window.iloc[3] > center_low and
                window.iloc[4] > center_low)

    def detect_bearish_fractal(self, data: pd.DataFrame, center_idx: int) -> bool:
        """
        Detect bearish (top) fractal pattern:
        Higher high with 2 lower highs on each side
        """
        if center_idx < 2 or center_idx >= len(data) - 2:
            return False
        
        window = data['high'].iloc[center_idx-2:center_idx+3]
        center_high = window.iloc[2]
        
        return (window.iloc[0] < center_high and
                window.iloc[1] < center_high and
                window.iloc[3] < center_high and
                window.iloc[4] < center_high)

    def find_zigzag_points(self, data: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """
        Identify zigzag pivot points using swing highs and lows
        Returns: (pivot_indices, pivot_values)
        """
        highs = data['high'].values
        lows = data['low'].values
        close = data['close'].values
        
        pivots = []
        pivot_values = []
        trend = 0  # 0=searching, 1=up, -1=down
        last_pivot_idx = 0
        last_pivot_value = close[0]
        
        for i in range(1, len(data)-1):
            # Potential swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_size = (highs[i] - last_pivot_value) / last_pivot_value
                
                if trend != 1 and abs(swing_size) >= self.min_swing_size:
                    pivots.append(i)
                    pivot_values.append(highs[i])
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_value = highs[i]
            
            # Potential swing low
            elif lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_size = (lows[i] - last_pivot_value) / last_pivot_value
                
                if trend != -1 and abs(swing_size) >= self.min_swing_size:
                    pivots.append(i)
                    pivot_values.append(lows[i])
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_value = lows[i]
        
        return pivots, pivot_values

    def detect_zigzag_pattern(self, data: pd.DataFrame, window_size: int = 10) -> List[Dict]:
        """
        Detect zigzag patterns and classify their shapes
        Returns list of pattern dictionaries with type and strength
        """
        pivots, values = self.find_zigzag_points(data)
        patterns = []
        
        for i in range(len(pivots)-3):  # Need at least 4 points for a pattern
            p1, p2, p3, p4 = pivots[i:i+4]
            v1, v2, v3, v4 = values[i:i+4]
            
            # Calculate swing sizes
            swing1 = (v2 - v1) / v1
            swing2 = (v3 - v2) / v2
            swing3 = (v4 - v3) / v3
            
            # Pattern strength based on swing consistency
            strength = min(abs(swing1), abs(swing2), abs(swing3))
            
            # Detect common patterns
            if v1 > v2 and v2 < v3 and v3 > v4:  # M top
                patterns.append({
                    'type': 'M_top',
                    'start_idx': p1,
                    'end_idx': p4,
                    'strength': strength,
                    'pivot_points': [p1, p2, p3, p4],
                    'pivot_values': [v1, v2, v3, v4]
                })
            
            elif v1 < v2 and v2 > v3 and v3 < v4:  # W bottom
                patterns.append({
                    'type': 'W_bottom',
                    'start_idx': p1,
                    'end_idx': p4,
                    'strength': strength,
                    'pivot_points': [p1, p2, p3, p4],
                    'pivot_values': [v1, v2, v3, v4]
                })
            
            # Add more pattern types here (triangles, flags, etc.)
        
        return patterns

    def label_training_data(self, data: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
        """
        Create training labels for the neural network
        """
        df = data.copy()
        
        # Initialize pattern columns
        df['bullish_fractal'] = False
        df['bearish_fractal'] = False
        df['zigzag_pattern'] = ''
        df['pattern_strength'] = 0.0
        
        # Detect fractals
        for i in range(2, len(df)-2):
            df.loc[i, 'bullish_fractal'] = self.detect_bullish_fractal(df, i)
            df.loc[i, 'bearish_fractal'] = self.detect_bearish_fractal(df, i)
        
        # Detect zigzag patterns
        patterns = self.detect_zigzag_pattern(df, window_size)
        for pattern in patterns:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            df.loc[start_idx:end_idx, 'zigzag_pattern'] = pattern['type']
            df.loc[start_idx:end_idx, 'pattern_strength'] = pattern['strength']
        
        return df

    def prepare_pattern_features(self, data: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for pattern recognition model
        """
        # Label the data
        labeled_data = self.label_training_data(data, window_size)
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(len(labeled_data) - window_size):
            # Get the sequence window
            sequence = labeled_data.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
            
            # Create pattern label vector [bullish_fractal, bearish_fractal, w_bottom, m_top]
            pattern_window = labeled_data.iloc[i+window_size-5:i+window_size]
            label = [
                pattern_window['bullish_fractal'].any(),
                pattern_window['bearish_fractal'].any(),
                (pattern_window['zigzag_pattern'] == 'W_bottom').any(),
                (pattern_window['zigzag_pattern'] == 'M_top').any()
            ]
            
            sequences.append(sequence)
            labels.append(label)
        
        return np.array(sequences), np.array(labels) 