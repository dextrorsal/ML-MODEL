import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

class PatternType(Enum):
    ASCENDING_CHANNEL = "ascending_channel"
    DESCENDING_CHANNEL = "descending_channel"
    RANGING_CHANNEL = "ranging_channel"
    RISING_WEDGE_EXPANDING = "rising_wedge_expanding"
    FALLING_WEDGE_EXPANDING = "falling_wedge_expanding"
    RISING_WEDGE_CONTRACTING = "rising_wedge_contracting"
    FALLING_WEDGE_CONTRACTING = "falling_wedge_contracting"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    CONVERGING_TRIANGLE = "converging_triangle"
    DIVERGING_TRIANGLE = "diverging_triangle"

@dataclass
class ZigZagPoint:
    index: int
    price: float
    direction: int  # 1 for up, -1 for down
    pivot_type: str  # "high" or "low"
    timestamp: pd.Timestamp

@dataclass
class GeometricPattern:
    pattern_type: PatternType
    start_index: int
    end_index: int
    upper_points: List[ZigZagPoint]
    lower_points: List[ZigZagPoint]
    confidence: float
    volume_confirmation: bool
    creation_time: pd.Timestamp = None

class GeometricPatternDetector:
    def __init__(
        self,
        zigzag_length: int = 8,
        zigzag_depth: int = 55,
        error_threshold: float = 0.2,
        flat_threshold: float = 0.2,
        min_pattern_bars: int = 10,
        volume_confirmation_threshold: float = 1.5
    ):
        """
        Initialize Geometric Pattern Detector
        
        Args:
            zigzag_length: Length parameter for zigzag calculation
            zigzag_depth: Depth parameter for zigzag calculation
            error_threshold: Maximum allowed error for trendline fitting (0-1)
            flat_threshold: Threshold to determine if a line is flat (0-1)
            min_pattern_bars: Minimum number of bars for valid pattern
            volume_confirmation_threshold: Volume threshold for pattern confirmation
        """
        self.zigzag_length = zigzag_length
        self.zigzag_depth = zigzag_depth
        self.error_threshold = error_threshold
        self.flat_threshold = flat_threshold
        self.min_pattern_bars = min_pattern_bars
        self.volume_threshold = volume_confirmation_threshold
        
    def calculate_zigzag(self, df: pd.DataFrame) -> List[ZigZagPoint]:
        """Calculate ZigZag points using the Trendscope method"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Initialize arrays for zigzag calculation
        zigzag_points = []
        current_dir = 1  # Start looking for high
        last_price = lows[0]
        last_idx = 0
        
        for i in range(1, len(df)):
            if current_dir == 1:  # Looking for high
                if highs[i] > last_price:
                    last_price = highs[i]
                    last_idx = i
                elif lows[i] < last_price - (last_price * self.zigzag_depth / 100):
                    # Confirmed high point
                    zigzag_points.append(
                        ZigZagPoint(
                            index=last_idx,
                            price=last_price,
                            direction=1,
                            pivot_type="high",
                            timestamp=df.index[last_idx]
                        )
                    )
                    current_dir = -1
                    last_price = lows[i]
                    last_idx = i
            else:  # Looking for low
                if lows[i] < last_price:
                    last_price = lows[i]
                    last_idx = i
                elif highs[i] > last_price + (last_price * self.zigzag_depth / 100):
                    # Confirmed low point
                    zigzag_points.append(
                        ZigZagPoint(
                            index=last_idx,
                            price=last_price,
                            direction=-1,
                            pivot_type="low",
                            timestamp=df.index[last_idx]
                        )
                    )
                    current_dir = 1
                    last_price = highs[i]
                    last_idx = i
                    
        return zigzag_points
    
    def fit_trendline(self, points: List[ZigZagPoint]) -> Tuple[float, float, float]:
        """
        Fit a trendline to a set of points
        Returns: slope, intercept, error
        """
        if len(points) < 2:
            return 0, 0, float('inf')
            
        x = np.array([p.index for p in points])
        y = np.array([p.price for p in points])
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate error
        y_pred = slope * x + intercept
        error = np.mean(np.abs(y - y_pred) / y)
        
        return slope, intercept, error
    
    def detect_channels(self, df: pd.DataFrame) -> List[GeometricPattern]:
        """Detect ascending, descending and ranging channels"""
        patterns = []
        zigzag_points = self.calculate_zigzag(df)
        
        # Need at least 4 points to form a channel
        if len(zigzag_points) < 4:
            return patterns
            
        # Group points into upper and lower trend lines
        for i in range(len(zigzag_points) - 3):
            upper_points = []
            lower_points = []
            
            # Get 4 consecutive points
            window = zigzag_points[i:i+4]
            
            # Separate into upper and lower points
            for point in window:
                if point.direction == 1:  # High point
                    upper_points.append(point)
                else:  # Low point
                    lower_points.append(point)
                    
            if len(upper_points) < 2 or len(lower_points) < 2:
                continue
                
            # Fit trend lines
            upper_slope, _, upper_error = self.fit_trendline(upper_points)
            lower_slope, _, lower_error = self.fit_trendline(lower_points)
            
            # Check if slopes are similar (parallel lines)
            slope_diff = abs(upper_slope - lower_slope)
            
            if slope_diff < self.error_threshold:
                # Determine channel type
                if abs(upper_slope) < self.flat_threshold:
                    pattern_type = PatternType.RANGING_CHANNEL
                elif upper_slope > 0:
                    pattern_type = PatternType.ASCENDING_CHANNEL
                else:
                    pattern_type = PatternType.DESCENDING_CHANNEL
                    
                # Calculate pattern confidence based on errors
                confidence = 1 - (upper_error + lower_error) / 2
                
                # Check volume confirmation
                volume_sma = df['volume'].rolling(20).mean()
                volume_confirmation = df['volume'].iloc[window[-1].index] > volume_sma.iloc[window[-1].index] * self.volume_threshold
                
                patterns.append(
                    GeometricPattern(
                        pattern_type=pattern_type,
                        start_index=window[0].index,
                        end_index=window[-1].index,
                        upper_points=upper_points,
                        lower_points=lower_points,
                        confidence=confidence,
                        volume_confirmation=volume_confirmation,
                        creation_time=df.index[window[-1].index]
                    )
                )
                
        return patterns
    
    def detect_wedges(self, df: pd.DataFrame) -> List[GeometricPattern]:
        """Detect expanding and contracting wedges"""
        # Implementation similar to channels but checking for converging/diverging lines
        pass
    
    def detect_triangles(self, df: pd.DataFrame) -> List[GeometricPattern]:
        """Detect various triangle patterns"""
        # Implementation similar to channels but with specific triangle criteria
        pass
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[GeometricPattern]:
        """Detect all geometric patterns in the data"""
        patterns = []
        patterns.extend(self.detect_channels(df))
        patterns.extend(self.detect_wedges(df))
        patterns.extend(self.detect_triangles(df))
        return patterns 