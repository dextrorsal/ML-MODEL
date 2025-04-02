import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

class Trend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'high' or 'low'
    confirmed: bool = False

class MarketStructureAnalyzer:
    def __init__(self, 
                 swing_length: int = 5,
                 choch_threshold: float = 0.001,  # 0.1% threshold for CHoCH
                 volume_factor: float = 1.5):     # Volume threshold for confirmation
        """
        Initialize Market Structure Analyzer
        
        Args:
            swing_length (int): Lookback period for swing point detection
            choch_threshold (float): Minimum price change % to confirm CHoCH
            volume_factor (float): Volume increase factor for confirmation
        """
        self.swing_length = swing_length
        self.choch_threshold = choch_threshold
        self.volume_factor = volume_factor
        
    def identify_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Identify swing highs and lows in price action
        
        Args:
            df (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            List[SwingPoint]: List of identified swing points
        """
        swing_points = []
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            # Get price window
            window = df.iloc[i-self.swing_length:i+self.swing_length+1]
            
            # Check for swing high
            if df['high'].iloc[i] == window['high'].max():
                # Confirm with volume
                if df['volume'].iloc[i] > df['volume'].iloc[i-1] * self.volume_factor:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=df['high'].iloc[i],
                            type='high',
                            confirmed=True
                        )
                    )
                else:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=df['high'].iloc[i],
                            type='high',
                            confirmed=False
                        )
                    )
            
            # Check for swing low
            if df['low'].iloc[i] == window['low'].min():
                if df['volume'].iloc[i] > df['volume'].iloc[i-1] * self.volume_factor:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=df['low'].iloc[i],
                            type='low',
                            confirmed=True
                        )
                    )
                else:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=df['low'].iloc[i],
                            type='low',
                            confirmed=False
                        )
                    )
        
        return swing_points
    
    def detect_choch(self, 
                    df: pd.DataFrame, 
                    swing_points: List[SwingPoint]
                    ) -> pd.Series:
        """
        Detect Change of Character (CHoCH) patterns
        
        Args:
            df (pd.DataFrame): Price data
            swing_points (List[SwingPoint]): Identified swing points
            
        Returns:
            pd.Series: Boolean series indicating CHoCH patterns
        """
        choch_signals = pd.Series(False, index=df.index)
        
        for i in range(3, len(swing_points)):
            current = swing_points[i]
            prev1 = swing_points[i-1]
            prev2 = swing_points[i-2]
            prev3 = swing_points[i-3]
            
            # Check for bullish CHoCH
            if (current.type == 'high' and prev2.type == 'high' and
                prev1.type == 'low' and prev3.type == 'low' and
                current.price > prev2.price and
                prev1.price > prev3.price and
                current.confirmed):
                
                # Calculate price change percentage
                price_change = (current.price - prev2.price) / prev2.price
                if price_change > self.choch_threshold:
                    choch_signals.iloc[current.index] = True
            
            # Check for bearish CHoCH
            if (current.type == 'low' and prev2.type == 'low' and
                prev1.type == 'high' and prev3.type == 'high' and
                current.price < prev2.price and
                prev1.price < prev3.price and
                current.confirmed):
                
                price_change = (prev2.price - current.price) / prev2.price
                if price_change > self.choch_threshold:
                    choch_signals.iloc[current.index] = True
        
        return choch_signals
    
    def detect_bos(self,
                  df: pd.DataFrame,
                  swing_points: List[SwingPoint]
                  ) -> pd.Series:
        """
        Detect Break of Structure (BOS) patterns
        
        Args:
            df (pd.DataFrame): Price data
            swing_points (List[SwingPoint]): Identified swing points
            
        Returns:
            pd.Series: 1 for bullish BOS, -1 for bearish BOS, 0 for no BOS
        """
        bos_signals = pd.Series(0, index=df.index)
        
        for i in range(2, len(swing_points)):
            current = swing_points[i]
            prev1 = swing_points[i-1]
            prev2 = swing_points[i-2]
            
            # Bullish BOS
            if (current.type == 'high' and prev1.type == 'low' and
                prev2.type == 'high' and
                current.price > prev2.price and
                current.confirmed):
                bos_signals.iloc[current.index] = 1
            
            # Bearish BOS
            if (current.type == 'low' and prev1.type == 'high' and
                prev2.type == 'low' and
                current.price < prev2.price and
                current.confirmed):
                bos_signals.iloc[current.index] = -1
        
        return bos_signals
    
    def get_market_structure(self, 
                           df: pd.DataFrame, 
                           lookback_periods: int = 20
                           ) -> Trend:
        """
        Determine overall market structure based on swing points and BOS
        
        Args:
            df (pd.DataFrame): Price data
            lookback_periods (int): Number of periods to analyze
            
        Returns:
            Trend: Current market structure trend
        """
        swing_points = self.identify_swing_points(df.iloc[-lookback_periods:])
        if len(swing_points) < 4:
            return Trend.SIDEWAYS
            
        # Get last 4 swing points
        last_swings = swing_points[-4:]
        
        # Count higher highs and lower lows
        higher_highs = 0
        lower_lows = 0
        
        for i in range(1, len(last_swings)):
            if (last_swings[i].type == 'high' and 
                last_swings[i-1].type == 'high' and
                last_swings[i].price > last_swings[i-1].price):
                higher_highs += 1
                
            if (last_swings[i].type == 'low' and 
                last_swings[i-1].type == 'low' and
                last_swings[i].price < last_swings[i-1].price):
                lower_lows += 1
        
        if higher_highs >= 2:
            return Trend.UPTREND
        elif lower_lows >= 2:
            return Trend.DOWNTREND
        else:
            return Trend.SIDEWAYS 