import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
from .market_structure import MarketStructureAnalyzer, SwingPoint, Trend

# Constants
class Bias(Enum):
    BULLISH = 1
    BEARISH = -1

class Leg(Enum):
    BULLISH = 1
    BEARISH = 0

class StructureType(Enum):
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHoCH"  # Change of Character

class OrderBlockType(Enum):
    INTERNAL = "internal"
    SWING = "swing"

@dataclass
class OrderBlock:
    index: int
    price_high: float
    price_low: float
    type: str  # 'bullish' or 'bearish'
    volume: float
    atr_multiple: float = 1.0
    is_valid: bool = True
    is_internal: bool = True  # True for internal OB, False for swing OB
    mitigation_type: str = "high_low"  # 'close' or 'high_low'
    creation_time: pd.Timestamp = None
    box_extend: int = 20  # Number of bars to extend the box

@dataclass
class FairValueGap:
    index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    size: float  # Gap size
    is_valid: bool = True
    creation_time: pd.Timestamp = None
    extend_to: int = 1  # Number of bars to extend

@dataclass
class EqualLevel:
    index: int
    price: float
    type: str  # 'high' or 'low'
    confirmation_bars: int = 3
    threshold: float = 0.1  # Sensitivity threshold (0-0.5)
    creation_time: pd.Timestamp = None

@dataclass
class StructurePoint:
    index: int
    price: float
    type: StructureType  # BOS or CHoCH
    level: str  # 'internal' or 'swing'
    direction: Bias  # BULLISH or BEARISH
    creation_time: pd.Timestamp = None
    leg_type: Leg = None

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'HH', 'LL', 'LH', 'HL'
    creation_time: pd.Timestamp = None
    strength: str = "strong"  # 'strong' or 'weak'

class SmartMoneyAnalyzer:
    def __init__(self,
                 atr_period: int = 14,
                 volume_threshold: float = 1.5,
                 ob_invalidation_atr: float = 1.0,
                 fvg_auto_threshold: bool = True,
                 structure_confluence_filter: bool = True,
                 swing_length: int = 50,
                 equal_length: int = 3,
                 mode: str = "historical"):  # 'historical' or 'present'
        """
        Initialize Smart Money Analyzer
        
        Args:
            atr_period (int): Period for ATR calculation
            volume_threshold (float): Minimum volume multiplier for OB confirmation
            ob_invalidation_atr (float): ATR multiplier for OB invalidation
            fvg_auto_threshold (bool): Auto filter insignificant FVGs
            structure_confluence_filter (bool): Filter non-significant internal structure
            swing_length (int): Length for swing point detection
            equal_length (int): Bars needed to confirm equal highs/lows
            mode (str): Analysis mode - 'historical' or 'present'
        """
        self.atr_period = atr_period
        self.volume_threshold = volume_threshold
        self.ob_invalidation_atr = ob_invalidation_atr
        self.fvg_auto_threshold = fvg_auto_threshold
        self.structure_confluence_filter = structure_confluence_filter
        self.swing_length = swing_length
        self.equal_length = equal_length
        self.mode = mode
        
        # Initialize state variables
        self.current_trend = Bias.BULLISH
        self.current_leg = Leg.BULLISH
        self.last_swing_high = None
        self.last_swing_low = None
        self.internal_trend = Bias.BULLISH
        self.internal_leg = Leg.BULLISH
        
        # Storage for various components
        self.swing_points: List[SwingPoint] = []
        self.structure_points: List[StructurePoint] = []
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.equal_levels: List[EqualLevel] = []
        
        self.market_structure = MarketStructureAnalyzer()
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(self.atr_period).mean()
    
    def detect_leg(self, df: pd.DataFrame, size: int = None) -> Leg:
        """
        Detect the current market leg (bullish or bearish)
        
        Args:
            df: OHLCV DataFrame
            size: Window size for leg detection
            
        Returns:
            Leg: Current leg type (BULLISH or BEARISH)
        """
        if size is None:
            size = self.swing_length
            
        highest = df['high'].rolling(size).max()
        lowest = df['low'].rolling(size).min()
        
        new_leg_high = df['high'] > highest.shift(1)
        new_leg_low = df['low'] < lowest.shift(1)
        
        return Leg.BEARISH if new_leg_high.iloc[-1] else Leg.BULLISH if new_leg_low.iloc[-1] else self.current_leg

    def is_high_volatility_bar(self, df: pd.DataFrame, index: int) -> bool:
        """Check if the current bar is a high volatility bar"""
        atr = self.calculate_atr(df)
        bar_range = df['high'].iloc[index] - df['low'].iloc[index]
        return bar_range >= 2 * atr.iloc[index]

    def get_parsed_price(self, df: pd.DataFrame, index: int) -> Tuple[float, float]:
        """Get parsed high/low prices accounting for volatility"""
        is_volatile = self.is_high_volatility_bar(df, index)
        
        if is_volatile:
            return df['low'].iloc[index], df['high'].iloc[index]  # Reversed for volatile bars
        return df['high'].iloc[index], df['low'].iloc[index]
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Identify potential order blocks based on volume and price action
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[OrderBlock]: Identified order blocks
        """
        order_blocks = []
        atr = self.calculate_atr(df)
        volume_sma = df['volume'].rolling(20).mean()
        
        for i in range(3, len(df)):
            # Check for bullish order block
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Current candle bullish
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Previous candle bearish
                df['volume'].iloc[i] > volume_sma.iloc[i] * self.volume_threshold):
                
                order_blocks.append(
                    OrderBlock(
                        index=i,
                        price_high=df['high'].iloc[i-1],
                        price_low=df['low'].iloc[i-1],
                        type='bullish',
                        volume=df['volume'].iloc[i],
                        atr_multiple=atr.iloc[i]
                    )
                )
            
            # Check for bearish order block
            elif (df['close'].iloc[i] < df['open'].iloc[i] and  # Current candle bearish
                  df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Previous candle bullish
                  df['volume'].iloc[i] > volume_sma.iloc[i] * self.volume_threshold):
                
                order_blocks.append(
                    OrderBlock(
                        index=i,
                        price_high=df['high'].iloc[i-1],
                        price_low=df['low'].iloc[i-1],
                        type='bearish',
                        volume=df['volume'].iloc[i],
                        atr_multiple=atr.iloc[i]
                    )
                )
        
        return order_blocks
    
    def calculate_take_profit_levels(self,
                                   entry_price: float,
                                   atr: float,
                                   is_long: bool) -> List[float]:
        """
        Calculate tiered take-profit levels based on ATR
        
        Args:
            entry_price (float): Entry price level
            atr (float): Current ATR value
            is_long (bool): True for long positions, False for shorts
            
        Returns:
            List[float]: Three take-profit levels
        """
        multipliers = [1.5, 2.5, 3.5]  # ATR multipliers for each TP level
        levels = []
        
        for mult in multipliers:
            if is_long:
                tp = entry_price + (atr * mult)
            else:
                tp = entry_price - (atr * mult)
            levels.append(tp)
            
        return levels
    
    def generate_volume_heatmap(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate volume heatmap scores
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.Series: Volume heatmap scores (0-1 scale)
        """
        # Calculate relative volume
        volume_sma = df['volume'].rolling(20).mean()
        relative_volume = df['volume'] / volume_sma
        
        # Normalize to 0-1 scale
        min_vol = relative_volume.rolling(50).min()
        max_vol = relative_volume.rolling(50).max()
        
        heatmap = (relative_volume - min_vol) / (max_vol - min_vol)
        return heatmap.fillna(0.5)
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price action
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[FairValueGap]: Detected FVGs
        """
        fvgs = []
        atr = self.calculate_atr(df)
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                
                # Filter small gaps if auto threshold enabled
                if not self.fvg_auto_threshold or gap_size > atr.iloc[i] * 0.5:
                    fvgs.append(
                        FairValueGap(
                            index=i,
                            high=df['low'].iloc[i],
                            low=df['high'].iloc[i-2],
                            type='bullish',
                            size=gap_size
                        )
                    )
            
            # Bearish FVG
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                
                if not self.fvg_auto_threshold or gap_size > atr.iloc[i] * 0.5:
                    fvgs.append(
                        FairValueGap(
                            index=i,
                            high=df['low'].iloc[i-2],
                            low=df['high'].iloc[i],
                            type='bearish',
                            size=gap_size
                        )
                    )
        
        return fvgs

    def detect_equal_levels(self, 
                          df: pd.DataFrame, 
                          confirmation_bars: int = 3,
                          tolerance: float = 0.0001  # 0.01% tolerance
                          ) -> List[EqualLevel]:
        """
        Detect Equal Highs and Lows
        
        Args:
            df (pd.DataFrame): OHLCV data
            confirmation_bars (int): Bars needed to confirm EQH/EQL
            tolerance (float): Price tolerance for equality
            
        Returns:
            List[EqualLevel]: Detected equal levels
        """
        levels = []
        
        for i in range(confirmation_bars + 1, len(df)):
            # Check for Equal Highs
            high_diff = abs(df['high'].iloc[i] - df['high'].iloc[i-confirmation_bars])
            if high_diff <= df['high'].iloc[i] * tolerance:
                levels.append(
                    EqualLevel(
                        index=i,
                        price=df['high'].iloc[i],
                        type='high',
                        confirmation_bars=confirmation_bars
                    )
                )
            
            # Check for Equal Lows
            low_diff = abs(df['low'].iloc[i] - df['low'].iloc[i-confirmation_bars])
            if low_diff <= df['low'].iloc[i] * tolerance:
                levels.append(
                    EqualLevel(
                        index=i,
                        price=df['low'].iloc[i],
                        type='low',
                        confirmation_bars=confirmation_bars
                    )
                )
        
        return levels

    def calculate_premium_discount_zones(self, 
                                      df: pd.DataFrame, 
                                      period: int = 20
                                      ) -> Dict[str, pd.Series]:
        """
        Calculate Premium, Discount, and Equilibrium zones
        
        Args:
            df (pd.DataFrame): OHLCV data
            period (int): Period for calculations
            
        Returns:
            Dict[str, pd.Series]: Zone levels
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        std = typical_price.rolling(period).std()
        
        premium_zone = sma + std
        discount_zone = sma - std
        equilibrium = sma
        
        return {
            'premium': premium_zone,
            'discount': discount_zone,
            'equilibrium': equilibrium
        }

    def analyze_market_context(self, 
                             df: pd.DataFrame,
                             lookback_periods: int = 50
                             ) -> Dict:
        """
        Comprehensive market analysis combining all SMC concepts
        
        Args:
            df (pd.DataFrame): OHLCV data
            lookback_periods (int): Analysis lookback period
            
        Returns:
            Dict: Analysis results including all SMC components
        """
        # Get base analysis
        base_analysis = super().analyze_market_context(df, lookback_periods)
        
        # Add new components
        fvgs = self.detect_fair_value_gaps(df)
        equal_levels = self.detect_equal_levels(df)
        zones = self.calculate_premium_discount_zones(df)
        
        # Update return dictionary
        base_analysis.update({
            'fair_value_gaps': fvgs,
            'equal_levels': equal_levels,
            'premium_discount_zones': zones
        })
        
        return base_analysis 

    def detect_structure_points(self, df: pd.DataFrame, internal: bool = False) -> List[StructurePoint]:
        """
        Detect market structure points (BOS and CHoCH)
        
        Args:
            df: OHLCV DataFrame
            internal: If True, detect internal structure, else swing structure
            
        Returns:
            List[StructurePoint]: Detected structure points
        """
        size = 5 if internal else self.swing_length
        structure_points = []
        
        # Get current leg and trend
        current_leg = self.detect_leg(df, size)
        trend = self.internal_trend if internal else self.current_trend
        
        # Detect structure breaks
        for i in range(size + 1, len(df)):
            # Skip if filtering is enabled and conditions aren't met
            if internal and self.structure_confluence_filter:
                bullish_bar = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i]) > min(df['close'].iloc[i], df['open'].iloc[i] - df['low'].iloc[i])
                bearish_bar = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i]) < min(df['close'].iloc[i], df['open'].iloc[i] - df['low'].iloc[i])
                if not (bullish_bar or bearish_bar):
                    continue
            
            # Bullish structure break
            if df['close'].iloc[i] > df['high'].iloc[i-1]:
                structure_type = StructureType.CHOCH if trend == Bias.BEARISH else StructureType.BOS
                structure_points.append(
                    StructurePoint(
                        index=i,
                        price=df['high'].iloc[i-1],
                        type=structure_type,
                        level="internal" if internal else "swing",
                        direction=Bias.BULLISH,
                        creation_time=df.index[i],
                        leg_type=current_leg
                    )
                )
                trend = Bias.BULLISH
            
            # Bearish structure break
            elif df['close'].iloc[i] < df['low'].iloc[i-1]:
                structure_type = StructureType.CHOCH if trend == Bias.BULLISH else StructureType.BOS
                structure_points.append(
                    StructurePoint(
                        index=i,
                        price=df['low'].iloc[i-1],
                        type=structure_type,
                        level="internal" if internal else "swing",
                        direction=Bias.BEARISH,
                        creation_time=df.index[i],
                        leg_type=current_leg
                    )
                )
                trend = Bias.BEARISH
        
        # Update trend state
        if internal:
            self.internal_trend = trend
        else:
            self.current_trend = trend
            
        return structure_points

    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect swing points (HH, LL, LH, HL)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List[SwingPoint]: Detected swing points
        """
        swing_points = []
        current_leg = self.detect_leg(df)
        
        for i in range(self.swing_length + 1, len(df)):
            # Higher High
            if df['high'].iloc[i] > df['high'].iloc[i-1] and current_leg == Leg.BULLISH:
                swing_points.append(
                    SwingPoint(
                        index=i,
                        price=df['high'].iloc[i],
                        type='HH',
                        creation_time=df.index[i],
                        strength='strong' if self.current_trend == Bias.BULLISH else 'weak'
                    )
                )
                self.last_swing_high = df['high'].iloc[i]
            
            # Lower Low
            elif df['low'].iloc[i] < df['low'].iloc[i-1] and current_leg == Leg.BEARISH:
                swing_points.append(
                    SwingPoint(
                        index=i,
                        price=df['low'].iloc[i],
                        type='LL',
                        creation_time=df.index[i],
                        strength='strong' if self.current_trend == Bias.BEARISH else 'weak'
                    )
                )
                self.last_swing_low = df['low'].iloc[i]
            
            # Lower High
            elif df['high'].iloc[i] < self.last_swing_high and current_leg == Leg.BEARISH:
                swing_points.append(
                    SwingPoint(
                        index=i,
                        price=df['high'].iloc[i],
                        type='LH',
                        creation_time=df.index[i],
                        strength='strong' if self.current_trend == Bias.BEARISH else 'weak'
                    )
                )
            
            # Higher Low
            elif df['low'].iloc[i] > self.last_swing_low and current_leg == Leg.BULLISH:
                swing_points.append(
                    SwingPoint(
                        index=i,
                        price=df['low'].iloc[i],
                        type='HL',
                        creation_time=df.index[i],
                        strength='strong' if self.current_trend == Bias.BULLISH else 'weak'
                    )
                )
        
        return swing_points 