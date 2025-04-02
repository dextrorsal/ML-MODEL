from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class MarketConditions:
    volume_ratio: float  # Current volume compared to average
    sentiment_score: float  # -1 to 1 scale
    is_bullish: bool

class PositionSizer:
    def __init__(self, portfolio_value: float):
        """
        Initialize PositionSizer
        
        Args:
            portfolio_value (float): Total portfolio value
        """
        self.portfolio_value = portfolio_value
        self.leverage_allocations = {
            range(50, 101): (0.01, 0.10),  # 1-10% for 50-100x
            range(20, 50): (0.10, 0.20),   # 10-20% for 20-50x
            range(1, 20): (0.20, 1.0)      # 20%+ for under 20x
        }
    
    def calculate_position_size(
        self, 
        leverage: int,
        market_conditions: MarketConditions
    ) -> Tuple[float, float]:
        """
        Calculate position size based on leverage and market conditions
        
        Args:
            leverage (int): Intended leverage
            market_conditions (MarketConditions): Current market conditions
            
        Returns:
            Tuple[float, float]: (min_size, max_size) in terms of portfolio percentage
        """
        # Get base allocation range for leverage
        base_range = self._get_leverage_allocation(leverage)
        if not base_range:
            raise ValueError(f"Unsupported leverage: {leverage}")
        
        min_size, max_size = base_range
        
        # Adjust based on market conditions
        adjustment = self._calculate_condition_adjustment(market_conditions)
        
        # Apply adjustment while keeping within reasonable bounds
        adjusted_min = max(0.01, min(0.5, min_size * adjustment))
        adjusted_max = max(0.01, min(1.0, max_size * adjustment))
        
        return adjusted_min, adjusted_max
    
    def _get_leverage_allocation(self, leverage: int) -> Optional[Tuple[float, float]]:
        """Get base allocation range for given leverage"""
        for leverage_range, allocation in self.leverage_allocations.items():
            if leverage in leverage_range:
                return allocation
        return None
    
    def _calculate_condition_adjustment(self, conditions: MarketConditions) -> float:
        """
        Calculate position size adjustment based on market conditions
        Returns a multiplier (0.5 to 1.5)
        """
        # Start with base multiplier
        multiplier = 1.0
        
        # Volume impact (-0.25 to +0.25)
        volume_impact = (conditions.volume_ratio - 1) * 0.25
        multiplier += volume_impact
        
        # Sentiment impact (-0.15 to +0.15)
        sentiment_impact = conditions.sentiment_score * 0.15
        multiplier += sentiment_impact
        
        # Trend impact (+0.1 if bullish)
        if conditions.is_bullish:
            multiplier += 0.1
            
        # Ensure multiplier stays within reasonable bounds
        return max(0.5, min(1.5, multiplier))
    
    def calculate_scale_in_levels(
        self,
        entry_price: float,
        position_size: float,
        max_loss_percentage: float = 0.5  # 50% loss
    ) -> Dict[float, float]:
        """
        Calculate scale-in levels based on position loss
        
        Args:
            entry_price (float): Initial entry price
            position_size (float): Initial position size in portfolio percentage
            max_loss_percentage (float): Maximum loss percentage to scale in
            
        Returns:
            Dict[float, float]: Price levels mapped to additional position sizes
        """
        scale_levels = {}
        
        # Calculate 3 scale-in levels
        for i in range(3):
            loss_pct = (i + 1) * (max_loss_percentage / 3)
            price_level = entry_price * (1 - loss_pct)
            
            # Increase position size at each level
            additional_size = position_size * (1 + (i * 0.5))
            
            scale_levels[price_level] = additional_size
            
        return scale_levels 