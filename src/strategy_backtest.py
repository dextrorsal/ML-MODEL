import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier, Direction, LorentzianSettings
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: Direction
    pnl: float
    pnl_pct: float
    stop_loss: float
    take_profit: float
    confidence: float

class StrategyBacktest:
    def __init__(
        self,
        initial_capital: float = 10000,
        max_positions: int = 3,  # Max concurrent positions
        min_confidence: float = 0.3  # Minimum confidence for entry
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.trades: List[Trade] = []
        self.open_positions: List[Dict] = []
        
        # Initialize ML model with default settings
        self.model = LorentzianClassifier(
            config=LorentzianSettings(
                use_volatility_filter=True,
                use_regime_filter=True,
                use_adx_filter=True
            )
        )
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        # Get predictions and confidence scores
        predictions, confidence_scores = self.model.predict(df)
        
        # Track equity curve
        equity_curve = [self.initial_capital]
        
        # Iterate through data
        for i in range(len(predictions)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check open positions for exits
            for pos in self.open_positions[:]:  # Copy list for iteration
                # Check stop loss
                if pos['direction'] == Direction.LONG:
                    if df['low'].iloc[i] <= pos['stop_loss']:
                        self._close_position(pos, current_time, pos['stop_loss'])
                        self.open_positions.remove(pos)
                    elif df['high'].iloc[i] >= pos['take_profit']:
                        self._close_position(pos, current_time, pos['take_profit'])
                        self.open_positions.remove(pos)
                else:  # Short position
                    if df['high'].iloc[i] >= pos['stop_loss']:
                        self._close_position(pos, current_time, pos['stop_loss'])
                        self.open_positions.remove(pos)
                    elif df['low'].iloc[i] <= pos['take_profit']:
                        self._close_position(pos, current_time, pos['take_profit'])
                        self.open_positions.remove(pos)
            
            # Check for new entries
            if len(self.open_positions) < self.max_positions:
                prediction = predictions[i]
                confidence = confidence_scores[i]
                
                if confidence >= self.min_confidence:
                    # Get trade parameters
                    params = self.model.get_trade_params(df, i)
                    
                    if prediction == Direction.LONG:
                        stop_loss = current_price * (1 - params['stop_loss_pct'])
                        take_profit = current_price * (1 + params['take_profit_pct'])
                        position_size = self.capital * params['position_size_pct']
                        
                        self.open_positions.append({
                            'direction': Direction.LONG,
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size,
                            'confidence': confidence
                        })
                        
                    elif prediction == Direction.SHORT:
                        stop_loss = current_price * (1 + params['stop_loss_pct'])
                        take_profit = current_price * (1 - params['take_profit_pct'])
                        position_size = self.capital * params['position_size_pct']
                        
                        self.open_positions.append({
                            'direction': Direction.SHORT,
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size,
                            'confidence': confidence
                        })
            
            # Update equity curve
            equity_curve.append(self.capital + sum(
                pos['position_size'] * (
                    (current_price - pos['entry_price'])/pos['entry_price'] 
                    if pos['direction'] == Direction.LONG else
                    (pos['entry_price'] - current_price)/pos['entry_price']
                ) for pos in self.open_positions
            ))
        
        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_capital': equity_curve[-1],
            'return_pct': (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100,
            'equity_curve': equity_curve
        }
    
    def _close_position(self, position: Dict, exit_time: pd.Timestamp, exit_price: float):
        """Handle position closure and trade recording"""
        pnl = (
            (exit_price - position['entry_price']) * position['position_size'] / position['entry_price']
            if position['direction'] == Direction.LONG else
            (position['entry_price'] - exit_price) * position['position_size'] / position['entry_price']
        )
        
        pnl_pct = pnl / position['position_size'] * 100
        
        self.capital += pnl
        
        self.trades.append(Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            direction=position['direction'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            confidence=position['confidence']
        ))
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.trades:
            print("No trades to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot equity curve
        equity_curve = pd.Series([t.pnl for t in self.trades]).cumsum() + self.initial_capital
        equity_curve.plot(ax=ax1, title='Equity Curve')
        ax1.set_ylabel('Capital')
        
        # Plot trade PnL distribution
        sns.histplot([t.pnl for t in self.trades], ax=ax2, bins=50)
        ax2.set_title('Trade PnL Distribution')
        ax2.set_xlabel('PnL')
        
        # Plot confidence vs PnL scatter
        confidences = [t.confidence for t in self.trades]
        pnls = [t.pnl for t in self.trades]
        ax3.scatter(confidences, pnls)
        ax3.set_title('Confidence vs PnL')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('PnL')
        
        plt.tight_layout()
        plt.show()
        
    def print_stats(self):
        """Print detailed strategy statistics"""
        if not self.trades:
            print("No trades to analyze")
            return
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        win_rate = winning_trades / total_trades * 100
        
        print("\n=== Strategy Statistics ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {((self.capital - self.initial_capital) / self.initial_capital * 100):.2f}%")
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Average trade metrics
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0])
        avg_loss = abs(np.mean([t.pnl for t in self.trades if t.pnl < 0]))
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {(avg_win/avg_loss if avg_loss > 0 else float('inf')):.2f}")
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        equity_curve = pd.Series([t.pnl for t in self.trades]).cumsum() + self.initial_capital
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max * 100
        return abs(drawdowns.min()) 