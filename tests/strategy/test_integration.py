import pytest
import pandas as pd
import numpy as np
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier
from src.models.strategy.confirmation.logistic_regression_torch import LogisticRegression
from src.models.strategy.risk_management.chandelier_exit import ChandelierExit
from src.utils.position_sizing import calculate_position_size
import torch

class StrategyTester:
    def __init__(
        self,
        initial_capital: float = 10000,
        max_positions: int = 2,
        min_confidence: float = 0.3
    ):
        # Initialize strategy components
        self.classifier = LorentzianClassifier()
        self.confirmation = LogisticRegression()
        self.risk_manager = ChandelierExit()
        
        # Strategy parameters
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.positions = []
        
    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run strategy backtest with all components"""
        # Generate signals from each component
        primary_signals = self.classifier.calculate_signals(data)
        confirmed_signals = self.confirmation.calculate_signals(data)
        risk_signals = self.risk_manager.calculate_signals(data)
        
        # Track results
        equity_curve = [self.initial_capital]
        trades = []
        
        for i in range(len(data)):
            # Update positions and check exits
            for pos in self.positions[:]:
                if self._check_exit(pos, data.iloc[i], risk_signals, i):
                    self.positions.remove(pos)
                    trades.append(pos)
            
            # Check for new entries
            if len(self.positions) < self.max_positions:
                entry = self._check_entry(
                    data.iloc[i],
                    primary_signals,
                    confirmed_signals,
                    i
                )
                if entry:
                    self.positions.append(entry)
            
            # Update equity
            current_value = self._calculate_portfolio_value(data.iloc[i])
            equity_curve.append(current_value)
        
        return self._calculate_results(trades, equity_curve)
    
    def _check_exit(self, position, current_bar, risk_signals, idx):
        """Check if position should be exited"""
        if position['direction'] == 'LONG':
            stop_hit = current_bar['low'] <= risk_signals['long_stop'][idx]
            if stop_hit:
                position['exit_price'] = risk_signals['long_stop'][idx]
                position['exit_time'] = current_bar.name
                return True
        else:  # SHORT
            stop_hit = current_bar['high'] >= risk_signals['short_stop'][idx]
            if stop_hit:
                position['exit_price'] = risk_signals['short_stop'][idx]
                position['exit_time'] = current_bar.name
                return True
        return False
    
    def _check_entry(self, current_bar, primary_signals, confirmed_signals, idx):
        """Check for new position entry"""
        # Long entry conditions
        long_signal = (
            primary_signals['buy_signals'][idx] == 1 and
            confirmed_signals['predictions'][idx] > self.min_confidence
        )
        
        # Short entry conditions
        short_signal = (
            primary_signals['sell_signals'][idx] == 1 and
            confirmed_signals['predictions'][idx] < (1 - self.min_confidence)
        )
        
        if long_signal or short_signal:
            direction = 'LONG' if long_signal else 'SHORT'
            size = calculate_position_size(
                self.capital,
                risk_per_trade=0.02,  # 2% risk per trade
                stop_distance=self.risk_manager.config.atr_multiplier
            )
            
            return {
                'direction': direction,
                'entry_price': current_bar['close'],
                'entry_time': current_bar.name,
                'size': size,
                'confidence': confirmed_signals['predictions'][idx].item()
            }
        return None
    
    def _calculate_portfolio_value(self, current_bar):
        """Calculate current portfolio value including open positions"""
        value = self.capital
        for pos in self.positions:
            if pos['direction'] == 'LONG':
                pnl = (current_bar['close'] - pos['entry_price']) * pos['size']
            else:  # SHORT
                pnl = (pos['entry_price'] - current_bar['close']) * pos['size']
            value += pnl
        return value
    
    def _calculate_results(self, trades, equity_curve):
        """Calculate backtest results and statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'equity_curve': equity_curve
            }
        
        # Calculate trade stats
        wins = [t for t in trades if self._calculate_trade_pnl(t) > 0]
        losses = [t for t in trades if self._calculate_trade_pnl(t) <= 0]
        
        total_profit = sum(self._calculate_trade_pnl(t) for t in wins)
        total_loss = abs(sum(self._calculate_trade_pnl(t) for t in losses))
        
        # Calculate metrics
        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = total_profit / total_loss if total_loss else float('inf')
        
        # Calculate Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max()
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve
        }
    
    def _calculate_trade_pnl(self, trade):
        """Calculate P&L for a single trade"""
        if trade['direction'] == 'LONG':
            return (trade['exit_price'] - trade['entry_price']) * trade['size']
        else:  # SHORT
            return (trade['entry_price'] - trade['exit_price']) * trade['size']

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

@pytest.mark.strategy
class TestStrategy:
    def test_strategy_backtest(self, sample_data):
        """Test full strategy backtest"""
        strategy = StrategyTester(
            initial_capital=10000,
            max_positions=2,
            min_confidence=0.3
        )
        
        results = strategy.run_backtest(sample_data)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'profit_factor' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'equity_curve' in results
        
        # Verify metrics are within reasonable bounds
        assert results['win_rate'] >= 0 and results['win_rate'] <= 1
        assert results['profit_factor'] >= 0
        assert results['max_drawdown'] >= 0 and results['max_drawdown'] <= 1
        assert len(results['equity_curve']) == len(sample_data) + 1  # +1 for initial capital 