import pandas as pd
import numpy as np

def chandelier_exit(df, atr_period=22, atr_multiplier=3.0, use_close=True):
    """
    Calculate Chandelier Exit indicator for risk management.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'open', 'high', 'low', 'close' price data
    atr_period : int, default=22
        Period for ATR calculation
    atr_multiplier : float, default=3.0
        Multiplier for ATR to determine stop distance
    use_close : bool, default=True
        Whether to use close price for calculating extremums
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data plus:
        - long_stop: Long stop level (for short trades)
        - short_stop: Short stop level (for long trades)
        - dir: Direction (1=long, -1=short)
        - buy_signal: Boolean, True when signal changes to buy
        - sell_signal: Boolean, True when signal changes to sell
    """
    # Make a copy of the dataframe to avoid modifying the original
    result = df.copy()
    
    # Calculate ATR
    result['tr'] = calculate_true_range(result)
    result['atr'] = calculate_atr(result['tr'], atr_period)
    
    # Calculate highest high and lowest low
    if use_close:
        result['highest'] = result['close'].rolling(window=atr_period).max()
        result['lowest'] = result['close'].rolling(window=atr_period).min()
    else:
        result['highest'] = result['high'].rolling(window=atr_period).max()
        result['lowest'] = result['low'].rolling(window=atr_period).min()
    
    # Initialize long_stop and short_stop columns
    result['long_stop'] = result['highest'] - (atr_multiplier * result['atr'])
    result['short_stop'] = result['lowest'] + (atr_multiplier * result['atr'])
    
    # Create columns for previous values
    result['long_stop_prev'] = result['long_stop'].shift(1)
    result['short_stop_prev'] = result['short_stop'].shift(1)
    
    # Replace NaN values with current values
    result['long_stop_prev'] = result['long_stop_prev'].fillna(result['long_stop'])
    result['short_stop_prev'] = result['short_stop_prev'].fillna(result['short_stop'])
    
    # Apply the trailing stop logic
    for i in range(1, len(result)):
        prev_close = result.iloc[i-1]['close']
        prev_long_stop = result.iloc[i-1]['long_stop']
        prev_short_stop = result.iloc[i-1]['short_stop']
        
        # Long stop logic
        if prev_close > prev_long_stop:
            result.loc[result.index[i], 'long_stop'] = max(
                result.iloc[i]['long_stop'],
                prev_long_stop
            )
        
        # Short stop logic
        if prev_close < prev_short_stop:
            result.loc[result.index[i], 'short_stop'] = min(
                result.iloc[i]['short_stop'],
                prev_short_stop
            )
    
    # Initialize direction column (1 for long, -1 for short)
    result['dir'] = 1  # default to long
    
    # Calculate direction based on price vs stops
    for i in range(1, len(result)):
        prev_close = result.iloc[i-1]['close']
        long_stop_prev = result.iloc[i-1]['long_stop']
        short_stop_prev = result.iloc[i-1]['short_stop']
        
        if prev_close > short_stop_prev:
            result.loc[result.index[i], 'dir'] = 1
        elif prev_close < long_stop_prev:
            result.loc[result.index[i], 'dir'] = -1
        else:
            result.loc[result.index[i], 'dir'] = result.iloc[i-1]['dir']
    
    # Calculate buy/sell signals
    result['prev_dir'] = result['dir'].shift(1).fillna(1)
    result['buy_signal'] = (result['dir'] == 1) & (result['prev_dir'] == -1)
    result['sell_signal'] = (result['dir'] == -1) & (result['prev_dir'] == 1)
    
    # Clean up temporary columns
    result = result.drop(['tr', 'highest', 'lowest', 'long_stop_prev', 
                         'short_stop_prev', 'prev_dir'], axis=1)
    
    return result

def calculate_true_range(df):
    """Calculate True Range for ATR computation"""
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range

def calculate_atr(tr_series, period):
    """Calculate Average True Range"""
    atr = tr_series.rolling(window=period).mean()
    return atr

def calculate_ohlc4(df):
    """Calculate OHLC4 (average of open, high, low, close)"""
    return (df['open'] + df['high'] + df['low'] + df['close']) / 4

# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        'open': [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
        'high': [12, 13, 14, 12, 11, 10, 11, 12, 13, 14],
        'low': [9, 10, 11, 10, 9, 8, 9, 10, 11, 12],
        'close': [11, 12, 13, 10, 9, 9, 10, 11, 12, 13]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Chandelier Exit
    result = chandelier_exit(df, atr_period=3, atr_multiplier=2.0)
    
    # Print results
    print("Chandelier Exit Results:")
    print(result[['open', 'high', 'low', 'close', 'atr', 'long_stop', 'short_stop', 'dir', 'buy_signal', 'sell_signal']])