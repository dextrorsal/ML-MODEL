#!/usr/bin/env python3
"""
Script to monitor Freqtrade live trading performance
"""
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import argparse

def monitor_trades(db_path="tradesv3_live.sqlite", refresh_interval=60):
    """Monitor live trading performance"""
    
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        return False
    
    print(f"Monitoring trades from {db_path}")
    print(f"Refreshing every {refresh_interval} seconds. Press Ctrl+C to exit.")
    
    try:
        while True:
            # Connect to database
            conn = sqlite3.connect(db_path)
            
            # Load trades
            trades_df = pd.read_sql("SELECT * FROM trades", conn)
            
            # Close connection
            conn.close()
            
            if len(trades_df) == 0:
                print("No trades found in database")
                time.sleep(refresh_interval)
                continue
            
            # Convert dates
            for col in ['open_date', 'close_date']:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col], utc=True)
            
            # Calculate statistics
            open_trades = trades_df[trades_df['is_open']].copy()
            closed_trades = trades_df[~trades_df['is_open']].copy()
            
            # Print summary
            print("\n" + "="*50)
            print(f"Trading Summary ({pd.Timestamp.now()})")
            print("="*50)
            
            print(f"Total trades: {len(trades_df)}")
            print(f"Open trades: {len(open_trades)}")
            print(f"Closed trades: {len(closed_trades)}")
            
            if len(closed_trades) > 0:
                winning_trades = closed_trades[closed_trades['profit_ratio'] > 0]
                losing_trades = closed_trades[closed_trades['profit_ratio'] <= 0]
                
                win_rate = len(winning_trades) / len(closed_trades)
                avg_profit = closed_trades['profit_ratio'].mean() * 100
                total_profit = closed_trades['profit_abs'].sum()
                
                print(f"Win rate: {win_rate:.2%}")
                print(f"Average profit: {avg_profit:.2f}%")
                print(f"Total profit: {total_profit:.4f} {closed_trades['stake_currency'].iloc[0]}")
                
                # Calculate by pair
                pair_performance = closed_trades.groupby('pair')['profit_ratio'].agg(
                    ['count', 'mean', 'sum']
                ).sort_values('sum', ascending=False)
                
                pair_performance['mean'] *= 100  # Convert to percentage
                
                print("\nPerformance by pair:")
                print(pair_performance.head())
            
            if len(open_trades) > 0:
                print("\nCurrent open trades:")
                for _, trade in open_trades.iterrows():
                    current_profit = trade['profit_ratio'] * 100 if pd.notnull(trade['profit_ratio']) else 0
                    duration = pd.Timestamp.now(tz='UTC') - trade['open_date']
                    hours = duration.total_seconds() / 3600
                    
                    print(f"{trade['pair']}: {current_profit:.2f}% (open for {hours:.1f} hours)")
            
            # Create a visualization if we have closed trades
            if len(closed_trades) > 0:
                # Sort by close date
                closed_trades = closed_trades.sort_values('close_date')
                
                # Calculate cumulative profit
                closed_trades['cumulative_profit'] = closed_trades['profit_abs'].cumsum()
                
                # Plot
                plt.figure(figsize=(12, 6))
                plt.plot(closed_trades['close_date'], closed_trades['cumulative_profit'])
                plt.title('Cumulative Profit')
                plt.xlabel('Date')
                plt.ylabel(f"Profit ({closed_trades['stake_currency'].iloc[0]})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save
                plt.savefig('performance_chart.png')
                print("\nPerformance chart saved as 'performance_chart.png'")
            
            # Wait before refreshing
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except Exception as e:
        print(f"Error monitoring trades: {e}")
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor Freqtrade live trading performance')
    parser.add_argument('--db', type=str, default='tradesv3_live.sqlite', help='Path to database file')
    parser.add_argument('--interval', type=int, default=60, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor_trades(db_path=args.db, refresh_interval=args.interval) 