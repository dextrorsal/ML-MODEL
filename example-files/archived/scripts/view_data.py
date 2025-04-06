import pandas as pd
from datetime import datetime

# Read the 5m data file
df = pd.read_feather('/home/dex/user_data/data/bitget/futures/SOL_USDT_USDT-5m-futures.feather')

# Convert timestamp to datetime
df['date'] = pd.to_datetime(df['date'], unit='ms')

# Get March 28th, 2025 data
march_28_data = df[df['date'].dt.date == datetime(2025, 3, 28).date()]

if len(march_28_data) > 0:
    print("\nSOL/USDT Data for March 28th, 2025:")
    print("Time (UTC) | Open | High | Low | Close | Volume")
    print("-" * 60)
    for _, row in march_28_data.iterrows():
        print(f"{row['date'].strftime('%H:%M:%S')} | {row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | {row['close']:.2f} | {row['volume']:.2f}")
else:
    print("No data found for March 28th, 2025")

# Show data range
print(f"\nData range in file:")
print(f"From: {df['date'].min()}")
print(f"To: {df['date'].max()}")