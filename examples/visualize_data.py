"""
Example script showing how to visualize your trading data from Neon
"""

from src.utils.neon_visualizer import NeonVisualizer

# Your Neon connection string (you'll get this from your Neon dashboard)
NEON_CONNECTION = "postgresql://[YOUR-USERNAME]:[YOUR-PASSWORD]@[YOUR-ENDPOINT]/neondb"

# Create the visualizer
viz = NeonVisualizer(NEON_CONNECTION)

# Let's look at some BTC data
symbol = "BTC/USD"

# 1. First, let's see some basic stats
viz.get_quick_stats(symbol)

# 2. View recent price action (last 7 days)
viz.view_recent_data(symbol, days=7)

# 3. View all your indicators
viz.view_indicators(symbol, days=7)

# 4. If you have model predictions, view performance
viz.view_model_performance(symbol, days=30)

# That's it! The charts will open in your browser using Plotly 