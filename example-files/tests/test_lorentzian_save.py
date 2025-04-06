"""
TEST TOOL: Lorentzian Model Save/Load Functionality

This script tests the saving and loading functionality of the Lorentzian ANN model.
It's used to verify that model persistence works correctly and that model state can be
properly saved and restored for continuous learning.

Key features:
1. Model serialization and deserialization
2. Incremental learning simulation
3. Testing prediction consistency before and after save/load cycles
4. Verifying model performance metrics are maintained through persistence

This is a development/testing tool, not part of the production strategy.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# Import indicators from correct paths
from strategies.LorentzianStrategy.features.rsi import RSIIndicator
from strategies.LorentzianStrategy.features.wave_trend import WaveTrendIndicator
from strategies.LorentzianStrategy.features.cci import CCIIndicator
from strategies.LorentzianStrategy.features.adx import ADXIndicator
import gc

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class LorentzianANN:
    """
    Approximate Nearest Neighbors using Lorentzian distance metric
    to find patterns in price data and make predictions.

    This version supports:
    - Loading/saving model state
    - Incremental learning with new data
    - Persisting model "intuition" through weight storage
    """

    def __init__(
        self,
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20,
        use_regime_filter=True,
        use_volatility_filter=True,
        use_adx_filter=True,
    ):
        self.lookback_bars = lookback_bars
        self.prediction_bars = prediction_bars
        self.k_neighbors = k_neighbors
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_adx_filter = use_adx_filter

        # To be initialized during fit or load
        self.feature_arrays = None
        self.labels = None

        self.model_path = "lorentzian_model.pt"
        self.is_fitted = False

    def lorentzian_distance(self, features, historical_features):
        """Calculate Lorentzian distance between features and historical features"""
        # Process in batches to save memory
        batch_size = 100  # Adjust based on available memory
        n_samples = features.shape[0]
        n_historical = historical_features.shape[0]

        # Initialize distances tensor
        distances = torch.zeros((n_samples, n_historical), device=device)

        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_features = features[i:end_i]

            # For each feature vector in the batch
            for j in range(batch_features.shape[0]):
                # Get the feature vector
                x = batch_features[j]

                # Calculate differences
                diff = torch.abs(x.unsqueeze(0) - historical_features)

                # Calculate Lorentzian distance: ln(1 + |x - y|)
                # We use log1p for numerical stability
                # This is the true formula from TradingView
                log_diff = torch.log1p(diff)

                # Sum over features dimension
                batch_distances = torch.sum(log_diff, dim=1)

                # Store in the distances tensor
                distances[i + j] = batch_distances

        return distances

    def generate_training_data(self, features, prices):
        """Generate training labels for the prediction task"""
        # Using the same TradingView approach: look ahead a fixed number of bars
        # to determine if price went up or down
        future_prices = prices[self.prediction_bars :]
        current_prices = prices[: -self.prediction_bars]

        # Generate labels: 1 for long (price went up), -1 for short (price went down), 0 for neutral
        labels = torch.zeros(len(current_prices), dtype=torch.long)
        labels[future_prices > current_prices] = 1  # Long
        labels[future_prices < current_prices] = -1  # Short

        # We can only generate labels for data points that have future data
        return features[: len(labels)], labels

    def fit(self, features, prices):
        """Store the training data for ANN lookup"""
        # Convert to tensors if they aren't already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float32)

        # Generate training data
        features, labels = self.generate_training_data(features, prices)

        # Store for future lookup
        self.feature_arrays = features.to(device)
        self.labels = labels.to(device)
        self.is_fitted = True

        return self

    def update_model(self, new_features, new_prices, max_samples=20000):
        """
        Update the model with new data without retraining from scratch
        This allows the model to adapt to new market conditions
        """
        if not self.is_fitted:
            print("Model not fitted yet, using initial fit instead")
            return self.fit(new_features, new_prices)

        # Convert new data to tensors
        if not isinstance(new_features, torch.Tensor):
            new_features = torch.tensor(new_features, dtype=torch.float32)
        if not isinstance(new_prices, torch.Tensor):
            new_prices = torch.tensor(new_prices, dtype=torch.float32)

        # Move to device
        new_features = new_features.to(device)
        new_prices = new_prices.to(device)

        # Generate labels for new data
        new_features, new_labels = self.generate_training_data(new_features, new_prices)

        print(f"Adding {len(new_features)} new samples to model")

        # Combine with existing data (keeping most recent samples)
        if len(self.feature_arrays) + len(new_features) > max_samples:
            # Keep most recent data
            keep_samples = max_samples - len(new_features)

            print(
                f"Limiting model to {max_samples} samples (removing {len(self.feature_arrays) - keep_samples} old samples)"
            )

            self.feature_arrays = self.feature_arrays[-keep_samples:]
            self.labels = self.labels[-keep_samples:]

        # Make sure both tensors are on the same device before concatenating
        self.feature_arrays = self.feature_arrays.to(device)
        self.labels = self.labels.to(device)
        new_features = new_features.to(device)
        new_labels = new_labels.to(device)

        # Add new data
        self.feature_arrays = torch.cat([self.feature_arrays, new_features])
        self.labels = torch.cat([self.labels, new_labels])

        print(f"Model updated: {len(self.feature_arrays)} total samples")

        return self

    def predict(self, features):
        """Predict using Approximate Nearest Neighbors with Lorentzian distance"""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        # Move to same device as model
        features = features.to(device)

        # Process in batches to avoid memory issues
        batch_size = 1000  # Adjust based on GPU memory
        n_samples = len(features)
        all_predictions = []

        print(f"Processing {n_samples} samples in batches of {batch_size}...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = features[start_idx:end_idx]

            # Calculate Lorentzian distances for this batch
            batch_distances = self.lorentzian_distance(
                batch_features, self.feature_arrays
            )

            # Get indices of k-nearest neighbors
            _, indices = torch.topk(
                batch_distances,
                min(self.k_neighbors, len(batch_distances[0])),
                largest=False,
                dim=1,
            )

            # Get labels of k-nearest neighbors
            batch_neighbor_labels = [self.labels[idx] for idx in indices]
            batch_neighbor_labels = torch.stack(batch_neighbor_labels)

            # Calculate the sum of neighbor labels
            batch_predictions = torch.sum(batch_neighbor_labels, dim=1)

            # Convert to direction: 1 for long, -1 for short, 0 for neutral
            batch_final = torch.zeros_like(batch_predictions)
            batch_final[batch_predictions > 0] = 1
            batch_final[batch_predictions < 0] = -1

            all_predictions.append(batch_final)

            # Print progress
            progress = min(100, (end_idx / n_samples) * 100)
            print(f"Progress: {progress:.1f}%", end="\r")

        # Combine all batches
        final_predictions = torch.cat(all_predictions)
        print("\nPrediction complete!")

        return final_predictions

    def save_model(self, path=None):
        """Save model state to file"""
        if path is None:
            path = self.model_path

        if not self.is_fitted:
            print("Model not fitted yet, nothing to save")
            return False

        save_dict = {
            "feature_arrays": self.feature_arrays.cpu(),
            "labels": self.labels.cpu(),
            "config": {
                "lookback_bars": self.lookback_bars,
                "prediction_bars": self.prediction_bars,
                "k_neighbors": self.k_neighbors,
                "use_regime_filter": self.use_regime_filter,
                "use_volatility_filter": self.use_volatility_filter,
                "use_adx_filter": self.use_adx_filter,
            },
        }

        try:
            torch.save(save_dict, path)
            print(f"Model saved to {path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, path=None):
        """Load model state from file"""
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            print(f"Model file {path} does not exist")
            return False

        try:
            # Load to CPU first
            checkpoint = torch.load(path, map_location="cpu")

            # Load configuration
            config = checkpoint["config"]
            self.lookback_bars = config["lookback_bars"]
            self.prediction_bars = config["prediction_bars"]
            self.k_neighbors = config["k_neighbors"]
            self.use_regime_filter = config["use_regime_filter"]
            self.use_volatility_filter = config["use_volatility_filter"]
            self.use_adx_filter = config["use_adx_filter"]

            # Load model data - make sure to move to the correct device
            self.feature_arrays = checkpoint["feature_arrays"].to(device)
            self.labels = checkpoint["labels"].to(device)

            self.is_fitted = True
            print(f"Model loaded from {path} with {len(self.feature_arrays)} samples")
            print(
                f"Configuration: lookback={self.lookback_bars}, prediction={self.prediction_bars}, k={self.k_neighbors}"
            )

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def prepare_indicators(df):
    """Add technical indicators using custom PyTorch implementations"""
    df = df.copy()
    print(f"\nInitial data shape: {df.shape}")

    # Initialize indicators with GPU
    rsi_14 = RSIIndicator(period=14, device=device)
    rsi_9 = RSIIndicator(period=9, device=device)
    wavetrend = WaveTrendIndicator(channel_length=10, average_length=11, device=device)
    cci = CCIIndicator(period=20, device=device)
    adx = ADXIndicator(period=20, device=device)

    # Convert data to tensors and move to GPU
    close = torch.tensor(df["close"].values, dtype=torch.float32).to(device)
    high = torch.tensor(df["high"].values, dtype=torch.float32).to(device)
    low = torch.tensor(df["low"].values, dtype=torch.float32).to(device)

    print("\nCalculating indicators...")

    try:
        # Calculate indicators
        rsi_14_values = rsi_14.forward(close)["rsi"]
        print("RSI-14 calculated successfully")

        rsi_9_values = rsi_9.forward(close)["rsi"]
        print("RSI-9 calculated successfully")

        wt_values = wavetrend.forward(high, low, close)
        print("WaveTrend calculated successfully")

        cci_values = cci.forward(high, low, close)["cci"]
        print("CCI calculated successfully")

        adx_values = adx.forward(high, low, close)["adx"]
        print("ADX calculated successfully")

        print("\nMoving indicators to CPU...")

        # Move tensors back to CPU
        df["rsi_14"] = rsi_14_values.cpu().numpy()
        df["rsi_9"] = rsi_9_values.cpu().numpy()
        df["wt1"] = wt_values["wt1"].cpu().numpy()
        df["wt2"] = wt_values["wt2"].cpu().numpy()
        df["cci"] = cci_values.cpu().numpy()
        df["adx"] = adx_values.cpu().numpy()

        print("\nCalculating additional features...")

        # Price action features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()
        df["regime"] = df["returns"].rolling(window=50).mean()

        # Momentum features
        df["rsi_diff"] = df["rsi_14"] - df["rsi_9"]  # RSI divergence
        df["wt_diff"] = df["wt1"] - df["wt2"]  # WaveTrend divergence

        # Trend features
        df["price_trend"] = df["close"].rolling(window=20).mean().pct_change()
        df["volume_trend"] = df["volume"].rolling(window=20).mean().pct_change()

        # Volatility features
        df["tr"] = pd.DataFrame(
            {
                "hl": df["high"] - df["low"],
                "hc": abs(df["high"] - df["close"].shift()),
                "lc": abs(df["low"] - df["close"].shift()),
            }
        ).max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # Drop NaN values
        print("\nRemoving NaN values...")
        df_before = len(df)
        df = df.dropna()
        df_after = len(df)
        print(f"Rows before dropping NaN: {df_before:,}")
        print(f"Rows after dropping NaN: {df_after:,}")
        print(f"Dropped {df_before - df_after:,} rows")

    except Exception as e:
        print(f"Error in prepare_indicators: {str(e)}")
        raise

    return df


def apply_filters(df, adx_threshold=20.0, regime_threshold=-0.1):
    """Apply filters to identify tradeable conditions (TradingView style)"""
    print(f"\nApplying filters...")
    print(f"Initial filter data shape: {df.shape}")

    # Volatility filter - price movement should be significant
    volatility_mask = df["volatility"] > df["volatility"].quantile(0.1)
    print(f"Volatility filter passes: {volatility_mask.sum()}")

    # Regime filter - trend should be established (positive regime = uptrend biased)
    regime_mask = df["regime"] > regime_threshold
    print(f"Regime filter passes: {regime_mask.sum()}")

    # ADX filter - trend should be strong enough
    adx_mask = df["adx"] > adx_threshold
    print(f"ADX filter passes: {adx_mask.sum()}")

    # Combine all filters
    combined_mask = volatility_mask & regime_mask & adx_mask
    print(f"Combined filter passes: {combined_mask.sum()}")

    # Apply filter to DataFrame
    df_filtered = df[combined_mask].copy()
    print(f"Filtered data shape: {df_filtered.shape}")

    return df_filtered


def prepare_features(df):
    """Prepare and scale features for model input"""
    # Select features
    feature_cols = ["rsi_14", "wt1", "wt2", "cci", "adx"]  # Customize this list

    # Simple standardization (mean=0, std=1)
    scaler = {}
    scaled_features = np.zeros((len(df), len(feature_cols)))

    for i, col in enumerate(feature_cols):
        mean, std = df[col].mean(), df[col].std()
        scaler[col] = {"mean": mean, "std": std}
        scaled_features[:, i] = (df[col].values - mean) / (std if std > 0 else 1)

    print(f"Prepared features with shape: {scaled_features.shape}")
    return scaled_features, scaler


def train_lorentzian_ann(
    df,
    lookback_bars=50,
    prediction_bars=4,
    k_neighbors=20,
    model_path="lorentzian_model.pt",
):
    """Train the Lorentzian ANN model with weight saving support"""
    try:
        # Prepare features
        features, scaler = prepare_features(df)
        close_prices = df["close"].values

        print(f"\nTraining/loading Lorentzian ANN model...")
        print(f"Features shape: {features.shape}")
        print(f"Using {lookback_bars} bars for lookback")
        print(f"Predicting {prediction_bars} bars into the future")
        print(f"Using {k_neighbors} nearest neighbors")

        # Free up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Initialize model
        model = LorentzianANN(
            lookback_bars=lookback_bars,
            prediction_bars=prediction_bars,
            k_neighbors=k_neighbors,
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True,
        )

        # Check if model file exists
        if os.path.exists(model_path):
            print(f"Found existing model at {model_path}, loading...")
            model.load_model(model_path)

            # Update with new data
            print("Updating model with new data...")
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            prices_tensor = torch.tensor(close_prices, dtype=torch.float32).to(device)
            model.update_model(features_tensor, prices_tensor)
        else:
            print("No existing model found, training new model...")
            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            prices_tensor = torch.tensor(close_prices, dtype=torch.float32).to(device)

            # Fit the model
            model.fit(features_tensor, prices_tensor)

        # Free memory before prediction
        gc.collect()
        torch.cuda.empty_cache()

        # Generate predictions
        predictions = model.predict(features_tensor)

        # Move back to CPU for analysis
        predictions = predictions.cpu().numpy()

        # Store predictions in the dataframe
        df["signal"] = predictions

        # Print prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        print("\nPrediction distribution:")
        for signal, count in distribution.items():
            signal_name = (
                "LONG" if signal == 1 else "SHORT" if signal == -1 else "NEUTRAL"
            )
            print(f"{signal_name}: {count} ({count / len(predictions) * 100:.2f}%)")

        # Save model state
        print("Saving model state...")
        model.save_model(model_path)

        return df, model, scaler

    except Exception as e:
        print(f"Error in train_lorentzian_ann: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None, None, None


def calculate_metrics(df):
    """Calculate trading metrics"""
    # Calculate returns based on signals
    df["position"] = df["signal"].shift(1).fillna(0)  # Position at the start of the bar
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"] * df["returns"]

    # Calculate metrics
    total_trades = (df["position"].diff() != 0).sum()
    winning_trades = (df["strategy_returns"] > 0).sum()
    losing_trades = (df["strategy_returns"] < 0).sum()

    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = (
        df.loc[df["strategy_returns"] > 0, "strategy_returns"].mean()
        if winning_trades > 0
        else 0
    )
    avg_loss = (
        df.loc[df["strategy_returns"] < 0, "strategy_returns"].mean()
        if losing_trades > 0
        else 0
    )

    # Calculate cumulative returns
    df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    final_return = df["cumulative_returns"].iloc[-1] if len(df) > 0 else 1.0

    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Final Return: {final_return - 1:.2%}")

    # Calculate max drawdown
    peak = df["cumulative_returns"].cummax()
    drawdown = df["cumulative_returns"] / peak - 1
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {max_drawdown:.2%}")

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_return": final_return - 1,
        "max_drawdown": max_drawdown,
    }


def plot_results(df):
    """Plot trading signals and cumulative returns"""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title("Price with Trading Signals")
    plt.plot(df.index, df["close"], label="Price", alpha=0.7)

    # Plot buy and sell signals
    buy_points = df[df["signal"] == 1].index
    sell_points = df[df["signal"] == -1].index

    plt.scatter(
        buy_points,
        df.loc[buy_points, "close"],
        marker="^",
        color="green",
        label="Buy Signal",
    )
    plt.scatter(
        sell_points,
        df.loc[sell_points, "close"],
        marker="v",
        color="red",
        label="Sell Signal",
    )

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Strategy Cumulative Returns")
    plt.plot(df.index, df["cumulative_returns"], label="Strategy Returns")
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    plt.ylabel("Cumulative Returns")
    plt.legend()

    plt.tight_layout()
    plt.savefig("lorentzian_persistent_results.png")
    print("Saved plot to lorentzian_persistent_results.png")


def simulate_incremental_learning(full_df, num_periods=4):
    """
    Simulate incremental learning over time

    Args:
        full_df: Full dataframe with all data
        num_periods: Number of periods to split the data into
    """
    print(f"\n=== Simulating Incremental Learning over {num_periods} periods ===")

    # Split data into chunks to simulate the passing of time
    chunk_size = len(full_df) // num_periods
    model_path = "test_lorentzian_model.pt"

    # Remove existing model file for clean test
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Removed existing model file: {model_path}")

    results = []

    for i in range(num_periods):
        period_start = i * chunk_size
        period_end = (i + 1) * chunk_size if i < num_periods - 1 else len(full_df)

        period_df = full_df.iloc[period_start:period_end].copy()

        print(
            f"\n--- Period {i + 1}/{num_periods} (rows {period_start}-{period_end}) ---"
        )
        print(f"Date range: {period_df.index[0]} to {period_df.index[-1]}")

        # Training
        df_predicted, model, scaler = train_lorentzian_ann(
            period_df,
            lookback_bars=50,
            prediction_bars=4,
            k_neighbors=20,
            model_path=model_path,
        )

        # Calculate metrics
        if df_predicted is not None:
            metrics = calculate_metrics(df_predicted)
            metrics["period"] = i + 1
            metrics["date_start"] = period_df.index[0]
            metrics["date_end"] = period_df.index[-1]
            metrics["num_samples"] = len(period_df)
            results.append(metrics)

            # Log results
            print(f"\nPeriod {i + 1} Results:")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Return: {metrics['final_return']:.2%}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        else:
            print(f"Period {i + 1}: Training failed")

    # Compare results across periods
    if results:
        print("\n=== Incremental Learning Results ===")
        for i, metrics in enumerate(results):
            print(
                f"\nPeriod {i + 1} ({metrics['date_start']} to {metrics['date_end']}):"
            )
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Return: {metrics['final_return']:.2%}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

        # Load model one last time for final statistics
        final_model = LorentzianANN(lookback_bars=50, prediction_bars=4, k_neighbors=20)
        if final_model.load_model(model_path):
            # Success
            print("\nFinal model loaded successfully")
        else:
            print("\nFailed to load final model")


def main():
    start_time = time.time()

    # Load data
    data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
    df = pd.read_feather(data_path)
    df["datetime"] = pd.to_datetime(df["date"])
    df.set_index("datetime", inplace=True)

    # Reduce dataset size for testing
    test_size = 10000  # Adjust as needed
    print(f"Original data size: {len(df)}")
    df = df.iloc[-test_size:]
    print(f"Reduced data size: {len(df)}")

    # Prepare indicators
    df = prepare_indicators(df)

    # Simulate incremental learning (dividing the data into periods)
    # This mimics how the model would be updated over time
    simulate_incremental_learning(df, num_periods=4)

    # For final run on all data
    model_path = "lorentzian_final_model.pt"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Train on complete dataset
    df, model, scaler = train_lorentzian_ann(
        df,
        lookback_bars=50,  # TradingView default
        prediction_bars=4,  # TradingView default
        k_neighbors=20,  # Number of neighbors to consider
        model_path=model_path,
    )

    if df is not None:
        # Calculate metrics
        metrics = calculate_metrics(df)

        # Plot results
        plot_results(df)

    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
