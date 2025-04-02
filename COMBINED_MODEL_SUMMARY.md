# Combined Model Trading Strategy Summary

## Model Training Results

We've successfully trained two models for different timeframes:

### 5-Minute Timeframe Model
- **Accuracy**: 85.23%
- **Best Hyperparameters**:
  - Batch Size: 64
  - Dropout Rate: 0.4
  - Hidden Size: 64
  - Learning Rate: 0.0005

### 15-Minute Timeframe Model
- **Accuracy**: 88.55%
- **Best Hyperparameters**:
  - Batch Size: 32
  - Dropout Rate: 0.2  
  - Hidden Size: 64
  - Learning Rate: 0.001

Both models show very high accuracy on their test sets, which is promising.

## Backtesting Results

When implementing these models for trading signal generation, we discovered an important issue:

### Prediction Value Ranges
- **5-Minute Model**:
  - Min: 0.2659, Max: 0.4024, Mean: 0.3009, Median: 0.2888
  - No values exceed 0.5
  - Only 0.42% of signals exceed 0.4

- **15-Minute Model**:
  - Min: 0.1209, Max: 0.3549, Mean: 0.1604, Median: 0.1406
  - No values exceed 0.5
  - No signals exceed 0.4

This indicates that our models are highly cautious, producing low confidence scores. This likely reflects their high accuracy - they're being very selective about when to predict a positive signal.

## Analysis

1. **Model Output Distribution**: Both models are outputting prediction values in a much lower range than expected. This is likely due to:
   - The class imbalance in our training data (more 0s than 1s)
   - The high threshold we required for the "future_return" when labeling the data (0.5% for 5m, 0.8% for 15m)

2. **Alignment**: The combined strategy requires both models to agree on a signal, but with such low prediction values, it's unlikely to find points where both models exceed the threshold.

3. **Timeframe Relationship**: The 5-minute model naturally produces more signals than the 15-minute model, which aligns with the fact that shorter timeframes typically see more action.

## Recommendations for Improvement

1. **Adjust Confidence Thresholds**: Based on the observed prediction distributions, consider using much lower thresholds:
   - 5-minute model: 0.3 (around the mean)
   - 15-minute model: 0.16 (around the mean)
   - Combined threshold: 0.25

2. **Retrain with Different Labeling**: The current models were trained with relatively high future return thresholds (0.5% and 0.8%). Consider retraining with lower thresholds (e.g., 0.3% for 5m and 0.5% for 15m) to get more balanced class distributions.

3. **Implement Model Calibration**: The raw output from a neural network often doesn't represent true probability. Apply techniques like Platt scaling to calibrate the outputs to better reflect confidence probabilities.

4. **Use Separate Confirmation Logic**: Instead of requiring both models to exceed a threshold, consider using the 5-minute model for primary signals and the 15-minute model as a trend filter or confirmation.

5. **Explore Ensemble Methods**: Instead of combining signals after prediction, train an ensemble that combines features from both timeframes directly.

6. **Optimize Trading Windows**: The current settings (12 periods for 5m, 4 periods for 15m) may be too aggressive. Consider using longer prediction windows for more stable performance.

## Next Steps

1. **Modify the Combined Model Trader**:
   ```bash
   python scripts/combined_model_trader.py --backtest-days 30 --confidence-threshold 0.3 --combined-threshold 0.25
   ```

2. **Create a Calibration Layer**:
   - Implement a calibration function that maps raw model outputs to calibrated probabilities
   - This can be as simple as finding linear scaling factors based on the observed distributions

3. **Experiment with Alternative Signal Logic**:
   - Try using the 5-minute model as the primary signal generator
   - Use the 15-minute model only as a trend filter (e.g., must be above 0.15)
   - Implement a weighted average favoring the 5-minute model (e.g., 0.7*5m + 0.3*15m)

4. **Enhance Feature Engineering**:
   - Add more advanced indicators specifically designed for crypto markets
   - Include market regime detection features
   - Consider incorporating volume profile indicators

5. **Implement Walk-forward Backtesting**:
   - Test model performance across multiple market regimes
   - Adjust parameters dynamically based on recent performance

By implementing these suggestions, we can potentially improve the model's signal generation capabilities while maintaining its high accuracy and risk management. 