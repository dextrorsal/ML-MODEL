# ML-Enhanced Trading Strategy Documentation

## Overview
This strategy combines three key components in a specific hierarchy to create a robust trading system:

1. **Primary Signal (Lorentzian Classification)**
   - Main decision maker for trade entries
   - Identifies high-probability setups using ML
   - Uses multiple technical features for pattern recognition

2. **Secondary Confirmation (Logistic Regression)**
   - Validates Lorentzian signals
   - Provides probability score for trade confidence
   - Must align with primary signal for entry

3. **Risk Management (Chandelier Exit)**
   - Manages position risk once trade is executed
   - Provides trailing stop mechanism
   - Can influence entry timing but not primary decision maker

## Position Sizing and Leverage

- **Leverage Range**: 10-20x
- **Initial Position**: Single entry, no scaling
- **Risk Management**: Dynamic based on market conditions
- **Stop Loss**: Determined by Chandelier Exit indicator

## Component Details

### 1. Lorentzian Classification (Primary Signal)

#### Core Features
- RSI (14, 1)
- Wave Trend (10, 11)
- CCI (20, 1)
- ADX (20, 2)
- RSI (9, 1)

#### Signal Generation
- K-Nearest Neighbors approach (k=8)
- Chronological sampling (4 bar spacing)
- Dynamic threshold adjustment
- Neighbor voting for predictions

#### Filters
- Volatility Filter (ATR-based)
- Regime Filter
- ADX Filter
- Optional: EMA, SMA, Kernel Regression

### 2. Logistic Regression (Signal Confirmation)

#### Purpose
- Validates Lorentzian signals
- Provides probability score (0-1)
- Must exceed threshold for trade confirmation

#### Integration Rules
1. **Long Entry Requirements**
   - Lorentzian signals long
   - Logistic probability > 0.7
   - All filters passing

2. **Short Entry Requirements**
   - Lorentzian signals short
   - Logistic probability < 0.3
   - All filters passing

### 3. Chandelier Exit (Risk Management)

#### Implementation
- ATR-based trailing stop
- Volatility-adjusted stop distances
- Dynamic position sizing based on market conditions

## Trade Execution Flow

### Entry Process
1. **Primary Signal Detection**
   - Lorentzian classifier identifies potential setup
   - Check all filter conditions
   - Verify pattern strength

2. **Signal Confirmation**
   - Wait for Logistic Regression confirmation
   - Check probability threshold
   - Verify alignment with Lorentzian

3. **Position Execution**
   - Enter full position at once
   - Set Chandelier Exit stop
   - Document entry conditions

### Management Process
1. **Active Position Monitoring**
   - Track Chandelier Exit levels
   - Monitor signal strength
   - Watch for reversal signals

2. **Exit Conditions**
   - Primary: Chandelier Exit stop hit
   - Secondary: Signal reversal from both indicators

## Risk Management Protocol

### Position Management

- **Single Entry Approach**
  - Enter full position at once
  - No scaling in or out
  - Exit based on Chandelier Exit or signal reversal

### Chandelier Exit Integration

- **Stop Loss Placement**
  - Calculate Chandelier Exit stop (ATR-based)
  - Adjust based on market volatility
  - Trail stop as position moves in favor

### Market Adaptation

- Position sizing and risk parameters adapt to:
  - Current market volatility
  - Trading session characteristics
  - Overall market regime

## Model Training Focus

### Primary Objectives
1. **Signal Accuracy**
   - Optimize Lorentzian feature selection
   - Fine-tune Logistic confirmation thresholds
   - Minimize false signals

2. **Entry Timing**
   - Learn optimal entry points between signals
   - Consider Chandelier Exit levels
   - Factor in volatility conditions

3. **Exit Optimization**
   - Balance between Chandelier and signal exits
   - Learn from historical profit opportunities
   - Optimize trailing stop distances

### Training Priorities
1. **Pattern Recognition**
   - Train on high-probability setups
   - Focus on signal alignment
   - Learn volatility patterns

2. **Risk Management**
   - Optimize stop distances
   - Learn position sizing adjustments
   - Identify optimal scaling points

3. **Performance Metrics**
   - Win rate target: 50-60%
   - Profit factor target: >1.5
   - Average win/loss ratio: >1.8

## Implementation Status

### Completed
- [x] Basic feature calculations
- [x] Lorentzian distance metric
- [x] KNN prediction logic
- [x] Chandelier Exit implementation

### In Progress
- [ ] Logistic Regression integration
- [ ] Signal alignment optimization
- [ ] Position sizing automation
- [ ] Exit optimization

### To Do
- [ ] Model training pipeline
- [ ] Performance tracking system
- [ ] Risk management automation
- [ ] Strategy backtesting

## Next Steps
1. Finalize model training parameters
2. Implement combined signal validation
3. Test position sizing automation
4. Develop performance tracking

*Note: This is a living document that will be updated as we optimize the strategy through training and testing.*

## Key Differences from Pine Script
1. **Data Processing**
   - Pine: Real-time bar-by-bar processing
   - Python: Vectorized calculations where possible

2. **Feature Engineering**
   - Need to verify feature calculations match exactly
   - Ensure proper normalization of features

3. **Signal Generation**
   - Need to implement chronological spacing
   - Verify neighbor selection logic

## Questions to Address
1. Do you want to keep all the filter options from the Pine Script?
2. Should we implement the trade statistics tracking?
3. Do you want to maintain the same visualization features?
4. Should we keep the kernel regression component?

*Note: This document will be updated as we progress with the implementation.*

## Components Analysis Status

### 1. Lorentzian Classification
- [x] Initial Python implementation available
- [ ] Need to compare with Pine Script version
- [ ] Verify signal generation accuracy
- [ ] Optimize parameters

### 2. Logistic Regression
- [x] Initial Python implementation available
- [ ] Need to compare with Pine Script version
- [ ] Verify signal generation accuracy
- [ ] Optimize parameters

### 3. Chandelier Exit
- [ ] Need Pine Script reference
- [ ] Implement in Python
- [ ] Integrate with signal generation
- [ ] Test risk management rules

## Implementation Plan

### Phase 1: Analysis & Comparison
1. Compare each Python implementation with its Pine Script counterpart
2. Document any differences or missing features
3. Ensure accuracy of signal generation
4. Validate against historical trades

### Phase 2: Integration
1. Combine signal generators (Lorentzian + Logistic)
2. Implement Chandelier Exit for position management
3. Create unified strategy class
4. Add risk management rules

### Phase 3: Backtesting
1. Set up proper backtesting framework
2. Test individual components
3. Test combined strategy
4. Optimize parameters

### Phase 4: Optimization
1. Fine-tune signal generation
2. Optimize position sizing
3. Refine exit rules
4. Minimize drawdown

## Strategy Logic (Draft)

### Entry Rules
1. Primary Signal: Lorentzian Classification
   - Wait for confirmed signal
   - Check signal strength

2. Confirmation: Logistic Regression
   - Must align with Lorentzian signal
   - Check probability threshold

### Exit Rules
1. Chandelier Exit
   - Primary exit mechanism
   - Trailing stop calculation
   - Volatility-based adjustments

2. Additional Exit Conditions
   - Signal reversal
   - Time-based exits
   - Profit targets

### Risk Management
1. Position Sizing
   - Based on Chandelier Exit distance
   - Account % risk per trade

2. Stop Loss Rules
   - Initial stop from Chandelier
   - Trailing stop adjustments
   - Break-even rules

## Notes
- Need to maintain separate signal generation and execution logic
- Consider market conditions and volatility
- Plan for proper position sizing and risk management
- Document all parameters and their effects

## Questions to Address
1. How should signals from both indicators be weighted?
2. What are the exact entry/exit confirmation rules?
3. How to handle conflicting signals?
4. What are the position sizing rules?
5. How to integrate volatility into risk management?

---
*This is a living document and will be updated as we progress with the implementation.* 