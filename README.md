# NILM: Non-Intrusive Load Monitoring System

A "near-perfect" Non-Intrusive Load Monitoring (NILM) system for energy disaggregation using Sequence-to-Point (S2P) architecture with a Hybrid CNN-BiLSTM-Attention neural network.

## Overview

This NILM system disaggregates aggregate power consumption (`Total_ActivePower(W)`) into individual appliance power traces for:
- Air Conditioner
- Refrigerator
- Fan
- Washing Machine
- EV Charger

## Architecture

### Sequence-to-Point (S2P) Paradigm
Based on Zhang et al., the model:
- Inputs a window of 99 samples of the 'Mains' signal
- Predicts ONLY the central midpoint sample (t=50) for each appliance
- This eliminates "hallucinated" ghosting common in standard Bi-LSTMs

### Hybrid Multi-Task Neural Network
1. **Feature Extraction**: 1D-CNN layers identify sub-second "transients" (startup spikes)
2. **Temporal Modeling**: Bidirectional LSTM (128 units) learns "cycles" (e.g., Refrigerator compressor duration)
3. **Attention Mechanism**: Self-Attention layer weights "switching events" more heavily than steady-state noise
4. **Dual-Head Outputs**:
   - **Regression Head**: Predicts Active Power (Watts) with ReLU for zero-bounded output
   - **Classification Head**: Predicts Operating State (Multi-class)

### Multi-State Classification
| Appliance | States |
|-----------|--------|
| Fan | OFF, Low-Speed, Medium-Speed, High-Speed |
| Washing Machine | OFF, Wash, Rinse, Spin |
| Air Conditioner | OFF, ON |
| Refrigerator | OFF, ON |
| EV Charger | OFF, ON |

## Advanced Feature Engineering

The model uses 4 input channels:
1. **Standardized Mains Power** (RobustScaler)
2. **Differential Power** (`Mains.diff()`) - captures edge transients
3. **Second-derivative of Power** - captures "switching acceleration"
4. **Rolling 5-sample Standard Deviation** - identifies appliance "noise textures"

## Loss Functions & Optimization

- **Regression**: Huber Loss (robust to scale differences: 3000W EV Charger vs 20W Fan)
- **Classification**: Weighted Categorical Cross-Entropy (handles "Mostly-OFF" data imbalance)

## Post-Processing

- 3-tap Median Filter for smoothing
- Hard 10W noise floor for clean traces

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### With Real Data
```bash
python nilm_s2p_model.py --data combined_appliances_correct_order.xlsx --output results
```

### Demo Mode (Synthetic Data)
```bash
python nilm_s2p_model.py --demo
```

## Output

The system generates:
1. **Performance Summary Table**: MAE (Watts), F1-Score (States), Energy Accuracy (%)
2. **Visualizations**:
   - Side-by-side Regression Plots (Actual vs Predicted Power)
   - Labeled Multi-Class Confusion Matrices for all appliances

Results are saved to the specified output directory:
- `performance_summary.csv` - Metrics table
- `training_history.csv` - Training loss history
- `{appliance}_regression.png` - Regression plots
- `{appliance}_confusion_matrix.png` - Confusion matrices
- `performance_summary.png` - Summary visualization
- `best_model.keras` - Trained model checkpoint

## Requirements

- TensorFlow >= 2.10.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- openpyxl >= 3.0.10
- scipy >= 1.9.0

## References

- Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. In AAAI Conference on Artificial Intelligence.
