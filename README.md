# Production-Ready NILM System with BiLSTM and Multiple Model Architectures

A comprehensive Non-Intrusive Load Monitoring (NILM) system for energy disaggregation with multiple deep learning model architectures.

## Overview

This system implements state-of-the-art deep learning models for disaggregating household energy consumption into individual appliance usage. It features:

- **Multiple Model Architectures**: CNN-BiLSTM-Attention, Transformer, UNet, ResNet-LSTM
- **Advanced Feature Engineering**: 8-channel feature extraction including derivatives, rolling statistics, and higher-order moments
- **Dual-Head Output**: Simultaneous power regression and state classification
- **Robust Training**: Focal loss, combined regression loss, learning rate scheduling, early stopping
- **Comprehensive Evaluation**: Multiple metrics, confusion matrices, visualizations
- **Production-Ready**: Clean code, type hints, logging, configuration management

## Features

### Supported Appliances

1. **Air Conditioner** (Binary: OFF/ON)
2. **Refrigerator** (Binary: OFF/ON)
3. **Fan** (Multi-state: OFF/Low/Medium/High)
4. **Washing Machine** (Multi-state: OFF/Wash/Rinse/Spin)
5. **EV Charger** (Binary: OFF/ON)

### Data Processing Pipeline

- Load data from Excel/CSV files
- Handle missing values with forward/backward fill
- Clip negative power values
- Compute comprehensive statistics
- 8-channel feature engineering:
  1. Standardized mains power (RobustScaler)
  2. First derivative (power transitions)
  3. Second derivative (switching acceleration)
  4. Rolling mean (smoothed signal)
  5. Rolling standard deviation (local variance)
  6. Rolling range (local dynamics)
  7. Rolling skewness (asymmetry)
  8. Rolling kurtosis (tail behavior)
- Sequence-to-Point windowing (window_size=99)
- Time-series aware train/val/test split (70/15/15)

### Model Architectures

#### 1. CNN-BiLSTM-Attention (Primary Model)
- Multi-scale CNN with parallel kernels (3, 5, 7, 9)
- Squeeze-and-Excitation blocks for channel attention
- Bidirectional LSTM layers (128, 64 units)
- Multi-Head Self-Attention (4 heads)
- Positional encoding
- Dual heads for power regression and state classification

#### 2. Transformer-based
- Patch embedding
- Multiple transformer encoder blocks
- Multi-head attention
- Feed-forward networks
- Layer normalization

#### 3. UNet-style
- Encoder-decoder architecture
- Skip connections
- Dilated convolutions
- Multi-scale feature aggregation

#### 4. ResNet-LSTM Hybrid
- Residual blocks with skip connections
- BiLSTM layers
- Global pooling

### Loss Functions

- **Power Regression**: Combined MAE + Huber + MSE loss
- **State Classification**: Focal Loss with class weights (handles imbalance)

### Training Features

- Adam optimizer with gradient clipping (clipnorm=1.0)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience=30)
- Model checkpointing
- CSV logging
- Balanced class weights

### Post-Processing

- Median filtering (size=5)
- Gaussian smoothing (sigma=1.0)
- Noise floor thresholding per appliance
- Physical bounds clipping
- State refinement based on power values

### Evaluation Metrics

- **Power Regression**: MAE, RMSE, Relative MAE, Energy Accuracy
- **State Classification**: F1-Score, Accuracy, Precision, Recall, Balanced Accuracy
- **Visualizations**: Training history, disaggregation plots, confusion matrices, energy comparison

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- OpenPyXL

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from nilm_system import NILMSystem
from config import NILMConfig

# Create configuration
config = NILMConfig()

# Create and run NILM system
system = NILMSystem(config)
system.run(data_file='combined_appliances_correct_order.xlsx')
```

### Custom Configuration

```python
from nilm_system import NILMSystem
from config import NILMConfig, ApplianceConfig

# Create custom configuration
config = NILMConfig()

# Modify settings
config.model_type = 'transformer'  # or 'unet', 'resnet_lstm'
config.model.epochs = 100
config.model.batch_size = 256
config.data.window_size = 199

# Run system
system = NILMSystem(config)
system.run()
```

### Command Line

```bash
python nilm_system.py
```

## Project Structure

```
nilm/
├── config.py                 # Configuration classes
├── nilm_system.py           # Main NILM pipeline
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── cnn_bilstm_attention.py
│   ├── transformer.py
│   ├── unet.py
│   └── resnet_lstm.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   ├── post_processing.py
│   └── visualization.py
└── outputs/               # Generated outputs
    ├── models/           # Saved models (.keras)
    ├── plots/            # Visualizations (.png)
    ├── metrics/          # Metrics and logs (.csv)
    ├── predictions/      # Predictions (.csv)
    └── config.json       # Configuration snapshot
```

## Output Structure

After running the system, the following outputs are generated:

```
outputs/
├── models/
│   ├── AC_model_best.keras
│   ├── Refrigerator_model_best.keras
│   ├── Fan_model_best.keras
│   ├── Washing_Machine_model_best.keras
│   └── EV_Charger_model_best.keras
├── plots/
│   ├── AC_training_history.png
│   ├── Refrigerator_training_history.png
│   ├── Fan_training_history.png
│   ├── Washing_Machine_training_history.png
│   ├── EV_Charger_training_history.png
│   ├── disaggregation.png
│   ├── confusion_matrices.png
│   ├── energy_comparison.png
│   └── power_distributions.png
├── metrics/
│   ├── metrics.csv
│   ├── aggregated_metrics.csv
│   ├── AC_training_log.csv
│   ├── Refrigerator_training_log.csv
│   ├── Fan_training_log.csv
│   ├── Washing_Machine_training_log.csv
│   └── EV_Charger_training_log.csv
├── predictions/
│   ├── AC_predictions.csv
│   ├── Refrigerator_predictions.csv
│   ├── Fan_predictions.csv
│   ├── Washing_Machine_predictions.csv
│   └── EV_Charger_predictions.csv
└── config.json
```

## Expected Performance

The system is designed to achieve:

- **Average MAE**: < 50W across appliances
- **Average F1-Score**: > 0.85
- **Average Energy Accuracy**: > 80%
- **State Classification Accuracy**: > 85%

Actual performance depends on the quality and characteristics of the input data.

## Configuration Options

### Data Configuration

- `data_file`: Path to input data file (Excel/CSV)
- `window_size`: Size of input window (default: 99)
- `midpoint`: Position of target in window (default: 49)
- `train_ratio`: Training set ratio (default: 0.70)
- `val_ratio`: Validation set ratio (default: 0.15)
- `test_ratio`: Test set ratio (default: 0.15)

### Model Configuration

- `model_type`: Model architecture ('cnn_bilstm_attention', 'transformer', 'unet', 'resnet_lstm')
- `cnn_filters`: Number of CNN filters (default: 64)
- `lstm_units`: LSTM units per layer (default: [128, 64])
- `attention_heads`: Number of attention heads (default: 4)
- `batch_size`: Training batch size (default: 128)
- `epochs`: Maximum training epochs (default: 200)
- `learning_rate`: Initial learning rate (default: 0.001)
- `patience`: Early stopping patience (default: 30)

### Post-Processing Configuration

- `median_filter_size`: Median filter kernel size (default: 5)
- `gaussian_sigma`: Gaussian smoothing sigma (default: 1.0)
- `apply_physical_bounds`: Apply physical bounds clipping (default: True)

## Appliance Configuration

Each appliance can be configured with:

- `name`: Display name
- `type`: 'binary' or 'multi_state'
- `thresholds`: Power thresholds for state classification (list)
- `state_names`: Names for each state (list)
- `noise_floor`: Minimum power threshold (float)

Example:

```python
from config import ApplianceConfig

ac_config = ApplianceConfig(
    name='Air Conditioner',
    type='binary',
    thresholds=[80],
    state_names=['OFF', 'ON'],
    noise_floor=10.0
)
```

## Advanced Features

### Custom Loss Functions

The system includes custom loss functions:

- **CombinedRegressionLoss**: Weighted combination of MAE, Huber, and MSE
- **FocalLoss**: Handles class imbalance in state classification

### Attention Mechanisms

- **Squeeze-and-Excitation**: Channel attention for CNN features
- **Multi-Head Self-Attention**: Temporal attention for sequence modeling

### Feature Engineering

The 8-channel feature engineering captures multiple aspects of power signal:

1. **Standardized power**: Normalized baseline signal
2. **First derivative**: Detects power transitions (ON/OFF events)
3. **Second derivative**: Captures switching acceleration
4. **Rolling mean**: Smoothed signal for trend analysis
5. **Rolling std**: Local variance for volatility detection
6. **Rolling range**: Local dynamics and peak-to-peak variation
7. **Rolling skewness**: Asymmetry in power distribution
8. **Rolling kurtosis**: Tail behavior for outlier detection

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `window_size`
2. **Poor Performance**: Increase `epochs`, adjust `learning_rate`, or try different `model_type`
3. **Data Loading Error**: Ensure data file path is correct and format is supported (Excel/CSV)

### Logging

The system provides comprehensive logging. Check console output for:

- Data loading and preprocessing status
- Model architecture details
- Training progress (loss, metrics)
- Evaluation results
- File save locations

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Write clean, documented code
2. Follow PEP 8 style guidelines
3. Add type hints
4. Include docstrings
5. Test your changes

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this system in your research, please cite:

```
NILM System with Multiple Deep Learning Architectures
Non-Intrusive Load Monitoring for Energy Disaggregation
2024
```

## Contact

For questions or issues, please open an issue on the repository.

## Acknowledgments

This system implements techniques from various research papers on NILM and deep learning, including:

- Sequence-to-Point learning for NILM
- Attention mechanisms for time series
- Multi-task learning for power and state prediction
- Advanced feature engineering for energy signals

## Version History

- **v1.0.0** (2024): Initial release with four model architectures
