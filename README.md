# NILM - Non-Intrusive Load Monitoring with Deep Learning

A comprehensive Python package for **Non-Intrusive Load Monitoring (NILM)** / Energy Disaggregation using deep learning models including **BiLSTM**, **CNN**, **Seq2Point**, and **Seq2Seq** architectures.

## Overview

NILM is a technique for estimating individual appliance power consumption from aggregate whole-house electricity measurements. This package provides ready-to-use implementations of state-of-the-art deep learning models for energy disaggregation.

## Features

- **Multiple Model Architectures:**
  - **BiLSTM** - Bidirectional LSTM with optional convolutional feature extraction
  - **CNN** - 1D Convolutional Neural Network (Kelly & Knottenbelt architecture)
  - **Seq2Point** - Sequence-to-Point model (Zhang et al., 2018)
  - **Seq2Seq** - Sequence-to-Sequence encoder-decoder model
  - Plus variants: BiLSTM-Seq2Seq, DeepCNN, Seq2Point-Attention, ConvSeq2Seq

- **Data Handling:**
  - NILMTK dataset support (UK-DALE, REDD, etc.)
  - CSV file loading
  - Appliance-specific normalization
  - Sliding window creation
  - Data augmentation

- **Evaluation Metrics:**
  - MAE, RMSE (regression metrics)
  - SAE, NDE, EAC (energy metrics)
  - F1 Score, Precision, Recall (on/off detection)

- **Visualization:**
  - Training history plots
  - Prediction vs ground truth
  - Model comparison charts

## Installation

```bash
# Clone the repository
git clone https://github.com/arjunsunil19/nilm.git
cd nilm

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- tensorflow >= 2.8.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0 (for visualization)
- nilmtk >= 0.4.0 (optional, for NILMTK datasets)
- pyyaml >= 5.4.0

## Quick Start

```python
from nilm.models import BiLSTM
from nilm.data import DataLoader, DataPreprocessor
from nilm.utils import MetricsCalculator
import numpy as np

# Generate sample data (replace with real data)
data_loader = DataLoader()
X, y = data_loader.generate_synthetic_data(
    num_samples=1000,
    window_size=599,
    appliance="kettle"
)

# Split data
preprocessor = DataPreprocessor(window_size=599)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

# Create and train model
model = BiLSTM(
    appliance_name="kettle",
    window_size=599,
    lstm_units=[128, 64]
)
model.train(X_train, y_train, epochs=50, batch_size=64)

# Evaluate
y_pred = model.predict(X_test)
metrics = MetricsCalculator("kettle")
results = metrics.calculate_all(y_test, np.clip(y_pred, 0, None))
metrics.print_metrics(results)
```

## Usage

### Training with Command Line

```bash
# Train BiLSTM on synthetic data
python src/train.py --model bilstm --appliance kettle --synthetic --epochs 50

# Train CNN with custom configuration
python src/train.py --model cnn --appliance microwave --config config.yaml

# Train Seq2Point on real data
python src/train.py --model seq2point --appliance fridge --data-path path/to/data.h5
```

### Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `bilstm` | Bidirectional LSTM | General purpose, captures temporal patterns |
| `bilstm_seq2seq` | BiLSTM Seq2Seq | Full sequence output |
| `cnn` | 1D CNN | Fast training, local patterns |
| `deep_cnn` | Deeper CNN with residuals | Complex patterns |
| `seq2point` | Sequence-to-Point | Midpoint prediction |
| `seq2point_attention` | Seq2Point + Attention | Focus on relevant timesteps |
| `seq2seq` | LSTM Encoder-Decoder | Sequence-to-sequence |
| `conv_seq2seq` | Dilated Conv Seq2Seq | Fast sequence output |

### Loading Real Data

#### From NILMTK (UK-DALE, REDD)

```python
from nilm.data import DataLoader, DataPreprocessor

loader = DataLoader()
aggregate, appliance_data = loader.load_from_nilmtk(
    "path/to/ukdale.h5",
    building=1,
    appliances=["kettle", "microwave", "fridge"]
)

preprocessor = DataPreprocessor(window_size=599)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    aggregate.values,
    appliance_data["kettle"].values,
    appliance_name="kettle"
)
```

#### From CSV Files

```python
aggregate, appliance_data = loader.load_from_csv(
    aggregate_path="data/aggregate.csv",
    appliance_paths={"kettle": "data/kettle.csv"},
    timestamp_col="timestamp",
    power_col="power"
)
```

### Custom Model Configuration

```python
from nilm.models import BiLSTM

# Custom BiLSTM configuration
model = BiLSTM(
    appliance_name="dishwasher",
    window_size=599,
    learning_rate=1e-4,
    lstm_units=[256, 128, 64],  # 3 BiLSTM layers
    use_conv=True,              # Use initial conv layer
    conv_filters=64,            # Conv filter count
    dropout_rate=0.3            # Dropout rate
)

# Custom training
history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=100,
    batch_size=128,
    early_stopping_patience=15,
    reduce_lr_patience=7
)
```

### Visualization

```python
from nilm.utils import Visualizer

viz = Visualizer()

# Plot predictions
viz.plot_predictions(y_test, y_pred, appliance_name="Kettle")

# Plot training history
viz.plot_training_history(history)

# Compare multiple models
viz.plot_model_comparison(
    {"BiLSTM": bilstm_metrics, "CNN": cnn_metrics, "Seq2Point": s2p_metrics},
    metrics=["mae", "rmse", "f1"]
)
```

## Project Structure

```
nilm/
├── src/
│   ├── nilm/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py    # Base class for all models
│   │   │   ├── bilstm.py        # BiLSTM implementations
│   │   │   ├── cnn.py           # CNN implementations
│   │   │   ├── seq2point.py     # Seq2Point implementations
│   │   │   └── seq2seq.py       # Seq2Seq implementations
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py   # Data loading utilities
│   │   │   └── preprocessor.py  # Data preprocessing
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── metrics.py       # Evaluation metrics
│   │       └── visualization.py # Plotting utilities
│   └── train.py                 # Training script
├── examples/
│   ├── example_training.py      # Full training example
│   └── quick_start.py           # Minimal example
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Supported Appliances

The package includes pre-configured parameters for common appliances:

- Kettle
- Microwave
- Fridge/Freezer
- Dishwasher
- Washing Machine

Custom appliances can be added by specifying normalization parameters.

## References

1. Kelly, J., & Knottenbelt, W. (2015). Neural NILM: Deep Neural Networks Applied to Energy Disaggregation. BuildSys.
2. Zhang, C., et al. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. AAAI.
3. Kolter, J. Z., & Johnson, M. J. (2011). REDD: A public data set for energy disaggregation research.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
