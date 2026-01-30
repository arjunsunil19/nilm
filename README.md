# NILM - Non-Intrusive Load Monitoring

A comprehensive deep learning library for energy disaggregation using PyTorch.

## Overview

Non-Intrusive Load Monitoring (NILM) is the task of estimating individual appliance power consumption from aggregate household power measurements. This package provides state-of-the-art deep learning models for NILM, including:

- **BiLSTM**: Bidirectional LSTM with attention mechanism
- **BiLSTM+Conv**: BiLSTM with convolutional feature extraction
- **CNN**: Convolutional Neural Network based on Neural NILM
- **Dilated CNN**: Dilated convolutions for larger receptive field
- **Seq2Point**: Sequence-to-Point architecture
- **Seq2Seq**: Sequence-to-Sequence with attention
- **U-Net Seq2Seq**: U-Net style encoder-decoder

## Installation

```bash
# Clone the repository
git clone https://github.com/arjunsunil19/nilm.git
cd nilm

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from torch.utils.data import DataLoader

from nilm.models import BiLSTM
from nilm.data import SyntheticNILMDataset
from nilm.utils import Trainer, Evaluator

# Create synthetic dataset for demonstration
dataset = SyntheticNILMDataset(
    n_samples=5000,
    n_appliances=3,
    window_size=99,
)

# Create data loaders
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create BiLSTM model
model = BiLSTM(
    window_size=99,
    num_appliances=3,
    hidden_dim=128,
    num_layers=2,
)

# Train the model
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    optimizer="adam",
    lr=1e-3,
)

history = trainer.train(epochs=10)

# Evaluate
evaluator = Evaluator(model=model)
results = evaluator.evaluate(train_loader)
print(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
```

## Available Models

### BiLSTM
Bidirectional LSTM with attention mechanism for capturing temporal dependencies.

```python
from nilm.models import BiLSTM

model = BiLSTM(
    window_size=599,        # Input window size
    num_appliances=5,       # Number of appliances
    hidden_dim=128,         # LSTM hidden dimension
    num_layers=2,           # Number of LSTM layers
    dropout=0.2,            # Dropout rate
    output_type="seq2point" # "seq2point" or "seq2seq"
)
```

### CNN
Convolutional Neural Network based on the Neural NILM architecture.

```python
from nilm.models import CNN

model = CNN(
    window_size=599,
    num_appliances=5,
    num_filters=[30, 30, 40, 50, 50],
    kernel_sizes=[10, 8, 6, 5, 5],
)
```

### Seq2Point
Sequence-to-Point model that predicts power at the center of the input window.

```python
from nilm.models import Seq2Point

model = Seq2Point(
    window_size=599,
    num_appliances=5,
)
```

### Seq2Seq
Sequence-to-Sequence model with optional attention mechanism.

```python
from nilm.models import Seq2Seq

model = Seq2Seq(
    window_size=599,
    num_appliances=5,
    hidden_dim=128,
    use_attention=True,
)
```

## Dataset Support

### Using Real Datasets

Load data from common NILM datasets like REDD, UK-DALE, or REFIT:

```python
import numpy as np
from nilm.data import NILMDataset, DataPreprocessor

# Load your data (aggregate and appliance power)
aggregate = np.load("aggregate_power.npy")  # Shape: (n_samples,)
appliances = np.load("appliance_power.npy")  # Shape: (n_samples, n_appliances)

# Preprocess
preprocessor = DataPreprocessor(window_size=599)
aggregate_norm = preprocessor.fit_transform(aggregate, appliances)

# Create dataset
dataset = NILMDataset(
    aggregate=aggregate,
    appliances=appliances,
    window_size=599,
    stride=1,
    output_type="seq2point",
)
```

### Using Synthetic Data

For testing and demonstration:

```python
from nilm.data import SyntheticNILMDataset

dataset = SyntheticNILMDataset(
    n_samples=10000,
    n_appliances=5,
    window_size=599,
    noise_level=0.1,
    seed=42,
)
```

## Evaluation Metrics

The package includes comprehensive NILM evaluation metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **SAE**: Signal Aggregate Error
- **NDE**: Normalized Disaggregation Error
- **EAC**: Energy Accuracy
- **F1**: F1 Score for on/off classification
- **MCC**: Matthews Correlation Coefficient

```python
from nilm.utils.metrics import NILMMetrics

metrics = NILMMetrics()
results = metrics.compute_all_metrics(predictions, targets, threshold=10)
```

## Configuration

Use configuration classes for model setup:

```python
from nilm.configs import BiLSTMConfig, TrainingConfig

model_config = BiLSTMConfig(
    window_size=599,
    num_appliances=5,
    hidden_dim=128,
)

training_config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    epochs=100,
)
```

## Examples

Run example scripts:

```bash
# Quick start demonstration
python examples/quick_start.py

# Train a specific model
python examples/train_model.py --model bilstm --epochs 50

# Compare different models
python examples/compare_models.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nilm --cov-report=html
```

## Project Structure

```
nilm/
├── nilm/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Base model class
│   │   ├── bilstm.py        # BiLSTM models
│   │   ├── cnn.py           # CNN models
│   │   ├── seq2point.py     # Seq2Point models
│   │   └── seq2seq.py       # Seq2Seq models
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   └── preprocessing.py # Data preprocessing
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training utilities
│   │   ├── evaluator.py     # Evaluation utilities
│   │   └── metrics.py       # NILM metrics
│   └── configs/
│       ├── __init__.py
│       ├── model_configs.py      # Model configs
│       ├── training_configs.py   # Training configs
│       └── appliance_configs.py  # Appliance parameters
├── examples/
│   ├── quick_start.py
│   ├── train_model.py
│   └── compare_models.py
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

## References

This implementation is based on research from:

1. Kelly, J., & Knottenbelt, W. (2015). Neural NILM: Deep Neural Networks Applied to Energy Disaggregation.
2. Zhang, C., et al. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring.
3. Shin, C., et al. (2019). Subtask Gated Networks for Non-Intrusive Load Monitoring.

## License

MIT License - see LICENSE file for details.
