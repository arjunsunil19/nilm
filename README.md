# NILM System v7.0

Non-Intrusive Load Monitoring (NILM) System with Multiple Model Architectures

## Overview

This NILM system implements multiple deep learning architectures for energy disaggregation:
- **CNN-BiLSTM-Attention**: Convolutional Neural Network with Bidirectional LSTM and Attention mechanism
- **Transformer**: Transformer-based architecture for sequence-to-sequence learning
- **U-Net**: U-Net architecture adapted for time series disaggregation
- **ResNet-LSTM**: Residual Network combined with LSTM layers

## Features

- Multiple model architectures for comparison
- Configurable hyperparameters
- Support for Excel data files
- Comprehensive evaluation metrics
- Easy-to-use command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run with default settings (CNN-BiLSTM-Attention model):

```bash
python nilm_system.py
```

### Specify Model Architecture

```bash
# Use Transformer model
python nilm_system.py --model transformer

# Use U-Net model
python nilm_system.py --model unet

# Use ResNet-LSTM model
python nilm_system.py --model resnet_lstm
```

### Run All Models

Compare all available models:

```bash
python nilm_system.py --model all
```

### Custom Configuration

Specify custom data file, epochs, and batch size:

```bash
python nilm_system.py --model cnn_bilstm_attention \
                      --data path/to/your/data.xlsx \
                      --epochs 150 \
                      --batch 64
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `cnn_bilstm_attention` | Model architecture: `cnn_bilstm_attention`, `transformer`, `unet`, `resnet_lstm`, or `all` |
| `--data` | str | `combined_appliances_correct_order.xlsx` | Path to input data file |
| `--epochs` | int | `200` | Maximum number of training epochs |
| `--batch` | int | `32` | Batch size for training |

## Data Format

The system expects data in Excel format (`.xlsx`) with the following structure:
- Aggregate power consumption data
- Individual appliance power consumption data
- Properly ordered columns

Example data file: `combined_appliances_correct_order.xlsx`

## Model Architectures

### CNN-BiLSTM-Attention
Combines convolutional layers for feature extraction, bidirectional LSTM for temporal dependencies, and attention mechanism for focusing on relevant time steps.

### Transformer
Uses self-attention mechanisms to capture long-range dependencies in power consumption sequences.

### U-Net
Encoder-decoder architecture with skip connections, adapted from image segmentation to time series disaggregation.

### ResNet-LSTM
Combines residual connections for deep feature learning with LSTM layers for temporal modeling.

## Output

The system provides:
- Training progress and metrics
- Evaluation results (accuracy, MAE, RMSE)
- Model comparison (when using `--model all`)

## Example Output

```
######################################################################
  NILM SYSTEM v7.0 - CNN_BILSTM_ATTENTION
######################################################################

======================================================================
  Loading data from: combined_appliances_correct_order.xlsx
======================================================================
  ✓ Data loaded successfully

======================================================================
  Building cnn_bilstm_attention model
======================================================================
  ✓ cnn_bilstm_attention model built successfully

======================================================================
  Training model
======================================================================
  Max epochs: 200
  Batch size: 32
  ✓ Training completed successfully

======================================================================
  Evaluating model
======================================================================
  ✓ Evaluation completed
    Accuracy: 0.9500
    MAE: 0.0500
    RMSE: 0.0800

======================================================================
  NILM SYSTEM COMPLETE
======================================================================
```

## Error Handling

The system includes comprehensive error handling:
- **FileNotFoundError**: Raised when data file is not found
- **KeyboardInterrupt**: Gracefully handles user interruption
- **General Exceptions**: Provides detailed traceback for debugging

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Contact

For questions or support, please open an issue on the GitHub repository.

## Version History

- **v7.0**: Multiple model architectures with command-line interface
- Support for CNN-BiLSTM-Attention, Transformer, U-Net, and ResNet-LSTM models
