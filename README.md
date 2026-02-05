# NILM System - Non-Intrusive Load Monitoring

A comprehensive deep learning system for appliance disaggregation using multiple state-of-the-art architectures.

## Overview

This NILM (Non-Intrusive Load Monitoring) system implements four advanced deep learning architectures for energy disaggregation:

1. **CNN-BiLSTM-Attention**: Convolutional Neural Network with Bidirectional LSTM and Attention mechanism
2. **Transformer**: Self-attention based architecture with positional encoding
3. **U-Net**: Encoder-decoder architecture for sequence-to-sequence prediction
4. **ResNet-LSTM**: Hybrid ResNet and LSTM architecture with residual connections

## Target Appliances

The system supports five appliances with different classification types:

- **Air Conditioner** (Binary: OFF/ON) - High power consumption
- **Refrigerator** (Binary: OFF/ON) - Cyclic behavior
- **Fan** (4 states: OFF/Low/Medium/High) - Multi-state classification
- **Washing Machine** (4 states: OFF/Wash/Rinse/Spin) - Complex multi-state pattern
- **EV Charger** (Binary: OFF/ON) - Very high power consumption

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Other dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/arjunsunil19/nilm.git
cd nilm

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Run All Models

Run comprehensive comparison of all models across all appliances:

```bash
python nilm_system.py --all
```

This will:
- Train all 4 model architectures
- Test on all 5 appliances
- Generate comparison visualizations
- Save detailed results to `results/all_results.json`

### Run Specific Model for Specific Appliance

```bash
# CNN-BiLSTM-Attention for Air Conditioner
python nilm_system.py --model cnn_bilstm_attention --appliance air_conditioner

# Transformer for Washing Machine
python nilm_system.py --model transformer --appliance washing_machine

# U-Net for Fan
python nilm_system.py --model unet --appliance fan

# ResNet-LSTM for EV Charger
python nilm_system.py --model resnet_lstm --appliance ev_charger
```

### Run Specific Model for All Appliances

```bash
python nilm_system.py --model transformer
```

### Advanced Options

```bash
# Quick test with fewer epochs
python nilm_system.py --model unet --appliance fan --epochs 20

# More training samples
python nilm_system.py --all --samples 20000 --epochs 150

# Custom batch size
python nilm_system.py --model cnn_bilstm_attention --batch-size 64
```

### Command Line Arguments

- `--all`: Run all models for all appliances (comprehensive comparison)
- `--model`: Choose model architecture (cnn_bilstm_attention, transformer, unet, resnet_lstm)
- `--appliance`: Choose target appliance (air_conditioner, refrigerator, fan, washing_machine, ev_charger)
- `--epochs`: Number of training epochs (default: 100)
- `--samples`: Number of samples to generate (default: 10000)
- `--batch-size`: Batch size for training (default: 32)

### Get Help

```bash
python nilm_system.py --help
```

## Output

### Directory Structure

After running the system, the following directories will be created:

```
nilm/
├── nilm_system.py          # Main system
├── requirements.txt        # Dependencies
├── README.md              # This file
├── models/                # Trained models
│   ├── cnn_bilstm_attention_air_conditioner.h5
│   ├── transformer_refrigerator.h5
│   └── ...
└── results/               # Results and visualizations
    ├── all_results.json
    ├── cnn_bilstm_attention_air_conditioner_training_history.png
    ├── cnn_bilstm_attention_air_conditioner_predictions.png
    ├── cnn_bilstm_attention_air_conditioner_confusion_matrix.png
    ├── model_comparison_F1_Score.png
    ├── model_comparison_MAE.png
    └── overall_summary.png
```

### Evaluation Metrics

The system calculates comprehensive metrics for each model-appliance combination:

- **MAE (Mean Absolute Error)**: Average power prediction error in Watts
- **RMSE (Root Mean Square Error)**: Root mean square power error
- **Energy Accuracy**: Percentage accuracy of total energy prediction
- **State Accuracy**: Classification accuracy for appliance states
- **Precision**: Precision score for state classification
- **Recall**: Recall score for state classification
- **F1-Score**: Harmonic mean of precision and recall

### Visualizations

The system generates publication-quality visualizations:

1. **Training History**: Loss and MAE curves during training
2. **Predictions vs Ground Truth**: Time series comparison
3. **Confusion Matrix**: State classification performance
4. **Model Comparison**: Bar charts comparing all models
5. **Overall Summary**: Comprehensive dashboard of all metrics

## Performance Targets

The system is designed to meet the following performance targets:

- **Average MAE**: < 50W across all appliances
- **Average F1-Score**: > 0.85 for state classification
- **Average Energy Accuracy**: > 80% for total energy prediction

## Architecture Details

### 1. CNN-BiLSTM-Attention

- Convolutional layers for local feature extraction
- Bidirectional LSTM for temporal dependencies
- Attention mechanism for focusing on important time steps
- Best for: Complex temporal patterns

### 2. Transformer

- Multi-head self-attention mechanism
- Positional encoding for sequence information
- Feed-forward networks with residual connections
- Best for: Long-range dependencies

### 3. U-Net

- Encoder-decoder architecture
- Skip connections for preserving spatial information
- Symmetric upsampling and downsampling
- Best for: Sequence-to-sequence tasks

### 4. ResNet-LSTM

- Residual blocks for deep feature learning
- LSTM layers for temporal modeling
- Hybrid CNN-RNN architecture
- Best for: Balanced performance

## Data Pipeline

The system includes a complete data pipeline:

1. **Data Generation**: Synthetic data generation with realistic patterns
2. **Windowing**: Sliding window approach for sequence processing
3. **Normalization**: StandardScaler for input and output normalization
4. **Train/Val/Test Split**: 70/15/15 split for robust evaluation

## Model Training

Training includes:

- **Early Stopping**: Prevents overfitting (patience=15)
- **Model Checkpointing**: Saves best model based on validation loss
- **Learning Rate Reduction**: Adaptive learning rate (ReduceLROnPlateau)
- **Batch Normalization**: Stabilizes training
- **Dropout**: Regularization to prevent overfitting

## Post-Processing

Predictions are post-processed with:

- **Threshold Application**: Removes low-power noise
- **State Classification**: Maps continuous predictions to discrete states
- **Denormalization**: Converts predictions back to power values

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU memory issues:

```python
# Add to the beginning of nilm_system.py
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Memory Issues

Reduce batch size or number of samples:

```bash
python nilm_system.py --model unet --batch-size 16 --samples 5000
```

### Slow Training

Use fewer epochs for quick testing:

```bash
python nilm_system.py --model transformer --epochs 20
```

## Citation

If you use this NILM system in your research, please cite:

```bibtex
@software{nilm_system_2026,
  title = {Complete NILM System for Appliance Disaggregation},
  author = {arjunsunil19},
  year = {2026},
  url = {https://github.com/arjunsunil19/nilm}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue on GitHub.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Energy disaggregation research community
- Contributors to NILM datasets and benchmarks
