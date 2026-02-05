"""
NILM Training Script

Main entry point for training NILM models.
Supports multiple model architectures and configurations.
"""

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
import yaml

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nilm.models import BiLSTM, CNN, Seq2Point, Seq2Seq
from nilm.models.bilstm import BiLSTMSeq2Seq
from nilm.models.cnn import DeepCNN
from nilm.models.seq2point import Seq2PointAttention
from nilm.models.seq2seq import ConvSeq2Seq
from nilm.data import DataLoader, DataPreprocessor
from nilm.utils import MetricsCalculator, Visualizer


# Model registry
MODELS = {
    "bilstm": BiLSTM,
    "bilstm_seq2seq": BiLSTMSeq2Seq,
    "cnn": CNN,
    "deep_cnn": DeepCNN,
    "seq2point": Seq2Point,
    "seq2point_attention": Seq2PointAttention,
    "seq2seq": Seq2Seq,
    "conv_seq2seq": ConvSeq2Seq,
}


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(
    model_name: str,
    appliance: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
) -> object:
    """
    Train a NILM model.
    
    Args:
        model_name: Name of the model architecture
        appliance: Target appliance name
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Training configuration
        
    Returns:
        Trained model instance
    """
    # Get model class
    model_class = MODELS.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    # Create model
    model_config = config.get("model", {})
    model = model_class(
        appliance_name=appliance,
        window_size=config.get("window_size", 599),
        learning_rate=config.get("learning_rate", 1e-4),
        **model_config
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {appliance}")
    print(f"{'='*60}")
    model.summary()
    
    # Train
    train_config = config.get("training", {})
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=train_config.get("epochs", 100),
        batch_size=train_config.get("batch_size", 64),
        early_stopping_patience=train_config.get("early_stopping_patience", 10),
        reduce_lr_patience=train_config.get("reduce_lr_patience", 5),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        log_dir=config.get("log_dir", "./logs"),
        verbose=config.get("verbose", 1)
    )
    
    return model, history


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    appliance: str,
    preprocessor: Optional[DataPreprocessor] = None
) -> Dict:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test targets
        appliance: Appliance name
        preprocessor: Data preprocessor (for denormalization)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Denormalize if preprocessor provided
    if preprocessor is not None:
        y_test_denorm = preprocessor.denormalize(y_test, appliance, is_aggregate=False)
        y_pred_denorm = preprocessor.denormalize(y_pred, appliance, is_aggregate=False)
    else:
        y_test_denorm = y_test
        y_pred_denorm = y_pred
    
    # Clip negative predictions
    y_pred_denorm = np.clip(y_pred_denorm, 0, None)
    
    # Calculate metrics
    calculator = MetricsCalculator(appliance_name=appliance)
    metrics = calculator.calculate_all(y_test_denorm, y_pred_denorm)
    
    calculator.print_metrics(metrics)
    
    return metrics, y_pred_denorm


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NILM models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="bilstm",
        choices=list(MODELS.keys()),
        help="Model architecture to train"
    )
    parser.add_argument(
        "--appliance", "-a",
        type=str,
        default="kettle",
        help="Target appliance (e.g., kettle, microwave, fridge)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Path to data file (HDF5 or directory)"
    )
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=599,
        help="Input window size"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "window_size": args.window_size,
            "learning_rate": 1e-4,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
            },
            "checkpoint_dir": os.path.join(args.output_dir, "checkpoints"),
            "log_dir": os.path.join(args.output_dir, "logs"),
            "verbose": 1
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Load or generate data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    data_loader = DataLoader()
    preprocessor = DataPreprocessor(
        window_size=config["window_size"],
        normalize_method="appliance"
    )
    
    if args.synthetic or args.data_path is None:
        print("Using synthetic data for demonstration...")
        X, y = data_loader.generate_synthetic_data(
            num_samples=5000,
            window_size=config["window_size"],
            appliance=args.appliance
        )
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            X, y, train_ratio=0.8
        )
        X_train, X_val, y_train, y_val = preprocessor.train_test_split(
            X_train, y_train, train_ratio=0.875  # 0.8 * 0.875 = 0.7 for training
        )
    else:
        # Load real data
        if args.data_path.endswith(".h5") or args.data_path.endswith(".hdf5"):
            aggregate, appliance_data = data_loader.load_from_nilmtk(
                args.data_path,
                appliances=[args.appliance]
            )
        else:
            raise ValueError("For non-HDF5 data, please use load_from_csv() or provide CSV paths")
        
        # Prepare data
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            aggregate.values,
            appliance_data[args.appliance].values,
            args.appliance
        )
        X_train, X_val, y_train, y_val = preprocessor.train_test_split(
            X_train, y_train, train_ratio=0.875
        )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")
    
    # Train model
    model, history = train_model(
        args.model,
        args.appliance,
        X_train, y_train,
        X_val, y_val,
        config
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    metrics, y_pred = evaluate_model(
        model, X_test, y_test,
        args.appliance,
        preprocessor=None  # Data is already in normalized form
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model}_{args.appliance}.h5")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Visualize
    if args.visualize:
        try:
            visualizer = Visualizer()
            
            # Plot training history
            visualizer.plot_training_history(
                history,
                title=f"Training History - {args.model} for {args.appliance}",
                save_path=os.path.join(args.output_dir, f"{args.model}_{args.appliance}_history.png")
            )
            
            # Plot predictions
            visualizer.plot_predictions(
                y_test[:500], y_pred[:500],
                title=f"Predictions - {args.model}",
                appliance_name=args.appliance,
                save_path=os.path.join(args.output_dir, f"{args.model}_{args.appliance}_predictions.png")
            )
            
            # Plot distribution
            visualizer.plot_power_distribution(
                y_test, y_pred,
                title=f"Power Distribution - {args.model}",
                save_path=os.path.join(args.output_dir, f"{args.model}_{args.appliance}_distribution.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    main()
