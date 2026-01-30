#!/usr/bin/env python
"""
NILM Training Example

This script demonstrates how to train NILM models using the package.
It includes examples for BiLSTM, CNN, Seq2Point, and Seq2Seq models.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nilm.models import BiLSTM, CNN, Seq2Point, Seq2Seq
from nilm.data import NILMDataset, SyntheticNILMDataset
from nilm.utils import Trainer, Evaluator
from nilm.configs import get_default_config, TrainingConfig


def create_model(model_type: str, config: dict):
    """Create a model based on type and config."""
    models = {
        "bilstm": BiLSTM,
        "cnn": CNN,
        "seq2point": Seq2Point,
        "seq2seq": Seq2Seq,
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type.lower()](**config)


def main():
    parser = argparse.ArgumentParser(description="Train NILM models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="bilstm",
        choices=["bilstm", "cnn", "seq2point", "seq2seq"],
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=599,
        help="Input window size"
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=128,
        help="Hidden dimension for LSTM models"
    )
    parser.add_argument(
        "--n-appliances", 
        type=int, 
        default=5,
        help="Number of appliances"
    )
    parser.add_argument(
        "--synthetic", 
        action="store_true",
        help="Use synthetic data for demonstration"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create synthetic dataset for demonstration
    print("\nCreating synthetic NILM dataset...")
    dataset = SyntheticNILMDataset(
        n_samples=10000,
        n_appliances=args.n_appliances,
        window_size=args.window_size,
        seed=42,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Window size: {args.window_size}")
    print(f"Number of appliances: {args.n_appliances}")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
    )
    
    # Get model config
    config = get_default_config(args.model)
    config.window_size = args.window_size
    config.num_appliances = args.n_appliances
    
    if hasattr(config, "hidden_dim"):
        config.hidden_dim = args.hidden_dim
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model = create_model(args.model, config.to_dict())
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer="adam",
        lr=args.lr,
        scheduler="plateau",
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=5,
    )
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(epochs=args.epochs, verbose=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    appliance_names = [f"Appliance_{i}" for i in range(args.n_appliances)]
    evaluator = Evaluator(
        model=model,
        device=device,
        appliance_names=appliance_names,
    )
    
    results = evaluator.evaluate(test_loader, verbose=True)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Final test MAE: {results['mae']:.4f}")
    print(f"Final test RMSE: {results['rmse']:.4f}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    model.save_model(final_path)
    print(f"\nModel saved to: {final_path}")


if __name__ == "__main__":
    main()
