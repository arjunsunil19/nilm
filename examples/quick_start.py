#!/usr/bin/env python
"""
Quick Start Example for NILM

This example shows how to quickly get started with the NILM package
using synthetic data and a BiLSTM model.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Import from NILM package
from nilm.models import BiLSTM, CNN, Seq2Point, Seq2Seq
from nilm.data import SyntheticNILMDataset
from nilm.utils import Trainer, Evaluator


def quick_start_example():
    """
    A minimal example demonstrating the NILM workflow:
    1. Create/load data
    2. Create model
    3. Train
    4. Evaluate
    """
    print("=" * 60)
    print("NILM Quick Start Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # =========================================
    # 1. Create Dataset
    # =========================================
    print("\n1. Creating synthetic dataset...")
    
    window_size = 99  # Smaller for quick demo
    n_appliances = 3
    
    dataset = SyntheticNILMDataset(
        n_samples=5000,      # 5000 time steps
        n_appliances=n_appliances,
        window_size=window_size,
        noise_level=0.1,
        seed=42,
    )
    
    print(f"   Dataset size: {len(dataset)} windows")
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # =========================================
    # 2. Create Model
    # =========================================
    print("\n2. Creating BiLSTM model...")
    
    model = BiLSTM(
        window_size=window_size,
        num_appliances=n_appliances,
        hidden_dim=64,       # Smaller for quick demo
        num_layers=1,        # Fewer layers for quick demo
        dropout=0.2,
    )
    
    print(f"   Parameters: {model.count_parameters():,}")
    
    # =========================================
    # 3. Train Model
    # =========================================
    print("\n3. Training model (5 epochs)...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer="adam",
        lr=1e-3,
        device=device,
        early_stopping_patience=3,
    )
    
    history = trainer.train(epochs=5, verbose=True)
    
    # =========================================
    # 4. Evaluate Model
    # =========================================
    print("\n4. Evaluating on test set...")
    
    appliance_names = [f"Appliance_{i+1}" for i in range(n_appliances)]
    
    evaluator = Evaluator(
        model=model,
        device=device,
        appliance_names=appliance_names,
    )
    
    results = evaluator.evaluate(test_loader, verbose=True)
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print(f"Final Test MAE:  {results['mae']:.4f}")
    print(f"Final Test RMSE: {results['rmse']:.4f}")
    print("\nNext steps:")
    print("  - Use real data (REDD, UK-DALE, REFIT)")
    print("  - Try different models (CNN, Seq2Point, Seq2Seq)")
    print("  - Adjust hyperparameters")
    print("  - Train for more epochs")


if __name__ == "__main__":
    quick_start_example()
