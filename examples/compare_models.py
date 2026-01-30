#!/usr/bin/env python
"""
Model Comparison Example

This example compares different NILM models on the same dataset.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from nilm.models import BiLSTM, BiLSTMWithConv, CNN, DilatedCNN, Seq2Point, Seq2Seq
from nilm.data import SyntheticNILMDataset
from nilm.utils import Trainer, Evaluator, evaluate_models


def compare_models():
    """Compare different NILM model architectures."""
    
    print("=" * 70)
    print("NILM Model Comparison")
    print("=" * 70)
    
    # Settings
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    window_size = 99
    n_appliances = 3
    epochs = 5  # Quick comparison
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = SyntheticNILMDataset(
        n_samples=5000,
        n_appliances=n_appliances,
        window_size=window_size,
        seed=42,
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Define models to compare
    model_configs = {
        "BiLSTM": BiLSTM(
            window_size=window_size,
            num_appliances=n_appliances,
            hidden_dim=64,
            num_layers=1,
        ),
        "BiLSTM+Conv": BiLSTMWithConv(
            window_size=window_size,
            num_appliances=n_appliances,
            hidden_dim=64,
            num_layers=1,
            conv_filters=32,
        ),
        "CNN": CNN(
            window_size=window_size,
            num_appliances=n_appliances,
            num_filters=[16, 32, 32],
            kernel_sizes=[5, 5, 5],
        ),
        "Seq2Point": Seq2Point(
            window_size=window_size,
            num_appliances=n_appliances,
            num_filters=[16, 32, 32],
            kernel_sizes=[5, 5, 5],
        ),
    }
    
    # Train each model
    trained_models = {}
    
    for name, model in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"Parameters: {model.count_parameters():,}")
        print("=" * 50)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adam",
            lr=1e-3,
            device=device,
            early_stopping_patience=3,
        )
        
        trainer.train(epochs=epochs, verbose=True)
        trained_models[name] = model
    
    # Compare all models
    print("\n" + "=" * 70)
    print("Final Comparison")
    print("=" * 70)
    
    appliance_names = [f"App_{i+1}" for i in range(n_appliances)]
    results = evaluate_models(
        trained_models,
        test_loader,
        device=device,
        appliance_names=appliance_names,
    )
    
    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]["mae"])
    print(f"\n🏆 Best model by MAE: {best_model}")


if __name__ == "__main__":
    compare_models()
