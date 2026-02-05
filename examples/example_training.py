"""
Example script demonstrating how to use the NILM package.

This script shows:
1. Loading data
2. Training multiple models
3. Evaluating and comparing results
4. Visualizing predictions
"""

import os
import sys

import numpy as np

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nilm.models import BiLSTM, CNN, Seq2Point, Seq2Seq
from nilm.data import DataLoader, DataPreprocessor
from nilm.utils import MetricsCalculator, Visualizer


def main():
    """Run example NILM training and evaluation."""
    
    # Configuration
    APPLIANCE = "kettle"
    WINDOW_SIZE = 599
    EPOCHS = 10  # Use more epochs for real training
    BATCH_SIZE = 32
    
    print("=" * 60)
    print("NILM Example - Energy Disaggregation with Deep Learning")
    print("=" * 60)
    
    # 1. Load/Generate Data
    print("\n1. Loading Data...")
    data_loader = DataLoader()
    
    # Generate synthetic data for demonstration
    # Replace with real data loading for actual use
    X, y = data_loader.generate_synthetic_data(
        num_samples=2000,
        window_size=WINDOW_SIZE,
        appliance=APPLIANCE
    )
    
    # 2. Preprocess Data
    print("\n2. Preprocessing Data...")
    preprocessor = DataPreprocessor(
        window_size=WINDOW_SIZE,
        normalize_method="appliance"
    )
    
    # Split into train/val/test
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(X_train, y_train, train_ratio=0.875)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # 3. Define Models to Compare
    models = {
        "BiLSTM": BiLSTM(
            appliance_name=APPLIANCE,
            window_size=WINDOW_SIZE,
            lstm_units=[64, 32],
            dropout_rate=0.2
        ),
        "CNN": CNN(
            appliance_name=APPLIANCE,
            window_size=WINDOW_SIZE,
            dropout_rate=0.2
        ),
        "Seq2Point": Seq2Point(
            appliance_name=APPLIANCE,
            window_size=WINDOW_SIZE,
            dropout_rate=0.2
        ),
    }
    
    # 4. Train and Evaluate Each Model
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print("="*60)
        
        # Train
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative
        predictions[name] = y_pred
        
        # Calculate metrics
        calculator = MetricsCalculator(appliance_name=APPLIANCE)
        metrics = calculator.calculate_all(y_test, y_pred)
        results[name] = metrics
        
        print(f"\n{name} Results:")
        calculator.print_metrics(metrics)
    
    # 5. Compare Models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    print("\n{:<20} {:>10} {:>10} {:>10}".format("Model", "MAE", "RMSE", "F1"))
    print("-" * 50)
    for name, metrics in results.items():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, metrics["mae"], metrics["rmse"], metrics["f1"]
        ))
    
    # 6. Visualization (optional)
    try:
        print("\n6. Generating Visualizations...")
        visualizer = Visualizer()
        
        # Compare all models
        visualizer.plot_model_comparison(
            results,
            metrics=["mae", "rmse", "f1"],
            title="Model Comparison"
        )
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
    except Exception as e:
        print(f"  (Could not generate visualizations: {e})")
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
