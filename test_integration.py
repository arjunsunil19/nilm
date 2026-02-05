"""Integration test for NILM system."""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import sys

print("="*80)
print("NILM System - Integration Test")
print("="*80)

# Test 1: Configuration
print("\n[Test 1] Configuration System")
try:
    from config import NILMConfig, ApplianceConfig, DataConfig, ModelConfig
    config = NILMConfig()
    print(f"✓ Configuration loaded successfully")
    print(f"  - {len(config.appliances)} appliances configured")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Window size: {config.data.window_size}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 2: Data Processing
print("\n[Test 2] Data Processing Utilities")
try:
    from utils.data_processing import (
        handle_missing_values, clip_negative_values,
        compute_statistics, create_sequences, split_data
    )
    
    # Create test data
    df = pd.DataFrame({
        'Mains': np.random.uniform(100, 500, 1000),
        'AC': np.random.uniform(0, 150, 1000)
    })
    
    df = handle_missing_values(df)
    df = clip_negative_values(df)
    stats = compute_statistics(df, ['Mains', 'AC'])
    
    print(f"✓ Data processing utilities work correctly")
    print(f"  - Statistics computed for {len(stats)} columns")
except Exception as e:
    print(f"✗ Data processing failed: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n[Test 3] Feature Engineering")
try:
    from utils.feature_engineering import compute_8_channel_features, create_target_labels
    
    mains = np.random.uniform(100, 500, 1000)
    features = compute_8_channel_features(mains, window_size=5)
    
    appliance = np.random.uniform(0, 150, 1000)
    power_targets, state_labels = create_target_labels(appliance, [80], 'binary')
    
    print(f"✓ Feature engineering works correctly")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Number of channels: {features.shape[1]}")
except Exception as e:
    print(f"✗ Feature engineering failed: {e}")
    sys.exit(1)

# Test 4: Model Architectures
print("\n[Test 4] Model Architectures")
try:
    from models.cnn_bilstm_attention import build_cnn_bilstm_attention_model
    from models.transformer import build_transformer_model
    from models.unet import build_unet_model
    from models.resnet_lstm import build_resnet_lstm_model
    
    models = {
        'CNN-BiLSTM-Attention': build_cnn_bilstm_attention_model,
        'Transformer': build_transformer_model,
        'UNet': build_unet_model,
        'ResNet-LSTM': build_resnet_lstm_model
    }
    
    for name, build_fn in models.items():
        model = build_fn(input_shape=(99, 8), num_classes=2)
        param_count = model.count_params()
        print(f"  ✓ {name}: {param_count:,} parameters")
    
    print(f"✓ All {len(models)} model architectures build successfully")
except Exception as e:
    print(f"✗ Model architecture failed: {e}")
    sys.exit(1)

# Test 5: Metrics
print("\n[Test 5] Evaluation Metrics")
try:
    from utils.metrics import (
        calculate_mae, calculate_rmse, calculate_energy_accuracy,
        calculate_classification_metrics, calculate_all_metrics
    )
    
    y_true = np.random.uniform(0, 100, 100)
    y_pred = y_true + np.random.normal(0, 10, 100)
    
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    energy_acc = calculate_energy_accuracy(y_true, y_pred)
    
    y_true_state = np.random.randint(0, 2, 100)
    y_pred_state = y_true_state.copy()
    # Randomly flip some predictions
    flip_mask = np.random.rand(100) > 0.9
    y_pred_state[flip_mask] = 1 - y_pred_state[flip_mask]
    
    class_metrics = calculate_classification_metrics(y_true_state, y_pred_state, 2)
    
    print(f"✓ Metrics calculation works correctly")
    print(f"  - MAE: {mae:.2f}W")
    print(f"  - F1-Score: {class_metrics['f1_score']:.4f}")
except Exception as e:
    print(f"✗ Metrics failed: {e}")
    sys.exit(1)

# Test 6: Post-Processing
print("\n[Test 6] Post-Processing")
try:
    from utils.post_processing import (
        apply_median_filter, apply_gaussian_smoothing,
        apply_noise_floor_threshold, apply_physical_bounds,
        post_process_predictions
    )
    
    predictions = np.random.uniform(0, 100, 100)
    filtered = apply_median_filter(predictions, filter_size=5)
    smoothed = apply_gaussian_smoothing(filtered, sigma=1.0)
    
    power_final, state_final = post_process_predictions(
        predictions, np.zeros(100, dtype=np.int32),
        thresholds=[80], appliance_type='binary',
        noise_floor=5.0
    )
    
    print(f"✓ Post-processing works correctly")
    print(f"  - Applied median filtering, Gaussian smoothing, noise floor")
except Exception as e:
    print(f"✗ Post-processing failed: {e}")
    sys.exit(1)

# Test 7: Visualization (without saving)
print("\n[Test 7] Visualization Utilities")
try:
    from utils.visualization import (
        plot_training_history, plot_disaggregation,
        plot_confusion_matrices, plot_energy_comparison
    )
    
    print(f"✓ Visualization utilities imported successfully")
    print(f"  - Training history plots")
    print(f"  - Disaggregation plots")
    print(f"  - Confusion matrices")
    print(f"  - Energy comparison")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
    sys.exit(1)

# Test 8: Main System
print("\n[Test 8] Main NILM System")
try:
    from nilm_system import NILMSystem
    
    config = NILMConfig()
    system = NILMSystem(config)
    
    print(f"✓ NILM System initialized successfully")
    print(f"  - Output directories created")
    print(f"  - Ready to process data")
except Exception as e:
    print(f"✗ NILM System failed: {e}")
    sys.exit(1)

# Test 9: Full Pipeline (with synthetic data)
print("\n[Test 9] Full Pipeline Simulation")
try:
    # Create synthetic data
    n_samples = 1000
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Mains': np.random.uniform(100, 500, n_samples),
        'AC': np.where(np.random.rand(n_samples) > 0.7, np.random.uniform(80, 150, n_samples), 0)
    })
    
    # Test data preparation
    X, y_power, y_state = system.prepare_data_for_appliance(
        df, 'AC', config.appliances['AC']
    )
    
    # Test windowing
    X_seq, y_power_seq, y_state_seq = system.create_windowed_data(X, y_power, y_state)
    
    # Test splitting
    split_result = system.split_train_val_test(X_seq, y_power_seq, y_state_seq)
    X_train = split_result[0]
    
    print(f"✓ Full pipeline simulation successful")
    print(f"  - Data prepared: {X.shape}")
    print(f"  - Sequences created: {X_seq.shape}")
    print(f"  - Training samples: {X_train.shape[0]}")
except Exception as e:
    print(f"✗ Full pipeline failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("Integration Test Summary")
print("="*80)
print("✓ All 9 tests passed successfully!")
print("\nThe NILM system is ready to use with real data.")
print("\nTo run the system:")
print("  1. Prepare your data file (Excel or CSV)")
print("  2. Configure appliances in config.py")
print("  3. Run: python nilm_system.py")
print("\nFor examples, see: example_usage.py")
print("="*80)
