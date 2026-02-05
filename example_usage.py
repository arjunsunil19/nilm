"""Example usage script for NILM System."""

import os
import logging
from nilm_system import NILMSystem
from config import NILMConfig, ApplianceConfig, DataConfig, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    logger.info("Example 1: Basic Usage")
    
    # Create default configuration
    config = NILMConfig()
    
    # Create and run system
    system = NILMSystem(config)
    
    # Note: This will fail if the data file doesn't exist
    # system.run(data_file='combined_appliances_correct_order.xlsx')
    
    logger.info("Basic usage example completed (skipped actual run)")


def example_custom_model():
    """Example 2: Using different model architectures."""
    logger.info("Example 2: Custom Model Selection")
    
    # Try different model architectures
    model_types = ['cnn_bilstm_attention', 'transformer', 'unet', 'resnet_lstm']
    
    for model_type in model_types:
        logger.info(f"\nUsing model: {model_type}")
        
        config = NILMConfig()
        config.model_type = model_type
        
        # Adjust output directories for each model
        config.output_dir = f'outputs_{model_type}'
        config.models_dir = f'{config.output_dir}/models'
        config.plots_dir = f'{config.output_dir}/plots'
        config.metrics_dir = f'{config.output_dir}/metrics'
        config.predictions_dir = f'{config.output_dir}/predictions'
        
        system = NILMSystem(config)
        # system.run(data_file='combined_appliances_correct_order.xlsx')
    
    logger.info("Custom model example completed (skipped actual runs)")


def example_custom_configuration():
    """Example 3: Custom configuration for specific use case."""
    logger.info("Example 3: Custom Configuration")
    
    config = NILMConfig()
    
    # Customize data processing
    config.data.window_size = 199  # Larger window
    config.data.midpoint = 99
    config.data.train_ratio = 0.80  # More training data
    config.data.val_ratio = 0.10
    config.data.test_ratio = 0.10
    
    # Customize model
    config.model.batch_size = 256
    config.model.epochs = 150
    config.model.learning_rate = 0.0005
    config.model.patience = 20
    config.model.cnn_filters = 128  # More filters
    config.model.lstm_units = [256, 128]  # Larger LSTM
    
    # Customize post-processing
    config.post_processing.median_filter_size = 7
    config.post_processing.gaussian_sigma = 1.5
    
    system = NILMSystem(config)
    # system.run(data_file='combined_appliances_correct_order.xlsx')
    
    logger.info("Custom configuration example completed (skipped actual run)")


def example_custom_appliances():
    """Example 4: Add custom appliances."""
    logger.info("Example 4: Custom Appliances")
    
    config = NILMConfig()
    
    # Add a custom appliance
    config.appliances['Microwave'] = ApplianceConfig(
        name='Microwave',
        type='binary',
        thresholds=[100],
        state_names=['OFF', 'ON'],
        noise_floor=15.0
    )
    
    # Modify existing appliance
    config.appliances['Fan'].thresholds = [15, 30, 50]  # Different thresholds
    config.appliances['Fan'].noise_floor = 8.0
    
    system = NILMSystem(config)
    # system.run(data_file='combined_appliances_correct_order.xlsx')
    
    logger.info("Custom appliances example completed (skipped actual run)")


def example_training_single_appliance():
    """Example 5: Train model for a single appliance."""
    logger.info("Example 5: Single Appliance Training")
    
    import pandas as pd
    from utils.data_processing import load_data, handle_missing_values, clip_negative_values
    
    # Create configuration for single appliance
    config = NILMConfig()
    
    # Keep only AC
    config.appliances = {
        'AC': config.appliances['AC']
    }
    
    system = NILMSystem(config)
    
    # Load data (example - would need actual file)
    # df = load_data('combined_appliances_correct_order.xlsx')
    # df = handle_missing_values(df)
    # df = clip_negative_values(df)
    
    # Train single appliance
    # system.train_appliance(df, 'AC', config.appliances['AC'])
    
    logger.info("Single appliance training example completed (skipped actual run)")


def example_inference_only():
    """Example 6: Load trained model and make predictions."""
    logger.info("Example 6: Inference Only")
    
    import tensorflow as tf
    import numpy as np
    
    # Load a saved model
    model_path = 'outputs/models/AC_model_best.keras'
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'CombinedRegressionLoss': None,  # Would need actual class
                'FocalLoss': None
            }
        )
        
        # Make predictions on new data
        # X_new = np.random.randn(100, 99, 8)  # Example input
        # power_pred, state_pred = model.predict(X_new)
        
        logger.info("Model loaded successfully")
    else:
        logger.info(f"Model not found at {model_path}")
    
    logger.info("Inference example completed")


def example_evaluate_saved_model():
    """Example 7: Evaluate a saved model on new data."""
    logger.info("Example 7: Evaluate Saved Model")
    
    from nilm_system import NILMSystem
    from config import NILMConfig
    import pandas as pd
    
    config = NILMConfig()
    system = NILMSystem(config)
    
    # Load data
    # df = load_data('new_test_data.xlsx')
    
    # Load saved model
    # model = tf.keras.models.load_model('outputs/models/AC_model_best.keras')
    
    # Prepare data and make predictions
    # ...
    
    logger.info("Evaluation example completed (skipped actual run)")


def example_compare_models():
    """Example 8: Compare performance of different models."""
    logger.info("Example 8: Compare Models")
    
    import pandas as pd
    
    results = []
    model_types = ['cnn_bilstm_attention', 'transformer', 'unet', 'resnet_lstm']
    
    for model_type in model_types:
        config = NILMConfig()
        config.model_type = model_type
        config.output_dir = f'outputs_{model_type}'
        
        # Would train and evaluate each model
        # metrics = train_and_evaluate(config)
        # results.append({
        #     'model': model_type,
        #     'mae': metrics['avg_mae'],
        #     'f1_score': metrics['avg_f1_score']
        # })
    
    # Create comparison DataFrame
    # results_df = pd.DataFrame(results)
    # results_df.to_csv('model_comparison.csv', index=False)
    
    logger.info("Model comparison example completed (skipped actual runs)")


def main():
    """Run all examples."""
    logger.info("="*80)
    logger.info("NILM System - Usage Examples")
    logger.info("="*80 + "\n")
    
    # Run examples (most are demonstrations only)
    example_basic_usage()
    print()
    
    example_custom_model()
    print()
    
    example_custom_configuration()
    print()
    
    example_custom_appliances()
    print()
    
    example_training_single_appliance()
    print()
    
    example_inference_only()
    print()
    
    example_evaluate_saved_model()
    print()
    
    example_compare_models()
    print()
    
    logger.info("\n" + "="*80)
    logger.info("All examples completed!")
    logger.info("="*80)
    logger.info("\nNote: Most examples show usage patterns but skip actual execution")
    logger.info("To run the system, provide a valid data file:")
    logger.info("  system.run(data_file='combined_appliances_correct_order.xlsx')")


if __name__ == '__main__':
    main()
