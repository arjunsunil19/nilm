"""Main NILM System Pipeline."""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, List
from sklearn.utils.class_weight import compute_class_weight

from config import NILMConfig, ApplianceConfig
from utils.data_processing import (
    load_data, handle_missing_values, clip_negative_values,
    compute_statistics, create_sequences, split_data
)
from utils.feature_engineering import (
    compute_8_channel_features, create_target_labels
)
from utils.metrics import (
    calculate_all_metrics, aggregate_metrics, calculate_confusion_matrix
)
from utils.visualization import (
    plot_training_history, plot_disaggregation, plot_confusion_matrices,
    plot_energy_comparison, plot_power_distributions, save_predictions_to_csv
)
from utils.post_processing import post_process_predictions

from models.cnn_bilstm_attention import (
    build_cnn_bilstm_attention_model, FocalLoss, CombinedRegressionLoss
)
from models.transformer import build_transformer_model
from models.unet import build_unet_model
from models.resnet_lstm import build_resnet_lstm_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NILMSystem:
    """Main NILM System for energy disaggregation."""
    
    def __init__(self, config: NILMConfig):
        """
        Initialize NILM System.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models: Dict[str, keras.Model] = {}
        self.histories: Dict[str, Dict] = {}
        self.metrics: Dict[str, Dict] = {}
        self.predictions: Dict[str, Dict] = {}
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        tf.random.set_seed(config.random_seed)
        
        # Create output directories
        self._create_directories()
        
        logger.info("NILM System initialized")
    
    def _create_directories(self) -> None:
        """Create output directories if they don't exist."""
        for directory in [
            self.config.output_dir,
            self.config.models_dir,
            self.config.plots_dir,
            self.config.metrics_dir,
            self.config.predictions_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        logger.info("Output directories created")
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess data from file.
        
        Returns:
            Tuple of (preprocessed DataFrame, statistics dictionary)
        """
        logger.info("Loading data...")
        df = load_data(self.config.data.data_file)
        
        logger.info("Handling missing values...")
        df = handle_missing_values(df, method=self.config.data.fill_method)
        
        if self.config.data.clip_negative:
            logger.info("Clipping negative values...")
            df = clip_negative_values(df)
        
        # Get column names for mains and appliances
        columns = df.columns.tolist()
        logger.info(f"Available columns: {columns}")
        
        # Compute statistics
        stats = compute_statistics(df, columns)
        logger.info("Statistics computed")
        
        return df, stats
    
    def prepare_data_for_appliance(
        self,
        df: pd.DataFrame,
        appliance_key: str,
        appliance_config: ApplianceConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for a single appliance.
        
        Args:
            df: Input DataFrame
            appliance_key: Key for the appliance
            appliance_config: Configuration for the appliance
            
        Returns:
            Tuple of (X_features, y_power, y_state)
        """
        # Assume first column is mains power
        mains_power = df.iloc[:, 0].values
        
        # Find appliance column (try different naming conventions)
        appliance_col = None
        for col in df.columns:
            if appliance_key.lower() in col.lower() or appliance_config.name.lower() in col.lower():
                appliance_col = col
                break
        
        if appliance_col is None:
            logger.warning(f"Could not find column for {appliance_key}, using zeros")
            appliance_power = np.zeros_like(mains_power)
        else:
            appliance_power = df[appliance_col].values
        
        # Compute 8-channel features
        logger.info(f"Computing features for {appliance_key}...")
        X_features = compute_8_channel_features(
            mains_power,
            window_size=self.config.data.rolling_window
        )
        
        # Create target labels
        y_power, y_state = create_target_labels(
            appliance_power,
            appliance_config.thresholds,
            appliance_config.type
        )
        
        logger.info(f"Features shape: {X_features.shape}, Power shape: {y_power.shape}, State shape: {y_state.shape}")
        return X_features, y_power, y_state
    
    def create_windowed_data(
        self,
        X: np.ndarray,
        y_power: np.ndarray,
        y_state: np.ndarray
    ) -> Tuple:
        """
        Create sequence-to-point windowed data.
        
        Args:
            X: Input features
            y_power: Power targets
            y_state: State targets
            
        Returns:
            Tuple of windowed data
        """
        # Combine targets for windowing
        y_combined = np.column_stack([y_power, y_state])
        
        # Create sequences
        X_seq, y_seq = create_sequences(
            X,
            y_combined,
            window_size=self.config.data.window_size,
            midpoint=self.config.data.midpoint
        )
        
        # Split back into power and state
        y_power_seq = y_seq[:, 0]
        y_state_seq = y_seq[:, 1].astype(np.int32)
        
        return X_seq, y_power_seq, y_state_seq
    
    def split_train_val_test(
        self,
        X: np.ndarray,
        y_power: np.ndarray,
        y_state: np.ndarray
    ) -> Tuple:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Input features
            y_power: Power targets
            y_state: State targets
            
        Returns:
            Tuple of split datasets
        """
        # Combine targets for splitting
        y_combined = np.column_stack([y_power, y_state])
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y_combined,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            test_ratio=self.config.data.test_ratio
        )
        
        # Split back into power and state
        y_train_power = y_train[:, 0]
        y_train_state = y_train[:, 1].astype(np.int32)
        
        y_val_power = y_val[:, 0]
        y_val_state = y_val[:, 1].astype(np.int32)
        
        y_test_power = y_test[:, 0]
        y_test_state = y_test[:, 1].astype(np.int32)
        
        return (
            X_train, X_val, X_test,
            y_train_power, y_val_power, y_test_power,
            y_train_state, y_val_state, y_test_state
        )
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        model_type: str = 'cnn_bilstm_attention'
    ) -> keras.Model:
        """
        Build model based on configuration.
        
        Args:
            input_shape: Shape of input
            num_classes: Number of classes
            model_type: Type of model to build
            
        Returns:
            Keras model
        """
        logger.info(f"Building {model_type} model...")
        
        if model_type == 'cnn_bilstm_attention':
            model = build_cnn_bilstm_attention_model(
                input_shape=input_shape,
                num_classes=num_classes,
                cnn_filters=self.config.model.cnn_filters,
                cnn_kernels=self.config.model.cnn_kernels,
                lstm_units=self.config.model.lstm_units,
                attention_heads=self.config.model.attention_heads,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'transformer':
            model = build_transformer_model(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'unet':
            model = build_unet_model(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'resnet_lstm':
            model = build_resnet_lstm_model(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def compile_model(
        self,
        model: keras.Model,
        class_weights: np.ndarray
    ) -> None:
        """
        Compile model with losses and optimizer.
        
        Args:
            model: Keras model
            class_weights: Class weights for focal loss
        """
        # Define losses
        power_loss = CombinedRegressionLoss()
        state_loss = FocalLoss()
        
        # Define optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.model.learning_rate,
            clipnorm=self.config.model.clipnorm
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss={
                'power_output': power_loss,
                'state_output': state_loss
            },
            loss_weights={
                'power_output': 1.0,
                'state_output': 0.5
            },
            metrics={
                'power_output': ['mae', 'mse'],
                'state_output': ['accuracy']
            }
        )
        
        logger.info("Model compiled")
    
    def train_model(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train_power: np.ndarray,
        y_train_state: np.ndarray,
        X_val: np.ndarray,
        y_val_power: np.ndarray,
        y_val_state: np.ndarray,
        appliance_key: str
    ) -> Dict:
        """
        Train model with callbacks.
        
        Args:
            model: Keras model
            X_train: Training features
            y_train_power: Training power targets
            y_train_state: Training state targets
            X_val: Validation features
            y_val_power: Validation power targets
            y_val_state: Validation state targets
            appliance_key: Key for the appliance
            
        Returns:
            Training history
        """
        logger.info(f"Training model for {appliance_key}...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.model.patience,
                min_delta=self.config.model.min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.models_dir, f'{appliance_key}_model_best.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=os.path.join(self.config.metrics_dir, f'{appliance_key}_training_log.csv'),
                append=False
            )
        ]
        
        # Convert state labels to categorical
        y_train_state_cat = keras.utils.to_categorical(y_train_state)
        y_val_state_cat = keras.utils.to_categorical(y_val_state)
        
        # Train model
        history = model.fit(
            X_train,
            {
                'power_output': y_train_power,
                'state_output': y_train_state_cat
            },
            validation_data=(
                X_val,
                {
                    'power_output': y_val_power,
                    'state_output': y_val_state_cat
                }
            ),
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed for {appliance_key}")
        return history.history
    
    def predict_and_postprocess(
        self,
        model: keras.Model,
        X: np.ndarray,
        appliance_config: ApplianceConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and apply post-processing.
        
        Args:
            model: Trained model
            X: Input features
            appliance_config: Appliance configuration
            
        Returns:
            Tuple of (processed power, processed states)
        """
        # Make predictions
        power_pred, state_pred = model.predict(X, verbose=0)
        
        # Convert predictions
        power_pred = power_pred.flatten()
        state_pred = np.argmax(state_pred, axis=-1)
        
        # Apply post-processing
        power_final, state_final = post_process_predictions(
            power_predictions=power_pred,
            state_predictions=state_pred,
            thresholds=appliance_config.thresholds,
            appliance_type=appliance_config.type,
            noise_floor=appliance_config.noise_floor,
            median_filter_size=self.config.post_processing.median_filter_size,
            gaussian_sigma=self.config.post_processing.gaussian_sigma
        )
        
        return power_final, state_final
    
    def train_appliance(
        self,
        df: pd.DataFrame,
        appliance_key: str,
        appliance_config: ApplianceConfig
    ) -> None:
        """
        Train model for a single appliance.
        
        Args:
            df: Input DataFrame
            appliance_key: Key for the appliance
            appliance_config: Configuration for the appliance
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing appliance: {appliance_config.name}")
        logger.info(f"{'='*80}\n")
        
        # Prepare data
        X, y_power, y_state = self.prepare_data_for_appliance(df, appliance_key, appliance_config)
        
        # Create windowed data
        X_seq, y_power_seq, y_state_seq = self.create_windowed_data(X, y_power, y_state)
        
        # Split data
        (X_train, X_val, X_test,
         y_train_power, y_val_power, y_test_power,
         y_train_state, y_val_state, y_test_state) = self.split_train_val_test(
            X_seq, y_power_seq, y_state_seq
        )
        
        # Calculate class weights
        num_classes = len(appliance_config.state_names)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=y_train_state
        )
        
        # Build and compile model
        input_shape = (self.config.data.window_size, self.config.n_channels)
        model = self.build_model(input_shape, num_classes, self.config.model_type)
        self.compile_model(model, class_weights)
        
        # Display model summary
        logger.info(f"\nModel Summary for {appliance_config.name}:")
        model.summary(print_fn=logger.info)
        
        # Train model
        history = self.train_model(
            model, X_train, y_train_power, y_train_state,
            X_val, y_val_power, y_val_state, appliance_key
        )
        
        # Make predictions on test set
        y_pred_power, y_pred_state = self.predict_and_postprocess(
            model, X_test, appliance_config
        )
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            y_test_power, y_pred_power,
            y_test_state, y_pred_state,
            num_classes, appliance_config.name
        )
        
        # Calculate confusion matrix
        cm = calculate_confusion_matrix(y_test_state, y_pred_state, num_classes)
        
        # Store results
        self.models[appliance_key] = model
        self.histories[appliance_key] = history
        self.metrics[appliance_key] = metrics
        self.predictions[appliance_key] = {
            'y_true_power': y_test_power,
            'y_pred_power': y_pred_power,
            'y_true_state': y_test_state,
            'y_pred_state': y_pred_state,
            'confusion_matrix': cm
        }
        
        logger.info(f"Appliance {appliance_config.name} processing completed\n")
    
    def train_all_appliances(self, df: pd.DataFrame) -> None:
        """
        Train models for all appliances.
        
        Args:
            df: Input DataFrame
        """
        for appliance_key, appliance_config in self.config.appliances.items():
            self.train_appliance(df, appliance_key, appliance_config)
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        logger.info("Generating visualizations...")
        
        # Training history plots
        for appliance_key, history in self.histories.items():
            plot_path = os.path.join(
                self.config.plots_dir,
                f'{appliance_key}_training_history.png'
            )
            plot_training_history(history, plot_path)
        
        # Disaggregation plots
        y_true_dict = {
            key: pred['y_true_power']
            for key, pred in self.predictions.items()
        }
        y_pred_dict = {
            key: pred['y_pred_power']
            for key, pred in self.predictions.items()
        }
        
        plot_disaggregation(
            y_true_dict, y_pred_dict,
            os.path.join(self.config.plots_dir, 'disaggregation.png')
        )
        
        # Confusion matrices
        cm_dict = {
            key: pred['confusion_matrix']
            for key, pred in self.predictions.items()
        }
        state_names_dict = {
            key: self.config.appliances[key].state_names
            for key in self.predictions.keys()
        }
        
        plot_confusion_matrices(
            cm_dict, state_names_dict,
            os.path.join(self.config.plots_dir, 'confusion_matrices.png')
        )
        
        # Energy comparison
        plot_energy_comparison(
            y_true_dict, y_pred_dict,
            os.path.join(self.config.plots_dir, 'energy_comparison.png')
        )
        
        # Power distributions
        plot_power_distributions(
            y_true_dict, y_pred_dict,
            os.path.join(self.config.plots_dir, 'power_distributions.png')
        )
        
        logger.info("Visualizations generated")
    
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results...")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(os.path.join(self.config.metrics_dir, 'metrics.csv'))
        
        # Save predictions
        for appliance_key, pred in self.predictions.items():
            save_predictions_to_csv(
                pred['y_true_power'],
                pred['y_pred_power'],
                appliance_key,
                self.config.predictions_dir
            )
        
        # Save configuration
        config_dict = {
            'model_type': self.config.model_type,
            'window_size': self.config.data.window_size,
            'n_channels': self.config.n_channels,
            'appliances': {
                key: {
                    'name': cfg.name,
                    'type': cfg.type,
                    'thresholds': cfg.thresholds,
                    'state_names': cfg.state_names
                }
                for key, cfg in self.config.appliances.items()
            }
        }
        
        with open(os.path.join(self.config.output_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Calculate and save aggregated metrics
        agg_metrics = aggregate_metrics(self.metrics)
        agg_df = pd.DataFrame([agg_metrics])
        agg_df.to_csv(os.path.join(self.config.metrics_dir, 'aggregated_metrics.csv'), index=False)
        
        logger.info("Results saved")
    
    def run(self, data_file: str = None) -> None:
        """
        Run the complete NILM pipeline.
        
        Args:
            data_file: Optional path to data file (overrides config)
        """
        logger.info("\n" + "="*80)
        logger.info("Starting NILM System Pipeline")
        logger.info("="*80 + "\n")
        
        # Update data file if provided
        if data_file:
            self.config.data.data_file = data_file
        
        # Load and preprocess data
        df, stats = self.load_and_preprocess_data()
        
        # Log statistics
        logger.info("\nData Statistics:")
        for col, stat in stats.items():
            logger.info(f"{col}: Mean={stat['mean']:.2f}W, Max={stat['max']:.2f}W")
        
        # Train all appliances
        self.train_all_appliances(df)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("NILM System Pipeline Completed")
        logger.info("="*80)
        logger.info("\nFinal Results Summary:")
        
        agg_metrics = aggregate_metrics(self.metrics)
        logger.info(f"Average MAE: {agg_metrics.get('avg_mae', 0):.2f}W")
        logger.info(f"Average RMSE: {agg_metrics.get('avg_rmse', 0):.2f}W")
        logger.info(f"Average Energy Accuracy: {agg_metrics.get('avg_energy_accuracy', 0):.2f}%")
        logger.info(f"Average F1-Score: {agg_metrics.get('avg_f1_score', 0):.4f}")
        logger.info(f"Average Accuracy: {agg_metrics.get('avg_accuracy', 0):.4f}")
        
        logger.info(f"\nOutputs saved to: {self.config.output_dir}")


def main():
    """Main function to run NILM system."""
    # Create configuration
    config = NILMConfig()
    
    # Create and run NILM system
    system = NILMSystem(config)
    system.run()


if __name__ == '__main__':
    main()
