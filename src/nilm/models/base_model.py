"""
Base NILM Model

Abstract base class for all NILM models providing common functionality.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.models import Model


class BaseNILMModel(ABC):
    """
    Abstract base class for NILM (Non-Intrusive Load Monitoring) models.
    
    This class provides common functionality for all NILM models including
    training, prediction, saving/loading, and evaluation.
    
    Attributes:
        appliance_name: Name of the target appliance
        window_size: Size of the input window (sequence length)
        model: The underlying Keras model
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        """
        Initialize the base NILM model.
        
        Args:
            appliance_name: Name of the target appliance (e.g., 'kettle', 'fridge')
            window_size: Size of the input window (sequence length)
            learning_rate: Learning rate for the optimizer
            **kwargs: Additional arguments for model configuration
        """
        self.appliance_name = appliance_name
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.model: Optional[Model] = None
        self._build_model(**kwargs)
    
    @abstractmethod
    def _build_model(self, **kwargs) -> None:
        """
        Build the model architecture.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Model-specific configuration parameters
        """
        pass
    
    def compile_model(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: str = "mse",
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Compile the model with specified optimizer, loss, and metrics.
        
        Args:
            optimizer: Optimizer instance. If None, uses Adam with the model's learning rate
            loss: Loss function name
            metrics: List of metric names to track
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if metrics is None:
            metrics = ["mae", "mse"]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional)
            y_val: Validation target data (optional)
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Number of epochs with no improvement to trigger early stopping
            reduce_lr_patience: Number of epochs with no improvement to reduce learning rate
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history object
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=verbose
            ),
            ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir,
                    f"{self.__class__.__name__}_{self.appliance_name}_best.h5"
                ),
                monitor="val_loss" if X_val is not None else "loss",
                save_best_only=True,
                verbose=verbose
            ),
            TensorBoard(
                log_dir=os.path.join(log_dir, f"{self.__class__.__name__}_{self.appliance_name}"),
                histogram_freq=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data
            batch_size: Prediction batch size
            
        Returns:
            Model predictions
        """
        return self.model.predict(X, batch_size=batch_size)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            batch_size: Evaluation batch size
            
        Returns:
            Dictionary of metric names and values
        """
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        metric_names = ["loss"] + [m.name if hasattr(m, "name") else str(m) 
                                   for m in self.model.metrics]
        return dict(zip(metric_names, results))
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def summary(self) -> None:
        """Print a summary of the model architecture."""
        self.model.summary()
    
    def get_model(self) -> Model:
        """
        Get the underlying Keras model.
        
        Returns:
            The Keras model instance
        """
        return self.model
