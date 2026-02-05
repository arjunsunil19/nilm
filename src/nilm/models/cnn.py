"""
CNN Model for NILM

1D Convolutional Neural Network implementation for Non-Intrusive Load Monitoring.
Inspired by the approach described in Kelly and Knottenbelt (2015).
"""

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model

from .base_model import BaseNILMModel


class CNN(BaseNILMModel):
    """
    1D CNN Model for NILM.
    
    This model uses multiple 1D convolutional layers to extract features
    from power consumption time series for appliance disaggregation.
    
    Architecture:
        - Multiple 1D convolutional layers with increasing depth
        - MaxPooling for downsampling
        - BatchNormalization for training stability
        - Dense layers for final prediction
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        conv_layers: Optional[List[Tuple[int, int]]] = None,
        dense_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Initialize the CNN model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            learning_rate: Learning rate for optimizer
            conv_layers: List of (filters, kernel_size) tuples for each conv layer
                        Default: [(30, 10), (30, 8), (40, 6), (50, 5), (50, 5)]
            dense_units: List of units for dense layers. Default: [1024, 512]
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        # Default architecture based on Kelly and Knottenbelt
        self.conv_layers = conv_layers or [
            (30, 10),  # (filters, kernel_size)
            (30, 8),
            (40, 6),
            (50, 5),
            (50, 5)
        ]
        self.dense_units = dense_units or [1024, 512]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """
        Build the CNN model architecture.
        
        The architecture follows the approach from:
        "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation"
        by Jack Kelly and William Knottenbelt (2015)
        """
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        x = input_layer
        
        # Convolutional layers
        for i, (filters, kernel_size) in enumerate(self.conv_layers):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                name=f"conv1d_{i}"
            )(x)
            
            if self.use_batch_norm:
                x = BatchNormalization(name=f"bn_{i}")(x)
            
            # Add pooling every 2 layers to reduce dimensions
            if (i + 1) % 2 == 0 and i < len(self.conv_layers) - 1:
                x = MaxPooling1D(pool_size=2, name=f"maxpool_{i}")(x)
            
            x = Dropout(self.dropout_rate)(x)
        
        # Flatten for dense layers
        x = Flatten(name="flatten")(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        output_layer = Dense(1, activation="linear", name="power_output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"CNN_{self.appliance_name}"
        )
        
        self.compile_model()


class DeepCNN(BaseNILMModel):
    """
    Deeper CNN variant with residual connections for improved gradient flow.
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        num_blocks: int = 4,
        base_filters: int = 32,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the DeepCNN model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            learning_rate: Learning rate
            num_blocks: Number of convolutional blocks
            base_filters: Number of filters in the first block (doubles each block)
            dropout_rate: Dropout rate
        """
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """Build the DeepCNN model with residual blocks."""
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        # Initial convolution
        x = Conv1D(self.base_filters, kernel_size=5, padding="same", activation="relu")(input_layer)
        
        # Residual blocks
        for i in range(self.num_blocks):
            filters = self.base_filters * (2 ** min(i, 3))  # Cap at 256 filters
            
            # Main path
            y = Conv1D(filters, kernel_size=3, padding="same", activation="relu")(x)
            y = BatchNormalization()(y)
            y = Conv1D(filters, kernel_size=3, padding="same")(y)
            y = BatchNormalization()(y)
            
            # Skip connection - project if needed
            if x.shape[-1] != filters:
                x = Conv1D(filters, kernel_size=1, padding="same")(x)
            
            x = tf.keras.layers.Add()([x, y])
            x = tf.keras.layers.ReLU()(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Downsample
            if i < self.num_blocks - 1:
                x = MaxPooling1D(pool_size=2)(x)
        
        # Output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation="relu")(x)
        output_layer = Dense(1, activation="linear", name="power_output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"DeepCNN_{self.appliance_name}"
        )
        
        self.compile_model()
