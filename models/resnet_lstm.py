"""ResNet-LSTM Hybrid model for NILM."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class ResidualBlock(layers.Layer):
    """Residual block with skip connections."""
    
    def __init__(self, filters: int, kernel_size: int = 3, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.activation = layers.Activation('relu')
    
    def build(self, input_shape):
        # Add projection if input channels don't match output channels
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv1D(self.filters, 1, padding='same')
        else:
            self.projection = None
    
    def call(self, inputs, training=False):
        # First convolution
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.projection is not None:
            shortcut = self.projection(inputs)
        else:
            shortcut = inputs
        
        x = layers.Add()([x, shortcut])
        x = self.activation(x)
        x = self.dropout2(x, training=training)
        
        return x


def build_resnet_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    base_filters: int = 64,
    num_res_blocks: int = 3,
    lstm_units: int = 128,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build ResNet-LSTM Hybrid model with dual heads.
    
    Args:
        input_shape: Shape of input (window_size, n_channels)
        num_classes: Number of classes for state classification
        base_filters: Number of base filters
        num_res_blocks: Number of residual blocks
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        
    Returns:
        Keras model with dual outputs (power, state)
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial convolution
    x = layers.Conv1D(base_filters, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Stack residual blocks
    for i in range(num_res_blocks):
        filters = base_filters * (2 ** min(i, 2))  # Increase filters: 64, 128, 256, ...
        x = ResidualBlock(
            filters=filters,
            kernel_size=3,
            dropout_rate=dropout_rate,
            name=f'res_block_{i}'
        )(x)
        
        # Add pooling every 2 blocks
        if (i + 1) % 2 == 0 and i < num_res_blocks - 1:
            x = layers.MaxPooling1D(2)(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
        name='bilstm_1'
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate),
        name='bilstm_2'
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling
    x_avg = layers.GlobalAveragePooling1D()(x)
    x_max = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([x_avg, x_max])
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dual heads
    # Power regression head
    power_output = layers.Dense(64, activation='relu', name='power_dense')(x)
    power_output = layers.Dropout(dropout_rate)(power_output)
    power_output = layers.Dense(1, activation='relu', name='power_output')(power_output)
    
    # State classification head
    state_output = layers.Dense(64, activation='relu', name='state_dense')(x)
    state_output = layers.Dropout(dropout_rate)(state_output)
    state_output = layers.Dense(num_classes, activation='softmax', name='state_output')(state_output)
    
    # Create model
    model = keras.Model(
        inputs=inputs,
        outputs=[power_output, state_output],
        name='resnet_lstm'
    )
    
    return model
