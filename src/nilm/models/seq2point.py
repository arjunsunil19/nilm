"""
Seq2Point Model for NILM

Sequence-to-Point model implementation for Non-Intrusive Load Monitoring.
Based on Zhang et al. (2018) - "Sequence-to-point learning with neural networks
for non-intrusive load monitoring"
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
)
from tensorflow.keras.models import Model

from .base_model import BaseNILMModel


class Seq2Point(BaseNILMModel):
    """
    Sequence-to-Point (Seq2Point) Model for NILM.
    
    This model takes a window of aggregate power readings and predicts
    the power consumption of a single appliance at the midpoint of the window.
    
    Reference:
        Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018).
        Sequence-to-point learning with neural networks for non-intrusive load monitoring.
        In Proceedings of the AAAI Conference on Artificial Intelligence.
    
    Architecture:
        - 5 Convolutional layers with varying kernel sizes
        - Dense layers for final prediction
        - Predicts the midpoint value
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the Seq2Point model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window (should be odd for clear midpoint)
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate
        """
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """
        Build the Seq2Point model architecture.
        
        Architecture based on the original paper:
        - Conv1D(30, 10)
        - Conv1D(30, 8)
        - Conv1D(40, 6)
        - Conv1D(50, 5)
        - Conv1D(50, 5)
        - Dense(1024)
        - Dense(1)
        """
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        # Convolutional layers - architecture from the paper
        x = Conv1D(
            filters=30,
            kernel_size=10,
            activation="relu",
            padding="same",
            name="conv1"
        )(input_layer)
        
        x = Conv1D(
            filters=30,
            kernel_size=8,
            activation="relu",
            padding="same",
            name="conv2"
        )(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Conv1D(
            filters=40,
            kernel_size=6,
            activation="relu",
            padding="same",
            name="conv3"
        )(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Conv1D(
            filters=50,
            kernel_size=5,
            activation="relu",
            padding="same",
            name="conv4"
        )(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Conv1D(
            filters=50,
            kernel_size=5,
            activation="relu",
            padding="same",
            name="conv5"
        )(x)
        
        # Flatten and dense
        x = Flatten(name="flatten")(x)
        x = Dense(1024, activation="relu", name="dense1")(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output - single point prediction (midpoint)
        output_layer = Dense(1, activation="linear", name="power_output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"Seq2Point_{self.appliance_name}"
        )
        
        self.compile_model()


class Seq2PointAttention(BaseNILMModel):
    """
    Seq2Point model with attention mechanism.
    
    Enhanced version of Seq2Point that uses attention to focus on
    relevant parts of the input sequence.
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        attention_heads: int = 4,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the Seq2Point with Attention model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            learning_rate: Learning rate
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """Build the Seq2Point model with attention."""
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        # Convolutional feature extraction
        x = Conv1D(64, kernel_size=10, activation="relu", padding="same")(input_layer)
        x = Conv1D(64, kernel_size=8, activation="relu", padding="same")(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=64,
            dropout=self.dropout_rate,
            name="attention"
        )(x, x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # More convolutions
        x = Conv1D(128, kernel_size=6, activation="relu", padding="same")(x)
        x = Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        output_layer = Dense(1, activation="linear", name="power_output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"Seq2PointAttention_{self.appliance_name}"
        )
        
        self.compile_model()
