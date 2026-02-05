"""
BiLSTM Model for NILM

Bidirectional LSTM implementation for Non-Intrusive Load Monitoring.
This model processes sequences in both forward and backward directions,
capturing temporal dependencies from both past and future contexts.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    Reshape,
)
from tensorflow.keras.models import Model

from .base_model import BaseNILMModel


class BiLSTM(BaseNILMModel):
    """
    Bidirectional LSTM Model for NILM.
    
    This model uses bidirectional LSTM layers to process power consumption
    sequences and disaggregate individual appliance consumption.
    
    Architecture:
        1. Optional 1D Convolutional layer for feature extraction
        2. Multiple Bidirectional LSTM layers
        3. Dense layers for final prediction
    
    Attributes:
        lstm_units: List of units for each LSTM layer
        use_conv: Whether to use initial convolutional layer
        conv_filters: Number of convolutional filters (if use_conv=True)
        dropout_rate: Dropout rate between layers
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        lstm_units: Optional[list] = None,
        use_conv: bool = True,
        conv_filters: int = 32,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the BiLSTM model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            learning_rate: Learning rate for optimizer
            lstm_units: List of LSTM units per layer (default: [128, 64])
            use_conv: Whether to use initial 1D convolution
            conv_filters: Number of convolution filters
            dropout_rate: Dropout rate
            **kwargs: Additional arguments
        """
        self.lstm_units = lstm_units or [128, 64]
        self.use_conv = use_conv
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """
        Build the BiLSTM model architecture.
        
        The model consists of:
        - Optional Conv1D layer for local feature extraction
        - Stack of Bidirectional LSTM layers
        - Dense layers for output prediction
        """
        # Input layer
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        x = input_layer
        
        # Optional convolutional feature extraction
        if self.use_conv:
            x = Conv1D(
                filters=self.conv_filters,
                kernel_size=4,
                activation="relu",
                padding="same",
                name="conv1d_features"
            )(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    name=f"lstm_{i}"
                ),
                name=f"bilstm_{i}"
            )(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Dense output layers
        x = Dense(128, activation="relu", name="dense_1")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation="relu", name="dense_2")(x)
        
        # Output layer - single value for point prediction
        output_layer = Dense(1, activation="linear", name="power_output")(x)
        
        # Create model
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"BiLSTM_{self.appliance_name}"
        )
        
        # Compile with default settings
        self.compile_model()


class BiLSTMSeq2Seq(BaseNILMModel):
    """
    BiLSTM Sequence-to-Sequence Model for NILM.
    
    This variant outputs a full sequence instead of a single point,
    making it suitable for sequence-to-sequence energy disaggregation.
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        output_size: Optional[int] = None,
        learning_rate: float = 1e-4,
        lstm_units: Optional[list] = None,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the BiLSTM Seq2Seq model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            output_size: Size of the output sequence (default: same as input)
            learning_rate: Learning rate
            lstm_units: List of LSTM units per layer
            dropout_rate: Dropout rate
        """
        self.output_size = output_size or window_size
        self.lstm_units = lstm_units or [128, 128, 64]
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """Build the BiLSTM Seq2Seq model architecture."""
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        x = input_layer
        
        # Initial conv layer
        x = Conv1D(filters=32, kernel_size=4, activation="relu", padding="same")(x)
        
        # Bidirectional LSTM encoder
        for i, units in enumerate(self.lstm_units):
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=True,
                    activation="tanh",
                    name=f"lstm_{i}"
                ),
                name=f"bilstm_{i}"
            )(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output projection
        x = Conv1D(filters=64, kernel_size=4, activation="relu", padding="same")(x)
        output_layer = Conv1D(filters=1, kernel_size=1, activation="linear", name="output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"BiLSTM_Seq2Seq_{self.appliance_name}"
        )
        
        self.compile_model()
