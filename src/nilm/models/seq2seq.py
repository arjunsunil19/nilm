"""
Seq2Seq Model for NILM

Sequence-to-Sequence model implementation for Non-Intrusive Load Monitoring.
This model outputs a full sequence of appliance power consumption values.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model

from .base_model import BaseNILMModel


class Seq2Seq(BaseNILMModel):
    """
    Sequence-to-Sequence Model for NILM.
    
    Encoder-decoder architecture that takes a sequence of aggregate power
    readings and outputs a sequence of appliance power readings.
    
    Architecture:
        - Encoder: LSTM layers to encode the input sequence
        - Decoder: LSTM layers to decode to target sequence
        - Can be configured for different output lengths
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        output_size: Optional[int] = None,
        learning_rate: float = 1e-4,
        encoder_units: int = 128,
        decoder_units: int = 128,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the Seq2Seq model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input window
            output_size: Size of the output sequence (default: same as input)
            learning_rate: Learning rate for optimizer
            encoder_units: Number of LSTM units in encoder
            decoder_units: Number of LSTM units in decoder
            dropout_rate: Dropout rate
        """
        self.output_size = output_size or window_size
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """
        Build the Seq2Seq model architecture.
        
        Uses an encoder-decoder structure with LSTM layers.
        """
        # Encoder
        encoder_inputs = Input(shape=(self.window_size, 1), name="encoder_input")
        
        # Initial convolution for feature extraction
        x = Conv1D(32, kernel_size=5, activation="relu", padding="same")(encoder_inputs)
        x = Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
        
        # Encoder LSTM
        encoder_lstm = LSTM(
            self.encoder_units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            name="encoder_lstm"
        )
        encoder_outputs, state_h, state_c = encoder_lstm(x)
        encoder_states = [state_h, state_c]
        
        # Decoder
        # Repeat the encoder state for each decoder timestep
        decoder_inputs = RepeatVector(self.output_size)(state_h)
        
        decoder_lstm = LSTM(
            self.decoder_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # Dense output layer (applied to each timestep)
        decoder_dense = TimeDistributed(Dense(64, activation="relu"), name="td_dense")
        x = decoder_dense(decoder_outputs)
        x = Dropout(self.dropout_rate)(x)
        
        output_layer = TimeDistributed(
            Dense(1, activation="linear"),
            name="output"
        )(x)
        
        self.model = Model(
            inputs=encoder_inputs,
            outputs=output_layer,
            name=f"Seq2Seq_{self.appliance_name}"
        )
        
        self.compile_model()


class ConvSeq2Seq(BaseNILMModel):
    """
    Convolutional Sequence-to-Sequence Model for NILM.
    
    Uses only convolutional layers (no LSTM) for faster training
    while maintaining sequence-to-sequence capability.
    """
    
    def __init__(
        self,
        appliance_name: str,
        window_size: int = 599,
        learning_rate: float = 1e-4,
        num_layers: int = 6,
        base_filters: int = 64,
        kernel_size: int = 3,
        dilation_rates: Optional[list] = None,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize the ConvSeq2Seq model.
        
        Args:
            appliance_name: Name of the target appliance
            window_size: Size of the input/output window
            learning_rate: Learning rate
            num_layers: Number of convolutional layers
            base_filters: Base number of filters
            kernel_size: Convolution kernel size
            dilation_rates: List of dilation rates for dilated convolutions
            dropout_rate: Dropout rate
        """
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates or [1, 2, 4, 8, 16, 32]
        self.dropout_rate = dropout_rate
        
        super().__init__(
            appliance_name=appliance_name,
            window_size=window_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def _build_model(self, **kwargs) -> None:
        """Build the ConvSeq2Seq model with dilated convolutions."""
        input_layer = Input(shape=(self.window_size, 1), name="power_input")
        
        # Initial projection
        x = Conv1D(self.base_filters, kernel_size=1, padding="same")(input_layer)
        
        # Dilated convolutional layers
        for i in range(self.num_layers):
            dilation_rate = self.dilation_rates[i % len(self.dilation_rates)]
            
            # Residual block with dilated convolution
            residual = x
            
            x = Conv1D(
                self.base_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                padding="same",
                activation="relu"
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
            x = Conv1D(
                self.base_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                padding="same"
            )(x)
            x = BatchNormalization()(x)
            
            # Residual connection
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.ReLU()(x)
        
        # Output projection
        x = Conv1D(64, kernel_size=1, padding="same", activation="relu")(x)
        output_layer = Conv1D(1, kernel_size=1, padding="same", activation="linear", name="output")(x)
        
        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f"ConvSeq2Seq_{self.appliance_name}"
        )
        
        self.compile_model()
