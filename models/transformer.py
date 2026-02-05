"""Transformer-based model for NILM."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class TransformerBlock(layers.Layer):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def build_transformer_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    d_model: int = 128,
    num_heads: int = 8,
    ff_dim: int = 512,
    num_blocks: int = 4,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build Transformer-based model with dual heads.
    
    Args:
        input_shape: Shape of input (window_size, n_channels)
        num_classes: Number of classes for state classification
        d_model: Model dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        num_blocks: Number of transformer blocks
        dropout_rate: Dropout rate
        
    Returns:
        Keras model with dual outputs (power, state)
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Patch embedding (linear projection)
    x = layers.Dense(d_model)(inputs)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)
    x = x + pos_embedding
    
    # Stack transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'transformer_block_{i}'
        )(x)
    
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
        name='transformer'
    )
    
    return model
