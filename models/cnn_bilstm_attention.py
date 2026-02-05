"""CNN-BiLSTM-Attention model for NILM."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List


class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(channels // self.ratio, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels))
        self.multiply = layers.Multiply()
    
    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        attention_output = tf.matmul(attention_weights, v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        output = self.dense(attention_output)
        return output


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Generate positional encoding."""
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate([
        np.sin(angle_rads[:, 0::2]),
        np.cos(angle_rads[:, 1::2])
    ], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


def build_cnn_bilstm_attention_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    cnn_filters: int = 64,
    cnn_kernels: List[int] = [3, 5, 7, 9],
    lstm_units: List[int] = [128, 64],
    attention_heads: int = 4,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build CNN-BiLSTM-Attention model with dual heads.
    
    Args:
        input_shape: Shape of input (window_size, n_channels)
        num_classes: Number of classes for state classification
        cnn_filters: Number of filters in CNN layers
        cnn_kernels: List of kernel sizes for multi-scale CNN
        lstm_units: List of LSTM units for each layer
        attention_heads: Number of attention heads
        dropout_rate: Dropout rate
        
    Returns:
        Keras model with dual outputs (power, state)
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Add positional encoding
    pos_enc = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_enc
    
    # Multi-scale CNN with parallel kernels
    conv_outputs = []
    for kernel_size in cnn_kernels:
        conv = layers.Conv1D(
            filters=cnn_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=f'conv_{kernel_size}'
        )(x)
        conv = layers.BatchNormalization()(conv)
        conv_outputs.append(conv)
    
    # Concatenate multi-scale features
    if len(conv_outputs) > 1:
        x = layers.Concatenate()(conv_outputs)
    else:
        x = conv_outputs[0]
    
    # Squeeze-and-Excitation block
    x = SqueezeExcitation(ratio=16)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Bidirectional LSTM layers
    for i, units in enumerate(lstm_units):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, dropout=dropout_rate),
            name=f'bilstm_{i}'
        )(x)
        x = layers.BatchNormalization()(x)
    
    # Multi-Head Self-Attention
    attention_output = MultiHeadSelfAttention(
        d_model=x.shape[-1],
        num_heads=attention_heads,
        name='multi_head_attention'
    )(x)
    
    # Add & Norm (residual connection)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
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
        name='cnn_bilstm_attention'
    )
    
    return model


class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Convert labels to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)


class CombinedRegressionLoss(keras.losses.Loss):
    """Combined MAE + Huber + MSE loss for power regression."""
    
    def __init__(self, mae_weight=1.0, huber_weight=1.0, mse_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        self.mse_weight = mse_weight
        self.mae = keras.losses.MeanAbsoluteError()
        self.huber = keras.losses.Huber(delta=1.0)
        self.mse = keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        mae_loss = self.mae(y_true, y_pred)
        huber_loss = self.huber(y_true, y_pred)
        mse_loss = self.mse(y_true, y_pred)
        
        total_loss = (
            self.mae_weight * mae_loss +
            self.huber_weight * huber_loss +
            self.mse_weight * mse_loss
        )
        
        return total_loss
