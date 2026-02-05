"""UNet-style model for NILM."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_unet_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    base_filters: int = 64,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build UNet-style model with encoder-decoder architecture.
    
    Args:
        input_shape: Shape of input (window_size, n_channels)
        num_classes: Number of classes for state classification
        base_filters: Number of base filters
        dropout_rate: Dropout rate
        
    Returns:
        Keras model with dual outputs (power, state)
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Encoder
    # Block 1
    conv1 = layers.Conv1D(base_filters, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv1D(base_filters, 3, padding='same', activation='relu')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling1D(2)(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)
    
    # Block 2
    conv2 = layers.Conv1D(base_filters * 2, 3, padding='same', activation='relu', dilation_rate=2)(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv1D(base_filters * 2, 3, padding='same', activation='relu', dilation_rate=2)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling1D(2)(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)
    
    # Block 3
    conv3 = layers.Conv1D(base_filters * 4, 3, padding='same', activation='relu', dilation_rate=4)(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv1D(base_filters * 4, 3, padding='same', activation='relu', dilation_rate=4)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling1D(2)(conv3)
    pool3 = layers.Dropout(dropout_rate)(pool3)
    
    # Bottleneck
    bottleneck = layers.Conv1D(base_filters * 8, 3, padding='same', activation='relu', dilation_rate=8)(pool3)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Conv1D(base_filters * 8, 3, padding='same', activation='relu', dilation_rate=8)(bottleneck)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Dropout(dropout_rate)(bottleneck)
    
    # Decoder
    # Block 4
    up1 = layers.UpSampling1D(2)(bottleneck)
    up1 = layers.Conv1D(base_filters * 4, 2, padding='same', activation='relu')(up1)
    # Adjust shapes for concatenation
    if up1.shape[1] != conv3.shape[1]:
        up1 = layers.Cropping1D(cropping=(0, up1.shape[1] - conv3.shape[1]))(up1)
    merge1 = layers.Concatenate()([conv3, up1])
    conv4 = layers.Conv1D(base_filters * 4, 3, padding='same', activation='relu')(merge1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv1D(base_filters * 4, 3, padding='same', activation='relu')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(dropout_rate)(conv4)
    
    # Block 5
    up2 = layers.UpSampling1D(2)(conv4)
    up2 = layers.Conv1D(base_filters * 2, 2, padding='same', activation='relu')(up2)
    if up2.shape[1] != conv2.shape[1]:
        up2 = layers.Cropping1D(cropping=(0, up2.shape[1] - conv2.shape[1]))(up2)
    merge2 = layers.Concatenate()([conv2, up2])
    conv5 = layers.Conv1D(base_filters * 2, 3, padding='same', activation='relu')(merge2)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv1D(base_filters * 2, 3, padding='same', activation='relu')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(dropout_rate)(conv5)
    
    # Block 6
    up3 = layers.UpSampling1D(2)(conv5)
    up3 = layers.Conv1D(base_filters, 2, padding='same', activation='relu')(up3)
    if up3.shape[1] != conv1.shape[1]:
        up3 = layers.Cropping1D(cropping=(0, up3.shape[1] - conv1.shape[1]))(up3)
    merge3 = layers.Concatenate()([conv1, up3])
    conv6 = layers.Conv1D(base_filters, 3, padding='same', activation='relu')(merge3)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv1D(base_filters, 3, padding='same', activation='relu')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(dropout_rate)(conv6)
    
    # Global pooling
    x_avg = layers.GlobalAveragePooling1D()(conv6)
    x_max = layers.GlobalMaxPooling1D()(conv6)
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
        name='unet'
    )
    
    return model
