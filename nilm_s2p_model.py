"""
Non-Intrusive Load Monitoring (NILM) System using Sequence-to-Point (S2P) Architecture

This module implements a "near-perfect" NILM system for energy disaggregation using a 
Hybrid CNN-BiLSTM-Attention model with Dual-Head outputs (Regression + Classification).

Architecture: Sequence-to-Point (S2P) Paradigm (Zhang et al.)
- Input: Window of 99 samples of the 'Mains' signal with 4 feature channels
- Output: Predicted power (Watts) and operating state at the central midpoint (t=50)

Author: NILM Research Team
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# Configuration and Constants
# =============================================================================

# S2P Window Configuration
WINDOW_SIZE = 99
MIDPOINT = 49  # Central sample index (0-indexed, corresponds to t=50)

# Model Hyperparameters
LSTM_UNITS = 128
CNN_FILTERS = [32, 64, 128]
ATTENTION_HEADS = 4
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Training Configuration
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Post-Processing Configuration
MEDIAN_FILTER_SIZE = 3
NOISE_FLOOR = 10  # Watts

# Appliance Names
APPLIANCES = ['Air_Conditioner', 'Refrigerator', 'Fan', 'Washing_Machine', 'EV_Charger']

# Multi-State Thresholds (Watts) based on physical characteristics
STATE_THRESHOLDS = {
    'Air_Conditioner': {
        'OFF': (0, 50),
        'ON': (50, float('inf'))
    },
    'Refrigerator': {
        'OFF': (0, 20),
        'ON': (20, float('inf'))
    },
    'Fan': {
        'OFF': (0, 10),
        'Low-Speed': (10, 30),
        'Medium-Speed': (30, 50),
        'High-Speed': (50, float('inf'))
    },
    'Washing_Machine': {
        'OFF': (0, 10),
        'Wash': (10, 200),
        'Rinse': (200, 400),
        'Spin': (400, float('inf'))
    },
    'EV_Charger': {
        'OFF': (0, 50),
        'ON': (50, float('inf'))
    }
}

# Number of states per appliance
NUM_STATES = {
    'Air_Conditioner': 2,
    'Refrigerator': 2,
    'Fan': 4,
    'Washing_Machine': 4,
    'EV_Charger': 2
}

# State Labels for each appliance
STATE_LABELS = {
    'Air_Conditioner': ['OFF', 'ON'],
    'Refrigerator': ['OFF', 'ON'],
    'Fan': ['OFF', 'Low-Speed', 'Medium-Speed', 'High-Speed'],
    'Washing_Machine': ['OFF', 'Wash', 'Rinse', 'Spin'],
    'EV_Charger': ['OFF', 'ON']
}


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    """Advanced feature engineering for NILM with 4 input channels."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def fit(self, mains_power):
        """Fit the RobustScaler on the training data."""
        self.scaler.fit(mains_power.reshape(-1, 1))
        self.is_fitted = True
    
    def transform(self, mains_power):
        """
        Create 4 input channels from the mains power signal.
        
        Channels:
        1. Standardized Mains Power (RobustScaler)
        2. Differential Power (captures edge transients)
        3. Second-derivative of Power (switching acceleration)
        4. Rolling 5-sample Standard Deviation (noise textures)
        """
        if not self.is_fitted:
            self.fit(mains_power)
        
        n_samples = len(mains_power)
        features = np.zeros((n_samples, 4))
        
        # Channel 1: Standardized Mains Power
        features[:, 0] = self.scaler.transform(mains_power.reshape(-1, 1)).flatten()
        
        # Channel 2: Differential Power (first derivative)
        diff_power = np.diff(mains_power, prepend=mains_power[0])
        std_diff = np.std(diff_power)
        features[:, 1] = diff_power / (std_diff + 1e-8) if std_diff > 0 else diff_power
        
        # Channel 3: Second-derivative of Power (switching acceleration)
        second_diff = np.diff(diff_power, prepend=diff_power[0])
        std_second = np.std(second_diff)
        features[:, 2] = second_diff / (std_second + 1e-8) if std_second > 0 else second_diff
        
        # Channel 4: Rolling 5-sample Standard Deviation (noise textures)
        rolling_std = pd.Series(mains_power).rolling(window=5, min_periods=1).std().fillna(0).values
        std_rolling = np.std(rolling_std)
        features[:, 3] = rolling_std / (std_rolling + 1e-8) if std_rolling > 0 else rolling_std
        
        # Clip extreme values
        features = np.clip(features, -10, 10)
        
        # Replace any NaN or inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


def create_s2p_windows(features, targets, window_size=WINDOW_SIZE, midpoint=MIDPOINT):
    """
    Create Sequence-to-Point windows for training.
    
    Args:
        features: Input features (n_samples, n_channels)
        targets: Target values (n_samples, n_appliances) for regression or states
        window_size: Size of the input window (default: 99)
        midpoint: Index of the target sample in the window (default: 49)
    
    Returns:
        X: Windowed input features (n_windows, window_size, n_channels)
        y: Target values at midpoint (n_windows, n_appliances)
    """
    n_samples = len(features)
    n_channels = features.shape[1] if len(features.shape) > 1 else 1
    
    # Calculate padding needed
    pad_before = midpoint
    pad_after = window_size - midpoint - 1
    
    # Pad features using reflection to avoid edge effects
    if n_channels > 1:
        padded_features = np.pad(features, ((pad_before, pad_after), (0, 0)), mode='reflect')
    else:
        padded_features = np.pad(features, (pad_before, pad_after), mode='reflect')
    
    # Create windows
    X = np.zeros((n_samples, window_size, n_channels))
    for i in range(n_samples):
        X[i] = padded_features[i:i + window_size]
    
    return X, targets


# =============================================================================
# Attention Mechanism
# =============================================================================

class SelfAttentionLayer(layers.Layer):
    """Self-Attention layer to weight switching events more heavily."""
    
    def __init__(self, units, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W_q = self.add_weight(
            name='W_query',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_k = self.add_weight(
            name='W_key',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='W_value',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfAttentionLayer, self).build(input_shape)
    
    def call(self, x, training=None):
        # Query, Key, Value projections
        Q = tf.matmul(x, self.W_q)
        K = tf.matmul(x, self.W_k)
        V = tf.matmul(x, self.W_v)
        
        # Scaled dot-product attention
        d_k = tf.cast(self.units, tf.float32)
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, V)
        
        return output
    
    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


# =============================================================================
# Hybrid CNN-BiLSTM-Attention Model with Dual-Head Outputs
# =============================================================================

def build_nilm_model(window_size=WINDOW_SIZE, n_channels=4, n_appliances=5):
    """
    Build the Hybrid CNN-BiLSTM-Attention model with Dual-Head outputs.
    
    Architecture:
    1. 1D-CNN layers for feature extraction (sub-second transients)
    2. Bidirectional LSTM for temporal modeling (cycles)
    3. Self-Attention layer for weighting switching events
    4. Dual-Head outputs: Regression (Power) + Classification (States)
    """
    
    # Input Layer
    inputs = layers.Input(shape=(window_size, n_channels), name='mains_input')
    
    # ==========================================================================
    # Feature Extraction: 1D-CNN Layers for Sub-Second Transients
    # ==========================================================================
    x = layers.Conv1D(filters=CNN_FILTERS[0], kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=CNN_FILTERS[1], kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=CNN_FILTERS[2], kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # ==========================================================================
    # Temporal Modeling: Bidirectional LSTM
    # ==========================================================================
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # ==========================================================================
    # Attention Mechanism: Self-Attention Layer
    # ==========================================================================
    x = SelfAttentionLayer(units=LSTM_UNITS)(x)
    x = layers.LayerNormalization()(x)
    
    # Global pooling to get fixed-size representation
    x = layers.GlobalAveragePooling1D()(x)
    
    # Shared dense layers
    shared = layers.Dense(256, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(DROPOUT_RATE)(shared)
    
    shared = layers.Dense(128, activation='relu')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(DROPOUT_RATE)(shared)
    
    # ==========================================================================
    # Dual-Head Branching for Each Appliance
    # ==========================================================================
    regression_outputs = []
    classification_outputs = []
    
    for i, appliance in enumerate(APPLIANCES):
        # Appliance-specific branch with L2 regularization for stability
        app_branch = layers.Dense(64, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                  name=f'{appliance}_branch')(shared)
        app_branch = layers.Dropout(DROPOUT_RATE)(app_branch)
        
        # Regression Head: Predicts Active Power (Watts)
        # Using linear activation then clip to ensure zero-bounded output
        reg_output = layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(app_branch)
        reg_output = layers.Dense(1, activation='linear', name=f'{appliance}_power')(reg_output)
        regression_outputs.append(reg_output)
        
        # Classification Head: Predicts Operating State (Multi-class)
        n_states = NUM_STATES[appliance]
        cls_output = layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(app_branch)
        cls_output = layers.Dense(n_states, activation='softmax', name=f'{appliance}_state')(cls_output)
        classification_outputs.append(cls_output)
    
    # Concatenate all outputs
    all_outputs = regression_outputs + classification_outputs
    
    # Build model
    model = Model(inputs=inputs, outputs=all_outputs)
    
    return model


# =============================================================================
# Custom Losses
# =============================================================================

def weighted_categorical_crossentropy(class_weights):
    """
    Create a weighted categorical cross-entropy loss function.
    Robust implementation that handles class imbalance.
    """
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Standard categorical cross-entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Get sample weights based on true class
        sample_weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        
        # Apply weights
        weighted_ce = ce * sample_weights
        
        return tf.reduce_mean(weighted_ce)
    
    return loss


def compute_class_weights(y_states, n_classes):
    """Compute class weights based on class frequency."""
    class_counts = np.bincount(y_states.astype(int), minlength=n_classes)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
    total = np.sum(class_counts)
    weights = total / (n_classes * class_counts)
    # Normalize and clip to prevent extreme values
    weights = np.clip(weights, 0.5, 5.0)
    return weights


def compile_model(model, class_weights_dict):
    """Compile the model with Huber loss for regression and weighted CE for classification."""
    
    losses = {}
    loss_weights = {}
    
    for appliance in APPLIANCES:
        # Huber loss for regression (robust to scale differences)
        # Use higher delta for larger power ranges
        losses[f'{appliance}_power'] = tf.keras.losses.Huber(delta=10.0)
        loss_weights[f'{appliance}_power'] = 1.0
        
        # Weighted categorical cross-entropy for classification
        weights = class_weights_dict.get(appliance, np.ones(NUM_STATES[appliance]))
        losses[f'{appliance}_state'] = weighted_categorical_crossentropy(weights)
        loss_weights[f'{appliance}_state'] = 0.5  # Lower weight for classification
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={f'{app}_power': 'mae' for app in APPLIANCES}
    )
    
    return model


# =============================================================================
# State Mapping Functions
# =============================================================================

def power_to_state(power, appliance):
    """Convert power value to operating state based on thresholds."""
    thresholds = STATE_THRESHOLDS[appliance]
    for state_name, (low, high) in thresholds.items():
        if low <= power < high:
            return state_name
    return 'OFF'


def power_array_to_states(power_array, appliance):
    """Convert array of power values to state labels."""
    return np.array([power_to_state(p, appliance) for p in power_array])


def encode_states(states, appliance):
    """Encode state labels to integers."""
    state_labels = STATE_LABELS[appliance]
    label_to_idx = {label: idx for idx, label in enumerate(state_labels)}
    return np.array([label_to_idx.get(s, 0) for s in states])


def states_to_onehot(states, n_classes):
    """Convert state indices to one-hot encoding."""
    return tf.keras.utils.to_categorical(states, num_classes=n_classes)


# =============================================================================
# Post-Processing
# =============================================================================

def apply_post_processing(predictions, median_size=MEDIAN_FILTER_SIZE, noise_floor=NOISE_FLOOR):
    """
    Apply post-processing to the predicted power values.
    
    1. 3-tap Median Filter to smooth the output
    2. Hard 10W noise floor to ensure clean traces
    """
    # Apply median filter
    smoothed = median_filter(predictions, size=median_size)
    
    # Apply noise floor (set values below threshold to 0)
    cleaned = np.where(smoothed < noise_floor, 0, smoothed)
    
    return cleaned


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_prepare_data(filepath):
    """Load the dataset and prepare it for training."""
    
    # Load the Excel file
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract mains power
    mains_col = 'Total_ActivePower(W)'
    if mains_col not in df.columns:
        # Try alternative column names
        for col in df.columns:
            if 'total' in col.lower() and 'power' in col.lower():
                mains_col = col
                break
    
    mains_power = df[mains_col].values
    
    # Extract individual appliance powers
    appliance_columns = {}
    for appliance in APPLIANCES:
        # Find matching column
        for col in df.columns:
            if appliance.replace('_', ' ').lower() in col.lower() or \
               appliance.replace('_', '').lower() in col.lower():
                appliance_columns[appliance] = col
                break
        
        if appliance not in appliance_columns:
            # Try to find by partial match
            for col in df.columns:
                if any(word in col.lower() for word in appliance.lower().split('_')):
                    if col not in appliance_columns.values():
                        appliance_columns[appliance] = col
                        break
    
    print(f"\nAppliance column mapping: {appliance_columns}")
    
    # Get appliance power values
    appliance_power = {}
    for appliance in APPLIANCES:
        if appliance in appliance_columns:
            appliance_power[appliance] = df[appliance_columns[appliance]].values
        else:
            # If column not found, create dummy zeros
            print(f"Warning: Column for {appliance} not found. Using zeros.")
            appliance_power[appliance] = np.zeros(len(mains_power))
    
    return mains_power, appliance_power, df


def prepare_training_data(mains_power, appliance_power, normalize_targets=True):
    """Prepare training data with feature engineering."""
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    features = feature_engineer.transform(mains_power)
    print(f"\nFeature shape: {features.shape}")
    
    # Prepare targets
    n_samples = len(mains_power)
    
    # Compute power scalers for each appliance (for denormalization)
    power_scalers = {}
    
    # Regression targets (power values)
    y_power = np.zeros((n_samples, len(APPLIANCES)))
    for i, appliance in enumerate(APPLIANCES):
        power_vals = appliance_power[appliance].copy()
        power_vals = np.maximum(power_vals, 0)  # Ensure non-negative
        
        if normalize_targets:
            # Normalize to [0, 1] range based on max power
            max_power = np.max(power_vals) + 1e-8
            power_scalers[appliance] = max_power
            y_power[:, i] = power_vals / max_power
        else:
            power_scalers[appliance] = 1.0
            y_power[:, i] = power_vals
    
    # Classification targets (states)
    y_states = {}
    class_weights = {}
    
    for i, appliance in enumerate(APPLIANCES):
        # Use original (non-normalized) power for state classification
        states = power_array_to_states(appliance_power[appliance], appliance)
        state_indices = encode_states(states, appliance)
        y_states[appliance] = states_to_onehot(state_indices, NUM_STATES[appliance])
        class_weights[appliance] = compute_class_weights(state_indices, NUM_STATES[appliance])
    
    # Create S2P windows
    X, _ = create_s2p_windows(features, y_power)
    
    # Ensure no NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_power = np.nan_to_num(y_power, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split indices (keeping temporal order)
    train_size = int(0.8 * n_samples)
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    # Prepare output dictionary for training
    y_train = {}
    y_test = {}
    
    for i, appliance in enumerate(APPLIANCES):
        y_train[f'{appliance}_power'] = y_power[:train_size, i:i+1]
        y_test[f'{appliance}_power'] = y_power[train_size:, i:i+1]
        y_train[f'{appliance}_state'] = y_states[appliance][:train_size]
        y_test[f'{appliance}_state'] = y_states[appliance][train_size:]
    
    return X_train, X_test, y_train, y_test, class_weights, feature_engineer, power_scalers


# =============================================================================
# Visualization
# =============================================================================

def plot_regression_results(y_true, y_pred, appliance, save_path=None):
    """Plot actual vs predicted power for an appliance."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series comparison
    ax1 = axes[0]
    time_range = range(len(y_true))
    ax1.plot(time_range, y_true, label='Actual', alpha=0.8, linewidth=1)
    ax1.plot(time_range, y_pred, label='Predicted', alpha=0.8, linewidth=1)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Power (W)')
    ax1.set_title(f'{appliance.replace("_", " ")}: Actual vs Predicted Power')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=5)
    max_val = max(np.max(y_true), np.max(y_pred))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual Power (W)')
    ax2.set_ylabel('Predicted Power (W)')
    ax2.set_title(f'{appliance.replace("_", " ")}: Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, appliance, save_path=None):
    """Plot confusion matrix for appliance states."""
    
    labels = STATE_LABELS[appliance]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted State')
    ax.set_ylabel('Actual State')
    ax.set_title(f'{appliance.replace("_", " ")}: Multi-Class Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def create_summary_plot(performance_summary, save_path=None):
    """Create a summary visualization of model performance."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    appliances = list(performance_summary.keys())
    
    # MAE plot
    mae_values = [performance_summary[app]['MAE (W)'] for app in appliances]
    axes[0].barh(appliances, mae_values, color='steelblue')
    axes[0].set_xlabel('MAE (Watts)')
    axes[0].set_title('Mean Absolute Error')
    for i, v in enumerate(mae_values):
        axes[0].text(v + 0.1, i, f'{v:.2f}', va='center')
    
    # F1-Score plot
    f1_values = [performance_summary[app]['F1-Score'] for app in appliances]
    axes[1].barh(appliances, f1_values, color='forestgreen')
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('Classification F1-Score')
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(f1_values):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Energy Accuracy plot
    energy_acc = [performance_summary[app]['Energy Accuracy (%)'] for app in appliances]
    axes[2].barh(appliances, energy_acc, color='coral')
    axes[2].set_xlabel('Energy Accuracy (%)')
    axes[2].set_title('Energy Disaggregation Accuracy')
    axes[2].set_xlim(0, 100)
    for i, v in enumerate(energy_acc):
        axes[2].text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_energy_accuracy(y_true, y_pred):
    """Calculate energy accuracy as percentage."""
    total_true = np.sum(np.abs(y_true))
    total_pred = np.sum(np.abs(y_pred))
    
    if total_true == 0:
        return 100.0 if total_pred == 0 else 0.0
    
    accuracy = 100 * (1 - np.abs(total_true - total_pred) / total_true)
    return max(0, min(100, accuracy))


def evaluate_model(model, X_test, y_test, power_scalers=None):
    """Evaluate the model and compute performance metrics."""
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Handle predictions (model has multiple outputs)
    n_appliances = len(APPLIANCES)
    power_predictions = predictions[:n_appliances]
    state_predictions = predictions[n_appliances:]
    
    performance_summary = {}
    
    for i, appliance in enumerate(APPLIANCES):
        # Get ground truth
        y_true_power = y_test[f'{appliance}_power'].flatten()
        y_true_state = np.argmax(y_test[f'{appliance}_state'], axis=1)
        
        # Get predictions
        y_pred_power_raw = power_predictions[i].flatten()
        y_pred_state = np.argmax(state_predictions[i], axis=1)
        
        # Denormalize if scalers provided
        if power_scalers is not None and appliance in power_scalers:
            scale = power_scalers[appliance]
            y_true_power = y_true_power * scale
            y_pred_power_raw = y_pred_power_raw * scale
        
        # Ensure non-negative predictions
        y_pred_power_raw = np.maximum(y_pred_power_raw, 0)
        
        # Apply post-processing to power predictions
        y_pred_power = apply_post_processing(y_pred_power_raw)
        
        # Handle any remaining NaN values
        y_pred_power = np.nan_to_num(y_pred_power, nan=0.0, posinf=0.0, neginf=0.0)
        y_true_power = np.nan_to_num(y_true_power, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_power, y_pred_power)
        f1 = f1_score(y_true_state, y_pred_state, average='weighted', zero_division=0)
        energy_acc = calculate_energy_accuracy(y_true_power, y_pred_power)
        
        performance_summary[appliance] = {
            'MAE (W)': mae,
            'F1-Score': f1,
            'Energy Accuracy (%)': energy_acc,
            'y_true_power': y_true_power,
            'y_pred_power': y_pred_power,
            'y_true_state': y_true_state,
            'y_pred_state': y_pred_state
        }
    
    return performance_summary


def print_performance_table(performance_summary):
    """Print formatted performance summary table."""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Appliance':<20} {'MAE (W)':<12} {'F1-Score':<12} {'Energy Acc (%)':<15}")
    print("-" * 70)
    
    for appliance in APPLIANCES:
        metrics = performance_summary[appliance]
        print(f"{appliance.replace('_', ' '):<20} "
              f"{metrics['MAE (W)']:<12.2f} "
              f"{metrics['F1-Score']:<12.3f} "
              f"{metrics['Energy Accuracy (%)']:<15.1f}")
    
    print("=" * 70)
    
    # Calculate averages
    avg_mae = np.mean([performance_summary[app]['MAE (W)'] for app in APPLIANCES])
    avg_f1 = np.mean([performance_summary[app]['F1-Score'] for app in APPLIANCES])
    avg_energy = np.mean([performance_summary[app]['Energy Accuracy (%)'] for app in APPLIANCES])
    
    print(f"{'AVERAGE':<20} {avg_mae:<12.2f} {avg_f1:<12.3f} {avg_energy:<15.1f}")
    print("=" * 70)


# =============================================================================
# Main Training and Evaluation Pipeline
# =============================================================================

def train_and_evaluate(data_path, output_dir='results'):
    """
    Main pipeline for training and evaluating the NILM model.
    
    Args:
        data_path: Path to the input Excel file
        output_dir: Directory to save results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    mains_power, appliance_power, df = load_and_prepare_data(data_path)
    
    # Prepare training data
    X_train, X_test, y_train, y_test, class_weights, feature_engineer, power_scalers = \
        prepare_training_data(mains_power, appliance_power)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build model
    print("\nBuilding Hybrid CNN-BiLSTM-Attention model...")
    model = build_nilm_model()
    
    # Compile model
    model = compile_model(model, class_weights)
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    performance_summary = evaluate_model(model, X_test, y_test, power_scalers)
    
    # Print performance table
    print_performance_table(performance_summary)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    for appliance in APPLIANCES:
        metrics = performance_summary[appliance]
        
        # Regression plot
        plot_regression_results(
            metrics['y_true_power'],
            metrics['y_pred_power'],
            appliance,
            save_path=os.path.join(output_dir, f'{appliance}_regression.png')
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            metrics['y_true_state'],
            metrics['y_pred_state'],
            appliance,
            save_path=os.path.join(output_dir, f'{appliance}_confusion_matrix.png')
        )
    
    # Summary plot
    create_summary_plot(
        performance_summary,
        save_path=os.path.join(output_dir, 'performance_summary.png')
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Save performance summary to CSV
    summary_df = pd.DataFrame({
        'Appliance': APPLIANCES,
        'MAE (W)': [performance_summary[app]['MAE (W)'] for app in APPLIANCES],
        'F1-Score': [performance_summary[app]['F1-Score'] for app in APPLIANCES],
        'Energy Accuracy (%)': [performance_summary[app]['Energy Accuracy (%)'] for app in APPLIANCES]
    })
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}/")
    
    return model, performance_summary, history


# =============================================================================
# Demo / Test Mode (when dataset is not available)
# =============================================================================

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic NILM data for testing."""
    
    print("Generating synthetic NILM data for demonstration...")
    
    # Generate time index
    t = np.arange(n_samples)
    
    # Generate synthetic appliance powers with realistic patterns
    appliance_power = {}
    
    # Air Conditioner: Cyclic pattern (ON/OFF thermostat)
    ac_cycle = np.sin(2 * np.pi * t / 500) > 0.3
    appliance_power['Air_Conditioner'] = ac_cycle.astype(float) * (1200 + np.random.randn(n_samples) * 50)
    appliance_power['Air_Conditioner'] = np.maximum(0, appliance_power['Air_Conditioner'])
    
    # Refrigerator: Regular compressor cycles
    ref_cycle = np.sin(2 * np.pi * t / 200) > 0.5
    appliance_power['Refrigerator'] = ref_cycle.astype(float) * (150 + np.random.randn(n_samples) * 10)
    appliance_power['Refrigerator'] = np.maximum(0, appliance_power['Refrigerator'])
    
    # Fan: Variable speed (multi-mode)
    fan_mode = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1])
    fan_power_levels = [0, 25, 40, 60]  # OFF, Low, Medium, High
    appliance_power['Fan'] = np.array([fan_power_levels[m] for m in fan_mode]) + np.random.randn(n_samples) * 3
    appliance_power['Fan'] = np.maximum(0, appliance_power['Fan'])
    
    # Washing Machine: Long cycles with different phases
    wm_cycle = np.zeros(n_samples)
    cycle_start = 0
    while cycle_start < n_samples - 1000:
        if np.random.random() > 0.8:
            # Start a washing cycle
            wm_cycle[cycle_start:cycle_start+300] = 100  # Wash
            wm_cycle[cycle_start+300:cycle_start+500] = 300  # Rinse
            wm_cycle[cycle_start+500:cycle_start+700] = 500  # Spin
        cycle_start += 1500
    appliance_power['Washing_Machine'] = wm_cycle + np.random.randn(n_samples) * 10
    appliance_power['Washing_Machine'] = np.maximum(0, appliance_power['Washing_Machine'])
    
    # EV Charger: Long charging sessions
    ev_cycle = np.zeros(n_samples)
    charge_start = 0
    while charge_start < n_samples - 2000:
        if np.random.random() > 0.9:
            duration = np.random.randint(1000, 2000)
            ev_cycle[charge_start:charge_start+duration] = 3000
        charge_start += 3000
    appliance_power['EV_Charger'] = ev_cycle + np.random.randn(n_samples) * 20
    appliance_power['EV_Charger'] = np.maximum(0, appliance_power['EV_Charger'])
    
    # Total mains power (sum of all appliances + baseline + noise)
    baseline = 200  # Base load
    mains_power = baseline + sum(appliance_power.values()) + np.random.randn(n_samples) * 50
    mains_power = np.maximum(0, mains_power)
    
    return mains_power, appliance_power


def run_demo():
    """Run a demonstration with synthetic data."""
    
    print("=" * 70)
    print("NILM S2P Model Demonstration")
    print("=" * 70)
    
    # Generate synthetic data
    mains_power, appliance_power = generate_synthetic_data(10000)
    
    # Prepare training data
    X_train, X_test, y_train, y_test, class_weights, feature_engineer, power_scalers = \
        prepare_training_data(mains_power, appliance_power)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build and compile model
    print("\nBuilding model...")
    model = build_nilm_model()
    model = compile_model(model, class_weights)
    
    # Quick training for demo
    print("\nTraining model (demo mode - 10 epochs)...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=10,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    performance_summary = evaluate_model(model, X_test, y_test, power_scalers)
    print_performance_table(performance_summary)
    
    # Save demo results
    demo_dir = 'demo_results'
    os.makedirs(demo_dir, exist_ok=True)
    
    for appliance in APPLIANCES:
        metrics = performance_summary[appliance]
        plot_regression_results(
            metrics['y_true_power'],
            metrics['y_pred_power'],
            appliance,
            save_path=os.path.join(demo_dir, f'{appliance}_regression.png')
        )
        plot_confusion_matrix(
            metrics['y_true_state'],
            metrics['y_pred_state'],
            appliance,
            save_path=os.path.join(demo_dir, f'{appliance}_confusion_matrix.png')
        )
    
    create_summary_plot(
        performance_summary,
        save_path=os.path.join(demo_dir, 'performance_summary.png')
    )
    
    print(f"\nDemo results saved to {demo_dir}/")
    
    return model, performance_summary


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NILM S2P Model Training and Evaluation')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the input Excel file (combined_appliances_correct_order.xlsx)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        # Run demo mode
        model, performance_summary = run_demo()
    else:
        # Train with real data
        model, performance_summary, history = train_and_evaluate(args.data, args.output)
