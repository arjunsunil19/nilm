#!/usr/bin/env python3
"""
Complete NILM (Non-Intrusive Load Monitoring) System
Supports multiple deep learning architectures for appliance disaggregation
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


# ============================================================================
# Configuration
# ============================================================================

APPLIANCES = {
    'air_conditioner': {'states': 2, 'type': 'binary', 'power_range': (0, 3500)},
    'refrigerator': {'states': 2, 'type': 'binary', 'power_range': (0, 300)},
    'fan': {'states': 4, 'type': 'multi', 'power_range': (0, 100)},
    'washing_machine': {'states': 4, 'type': 'multi', 'power_range': (0, 2500)},
    'ev_charger': {'states': 2, 'type': 'binary', 'power_range': (0, 7000)}
}

WINDOW_SIZE = 128
STRIDE = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# ============================================================================
# Data Pipeline
# ============================================================================

class DataPipeline:
    """Handles data loading, preprocessing, and windowing"""
    
    def __init__(self, window_size: int = WINDOW_SIZE, stride: int = STRIDE):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.appliance_scalers = {}
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic NILM data for demonstration"""
        print("Generating synthetic data...")
        
        # Generate time series
        time = np.linspace(0, n_samples / 60, n_samples)  # 1 sample per minute
        
        # Generate appliance signals
        appliances_data = {}
        
        # Air Conditioner - periodic with long on/off cycles
        ac_signal = np.zeros(n_samples)
        ac_on_periods = [(i, min(i + 240, n_samples)) for i in range(0, n_samples, 600)]
        for start, end in ac_on_periods:
            ac_signal[start:end] = 3000 + np.random.normal(0, 100, end - start)
        appliances_data['air_conditioner'] = ac_signal
        
        # Refrigerator - cyclic behavior
        ref_signal = np.zeros(n_samples)
        for i in range(0, n_samples, 60):
            if i % 180 < 45:  # On for 45 min every 3 hours
                end = min(i + 45, n_samples)
                ref_signal[i:end] = 200 + np.random.normal(0, 20, end - i)
        appliances_data['refrigerator'] = ref_signal
        
        # Fan - 4 states (OFF, Low, Medium, High)
        fan_signal = np.zeros(n_samples)
        fan_states = [0, 25, 50, 75]
        for i in range(0, n_samples, 120):
            state = fan_states[np.random.randint(0, 4)]
            end = min(i + 120, n_samples)
            fan_signal[i:end] = state + np.random.normal(0, 2, end - i)
        appliances_data['fan'] = fan_signal
        
        # Washing Machine - 4 states (OFF, Wash, Rinse, Spin)
        wm_signal = np.zeros(n_samples)
        wm_states = [0, 500, 300, 2000]
        for i in range(0, n_samples, 480):
            if np.random.random() > 0.5:  # 50% chance of washing cycle
                # Complete cycle: Wash -> Rinse -> Spin
                for j, state in enumerate(wm_states[1:]):
                    start = i + j * 30
                    end = min(start + 30, n_samples)
                    if end > start:
                        wm_signal[start:end] = state + np.random.normal(0, 50, end - start)
        appliances_data['washing_machine'] = wm_signal
        
        # EV Charger - random long charging sessions
        ev_signal = np.zeros(n_samples)
        ev_on_periods = [(i, min(i + 360, n_samples)) for i in range(0, n_samples, 1200) if np.random.random() > 0.6]
        for start, end in ev_on_periods:
            ev_signal[start:end] = 6500 + np.random.normal(0, 200, end - start)
        appliances_data['ev_charger'] = ev_signal
        
        # Aggregate signal
        aggregate = np.sum([appliances_data[app] for app in APPLIANCES.keys()], axis=0)
        aggregate += np.random.normal(0, 50, n_samples)  # Add noise
        
        # Ensure non-negative
        aggregate = np.maximum(aggregate, 0)
        for app in appliances_data:
            appliances_data[app] = np.maximum(appliances_data[app], 0)
        
        return aggregate, appliances_data
    
    def create_windows(self, aggregate: np.ndarray, appliances: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create sliding windows from time series data"""
        print(f"Creating windows (size={self.window_size}, stride={self.stride})...")
        
        n_samples = len(aggregate)
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        X = np.zeros((n_windows, self.window_size, 1))
        y = {app: np.zeros((n_windows, self.window_size, 1)) for app in APPLIANCES.keys()}
        
        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            X[i, :, 0] = aggregate[start:end]
            for app in APPLIANCES.keys():
                y[app][i, :, 0] = appliances[app][start:end]
        
        return X, y
    
    def prepare_data(self, n_samples: int = 10000, train_split: float = 0.7, val_split: float = 0.15) -> Dict[str, Any]:
        """Complete data preparation pipeline"""
        # Generate data
        aggregate, appliances = self.generate_synthetic_data(n_samples)
        
        # Create windows
        X, y = self.create_windows(aggregate, appliances)
        
        # Split data
        n_train = int(len(X) * train_split)
        n_val = int(len(X) * val_split)
        
        X_train = X[:n_train]
        X_val = X[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        
        # Normalize
        X_train_reshaped = X_train.reshape(-1, 1)
        self.scaler.fit(X_train_reshaped)
        
        X_train = self.scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        
        # Prepare targets for each appliance
        data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': {}, 'y_val': {}, 'y_test': {}
        }
        
        for app in APPLIANCES.keys():
            # Normalize appliance data
            self.appliance_scalers[app] = StandardScaler()
            y_train = y[app][:n_train]
            y_val = y[app][n_train:n_train + n_val]
            y_test = y[app][n_train + n_val:]
            
            y_train_reshaped = y_train.reshape(-1, 1)
            self.appliance_scalers[app].fit(y_train_reshaped)
            
            data['y_train'][app] = self.appliance_scalers[app].transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
            data['y_val'][app] = self.appliance_scalers[app].transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
            data['y_test'][app] = self.appliance_scalers[app].transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        print(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return data


# ============================================================================
# Model Architectures
# ============================================================================

class ModelFactory:
    """Factory for creating different NILM model architectures"""
    
    @staticmethod
    def create_cnn_bilstm_attention(input_shape: Tuple[int, int], output_shape: Tuple[int, int], name: str = "cnn_bilstm_attention") -> Model:
        """CNN-BiLSTM with Attention mechanism"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # CNN layers for feature extraction
        x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(256)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        attended = layers.Multiply()([x, attention])
        
        # Upsampling to original sequence length
        x = layers.UpSampling1D(2)(attended)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.UpSampling1D(2)(x)
        
        # Output layer
        outputs = layers.Conv1D(1, 1, padding='same', activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    @staticmethod
    def create_transformer(input_shape: Tuple[int, int], output_shape: Tuple[int, int], name: str = "transformer") -> Model:
        """Transformer-based architecture"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=64)(positions)
        
        x = layers.Dense(64)(inputs)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(2):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
            x1 = layers.Add()([x, attn_output])
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            
            # Feed forward
            ffn = Sequential([
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(64)
            ])
            ffn_output = ffn(x1)
            x = layers.Add()([x1, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layer
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    @staticmethod
    def create_unet(input_shape: Tuple[int, int], output_shape: Tuple[int, int], name: str = "unet") -> Model:
        """U-Net architecture for sequence-to-sequence"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Encoder
        conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling1D(2)(conv1)
        
        conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling1D(2)(conv2)
        
        conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling1D(2)(conv3)
        
        # Bottleneck
        conv4 = layers.Conv1D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv1D(512, 3, activation='relu', padding='same')(conv4)
        
        # Decoder
        up5 = layers.UpSampling1D(2)(conv4)
        up5 = layers.Concatenate()([up5, conv3])
        conv5 = layers.Conv1D(256, 3, activation='relu', padding='same')(up5)
        conv5 = layers.Conv1D(256, 3, activation='relu', padding='same')(conv5)
        
        up6 = layers.UpSampling1D(2)(conv5)
        up6 = layers.Concatenate()([up6, conv2])
        conv6 = layers.Conv1D(128, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv1D(128, 3, activation='relu', padding='same')(conv6)
        
        up7 = layers.UpSampling1D(2)(conv6)
        up7 = layers.Concatenate()([up7, conv1])
        conv7 = layers.Conv1D(64, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output
        outputs = layers.Conv1D(1, 1, activation='linear', name='output')(conv7)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    @staticmethod
    def create_resnet_lstm(input_shape: Tuple[int, int], output_shape: Tuple[int, int], name: str = "resnet_lstm") -> Model:
        """ResNet-LSTM hybrid architecture"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Initial conv
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        for filters in [64, 128, 256]:
            # Residual block
            shortcut = x
            
            x = layers.Conv1D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv1D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Match dimensions for shortcut
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(2)(x)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        
        # Upsampling
        for _ in range(3):
            x = layers.UpSampling1D(2)(x)
            x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        
        # Output
        outputs = layers.Conv1D(1, 1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    @staticmethod
    def create_model(architecture: str, input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> Model:
        """Create model based on architecture name"""
        architectures = {
            'cnn_bilstm_attention': ModelFactory.create_cnn_bilstm_attention,
            'transformer': ModelFactory.create_transformer,
            'unet': ModelFactory.create_unet,
            'resnet_lstm': ModelFactory.create_resnet_lstm
        }
        
        if architecture not in architectures:
            raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(architectures.keys())}")
        
        return architectures[architecture](input_shape, output_shape, name=architecture)


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Handles model training with callbacks and monitoring"""
    
    def __init__(self, model: Model, appliance: str, model_dir: str = 'models'):
        self.model = model
        self.appliance = appliance
        self.model_dir = model_dir
        self.history = None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def compile_model(self, learning_rate: float = LEARNING_RATE):
        """Compile model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
        """Train the model"""
        print(f"\nTraining {self.model.name} for {self.appliance}...")
        
        # Callbacks
        model_path = os.path.join(self.model_dir, f"{self.model.name}_{self.appliance}.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history


# ============================================================================
# Prediction
# ============================================================================

class Predictor:
    """Handles model predictions"""
    
    def __init__(self, model: Model, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Make predictions"""
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        # Denormalize
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        return predictions


# ============================================================================
# Post-Processing
# ============================================================================

class PostProcessor:
    """Post-processing for predictions"""
    
    @staticmethod
    def apply_threshold(predictions: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """Apply threshold to predictions"""
        return np.where(predictions < threshold, 0, predictions)
    
    @staticmethod
    def classify_states(predictions: np.ndarray, appliance: str) -> np.ndarray:
        """Classify predictions into discrete states"""
        config = APPLIANCES[appliance]
        
        if config['type'] == 'binary':
            # Binary classification
            threshold = (config['power_range'][1] - config['power_range'][0]) * 0.1
            return (predictions > threshold).astype(int)
        else:
            # Multi-class classification
            if appliance == 'fan':
                # OFF, Low, Medium, High
                states = np.zeros_like(predictions, dtype=int)
                states[predictions > 10] = 1  # Low
                states[predictions > 35] = 2  # Medium
                states[predictions > 60] = 3  # High
                return states
            elif appliance == 'washing_machine':
                # OFF, Wash, Rinse, Spin
                states = np.zeros_like(predictions, dtype=int)
                states[predictions > 100] = 1  # Wash
                states[(predictions > 200) & (predictions < 800)] = 2  # Rinse
                states[predictions > 1500] = 3  # Spin
                return states
        
        return predictions


# ============================================================================
# Evaluation
# ============================================================================

class Evaluator:
    """Comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, appliance: str) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Regression metrics
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
        
        # Energy metrics
        true_energy = np.sum(y_true_flat)
        pred_energy = np.sum(y_pred_flat)
        energy_error = np.abs(true_energy - pred_energy) / (true_energy + 1e-10)
        energy_accuracy = 1.0 - energy_error
        
        # Classification metrics
        y_true_states = PostProcessor.classify_states(y_true_flat, appliance)
        y_pred_states = PostProcessor.classify_states(y_pred_flat, appliance)
        
        accuracy = accuracy_score(y_true_states, y_pred_states)
        precision = precision_score(y_true_states, y_pred_states, average='weighted', zero_division=0)
        recall = recall_score(y_true_states, y_pred_states, average='weighted', zero_division=0)
        f1 = f1_score(y_true_states, y_pred_states, average='weighted', zero_division=0)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'Energy_Accuracy': energy_accuracy * 100,
            'State_Accuracy': accuracy * 100,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str, appliance: str):
        """Print metrics in a formatted way"""
        print(f"\n{'='*60}")
        print(f"Results for {model_name} - {appliance}")
        print(f"{'='*60}")
        print(f"MAE:              {metrics['MAE']:.2f} W")
        print(f"RMSE:             {metrics['RMSE']:.2f} W")
        print(f"Energy Accuracy:  {metrics['Energy_Accuracy']:.2f}%")
        print(f"State Accuracy:   {metrics['State_Accuracy']:.2f}%")
        print(f"Precision:        {metrics['Precision']:.4f}")
        print(f"Recall:           {metrics['Recall']:.4f}")
        print(f"F1-Score:         {metrics['F1_Score']:.4f}")
        print(f"{'='*60}\n")


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Publication-quality visualizations"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], model_name: str, appliance: str):
        """Plot training and validation loss"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title(f'{model_name} - {appliance}\nTraining History', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(history['mae'], label='Train MAE', linewidth=2)
        ax2.plot(history['val_mae'], label='Val MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE (W)', fontsize=12)
        ax2.set_title(f'{model_name} - {appliance}\nMAE History', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'{model_name}_{appliance}_training_history.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training history plot: {filename}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, appliance: str, n_samples: int = 500):
        """Plot predictions vs ground truth"""
        fig, ax = plt.subplots(figsize=(15, 5))
        
        y_true_flat = y_true.flatten()[:n_samples]
        y_pred_flat = y_pred.flatten()[:n_samples]
        
        time_steps = np.arange(len(y_true_flat))
        
        ax.plot(time_steps, y_true_flat, label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(time_steps, y_pred_flat, label='Prediction', linewidth=2, alpha=0.7)
        ax.fill_between(time_steps, y_true_flat, y_pred_flat, alpha=0.2)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Power (W)', fontsize=12)
        ax.set_title(f'{model_name} - {appliance}\nPredictions vs Ground Truth', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'{model_name}_{appliance}_predictions.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved predictions plot: {filename}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, appliance: str):
        """Plot confusion matrix for state classification"""
        y_true_states = PostProcessor.classify_states(y_true.flatten(), appliance)
        y_pred_states = PostProcessor.classify_states(y_pred.flatten(), appliance)
        
        cm = confusion_matrix(y_true_states, y_pred_states)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted State', fontsize=12)
        ax.set_ylabel('True State', fontsize=12)
        ax.set_title(f'{model_name} - {appliance}\nConfusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'{model_name}_{appliance}_confusion_matrix.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix: {filename}")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                             metric: str = 'F1_Score'):
        """Plot comparison of all models across appliances"""
        models = list(results.keys())
        appliances = list(APPLIANCES.keys())
        
        data = []
        for model in models:
            for appliance in appliances:
                if appliance in results[model]:
                    data.append({
                        'Model': model,
                        'Appliance': appliance,
                        'Score': results[model][appliance][metric]
                    })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create grouped bar chart
        x = np.arange(len(appliances))
        width = 0.2
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = [model_data[model_data['Appliance'] == app]['Score'].values[0] 
                     if len(model_data[model_data['Appliance'] == app]) > 0 else 0 
                     for app in appliances]
            ax.bar(x + i * width, scores, width, label=model)
        
        ax.set_xlabel('Appliance', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([app.replace('_', ' ').title() for app in appliances], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'model_comparison_{metric}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved model comparison plot: {filename}")
    
    def plot_overall_summary(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Plot overall summary of all metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['MAE', 'F1_Score', 'Energy_Accuracy', 'State_Accuracy']
        titles = ['Mean Absolute Error (W)', 'F1-Score', 'Energy Accuracy (%)', 'State Accuracy (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            data = []
            for model in results.keys():
                for appliance in results[model].keys():
                    data.append({
                        'Model': model,
                        'Appliance': appliance.replace('_', ' ').title(),
                        'Value': results[model][appliance][metric]
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            models = df['Model'].unique()
            appliances = df['Appliance'].unique()
            x = np.arange(len(appliances))
            width = 0.2
            
            for i, model in enumerate(models):
                model_data = df[df['Model'] == model]
                values = [model_data[model_data['Appliance'] == app]['Value'].values[0]
                         if len(model_data[model_data['Appliance'] == app]) > 0 else 0
                         for app in appliances]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel('Appliance', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(appliances, rotation=45, ha='right', fontsize=8)
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'overall_summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved overall summary plot: {filename}")


# ============================================================================
# NILM Pipeline
# ============================================================================

class NILMPipeline:
    """Complete NILM pipeline orchestrator"""
    
    def __init__(self, architecture: str, appliance: str, data: Dict[str, Any], 
                 data_pipeline: DataPipeline, model_dir: str = 'models', 
                 results_dir: str = 'results'):
        self.architecture = architecture
        self.appliance = appliance
        self.data = data
        self.data_pipeline = data_pipeline
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        self.model = None
        self.trainer = None
        self.predictor = None
        self.visualizer = Visualizer(results_dir)
        
    def build_model(self):
        """Build the model architecture"""
        input_shape = (WINDOW_SIZE, 1)
        output_shape = (WINDOW_SIZE, 1)
        
        self.model = ModelFactory.create_model(self.architecture, input_shape, output_shape)
        print(f"\nModel architecture: {self.architecture}")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def train_model(self, epochs: int = EPOCHS):
        """Train the model"""
        self.trainer = Trainer(self.model, self.appliance, self.model_dir)
        self.trainer.compile_model()
        
        history = self.trainer.train(
            self.data['X_train'],
            self.data['y_train'][self.appliance],
            self.data['X_val'],
            self.data['y_val'][self.appliance],
            epochs=epochs
        )
        
        # Visualize training history
        self.visualizer.plot_training_history(history, self.architecture, self.appliance)
        
        return history
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model"""
        # Create predictor
        appliance_scaler = self.data_pipeline.appliance_scalers[self.appliance]
        self.predictor = Predictor(self.model, appliance_scaler)
        
        # Make predictions
        y_pred = self.predictor.predict(self.data['X_test'])
        
        # Denormalize ground truth
        y_true = appliance_scaler.inverse_transform(
            self.data['y_test'][self.appliance].reshape(-1, 1)
        ).reshape(self.data['y_test'][self.appliance].shape)
        
        # Apply post-processing
        y_pred = PostProcessor.apply_threshold(y_pred)
        
        # Calculate metrics
        metrics = Evaluator.calculate_metrics(y_true, y_pred, self.appliance)
        Evaluator.print_metrics(metrics, self.architecture, self.appliance)
        
        # Visualize results
        self.visualizer.plot_predictions(y_true, y_pred, self.architecture, self.appliance)
        self.visualizer.plot_confusion_matrix(y_true, y_pred, self.architecture, self.appliance)
        
        return metrics
    
    def run(self, epochs: int = EPOCHS) -> Dict[str, float]:
        """Run complete pipeline"""
        print(f"\n{'#'*60}")
        print(f"Running NILM Pipeline")
        print(f"Architecture: {self.architecture}")
        print(f"Appliance: {self.appliance}")
        print(f"{'#'*60}\n")
        
        self.build_model()
        self.train_model(epochs)
        metrics = self.evaluate_model()
        
        return metrics


# ============================================================================
# Multi-Model Comparison
# ============================================================================

def run_all_models(data: Dict[str, Any], data_pipeline: DataPipeline, 
                   appliances: List[str] = None, 
                   architectures: List[str] = None,
                   epochs: int = EPOCHS) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run all model architectures for all appliances and compare results"""
    
    if appliances is None:
        appliances = list(APPLIANCES.keys())
    
    if architectures is None:
        architectures = ['cnn_bilstm_attention', 'transformer', 'unet', 'resnet_lstm']
    
    results = {}
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    print(f"Architectures: {', '.join(architectures)}")
    print(f"Appliances: {', '.join(appliances)}")
    print("="*60 + "\n")
    
    for architecture in architectures:
        results[architecture] = {}
        
        for appliance in appliances:
            print(f"\n{'*'*60}")
            print(f"Training {architecture} for {appliance}")
            print(f"{'*'*60}")
            
            try:
                pipeline = NILMPipeline(
                    architecture=architecture,
                    appliance=appliance,
                    data=data,
                    data_pipeline=data_pipeline
                )
                
                metrics = pipeline.run(epochs=epochs)
                results[architecture][appliance] = metrics
                
            except Exception as e:
                print(f"Error training {architecture} for {appliance}: {str(e)}")
                continue
    
    # Save results
    results_file = os.path.join('results', 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    visualizer = Visualizer()
    visualizer.plot_model_comparison(results, 'F1_Score')
    visualizer.plot_model_comparison(results, 'MAE')
    visualizer.plot_overall_summary(results)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for architecture in results.keys():
        print(f"\n{architecture.upper()}")
        print("-" * 60)
        
        avg_mae = np.mean([results[architecture][app]['MAE'] for app in results[architecture].keys()])
        avg_f1 = np.mean([results[architecture][app]['F1_Score'] for app in results[architecture].keys()])
        avg_energy = np.mean([results[architecture][app]['Energy_Accuracy'] for app in results[architecture].keys()])
        
        print(f"Average MAE:             {avg_mae:.2f} W")
        print(f"Average F1-Score:        {avg_f1:.4f}")
        print(f"Average Energy Accuracy: {avg_energy:.2f}%")
    
    print("\n" + "="*60)
    
    return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Complete NILM System - Non-Intrusive Load Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models for all appliances
  python nilm_system.py --all
  
  # Run specific model for specific appliance
  python nilm_system.py --model cnn_bilstm_attention --appliance air_conditioner
  
  # Run specific model for all appliances
  python nilm_system.py --model transformer
  
  # Quick test with fewer epochs
  python nilm_system.py --model unet --appliance fan --epochs 20
  
Available Models:
  - cnn_bilstm_attention: CNN-BiLSTM with Attention mechanism
  - transformer: Transformer-based architecture
  - unet: U-Net for sequence-to-sequence
  - resnet_lstm: ResNet-LSTM hybrid
  
Available Appliances:
  - air_conditioner: Binary classification (OFF/ON)
  - refrigerator: Binary classification (OFF/ON)
  - fan: 4-state classification (OFF/Low/Medium/High)
  - washing_machine: 4-state classification (OFF/Wash/Rinse/Spin)
  - ev_charger: Binary classification (OFF/ON)
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all models for all appliances (comprehensive comparison)')
    parser.add_argument('--model', type=str, choices=['cnn_bilstm_attention', 'transformer', 'unet', 'resnet_lstm'],
                       help='Model architecture to use')
    parser.add_argument('--appliance', type=str, choices=list(APPLIANCES.keys()),
                       help='Target appliance')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for training (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and args.model is None:
        parser.error("Either --all or --model must be specified")
    
    if args.model and not args.all and args.appliance is None:
        print("Warning: No appliance specified. Will run for all appliances.")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Setup
    print("\n" + "="*60)
    print("NILM SYSTEM - Non-Intrusive Load Monitoring")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("="*60 + "\n")
    
    # Prepare data
    print("Preparing data pipeline...")
    data_pipeline = DataPipeline()
    data = data_pipeline.prepare_data(n_samples=args.samples)
    
    # Run models
    if args.all:
        # Run all models for all appliances
        results = run_all_models(data, data_pipeline, epochs=args.epochs)
    elif args.appliance:
        # Run specific model for specific appliance
        pipeline = NILMPipeline(
            architecture=args.model,
            appliance=args.appliance,
            data=data,
            data_pipeline=data_pipeline
        )
        metrics = pipeline.run(epochs=args.epochs)
    else:
        # Run specific model for all appliances
        results = {}
        results[args.model] = {}
        
        for appliance in APPLIANCES.keys():
            pipeline = NILMPipeline(
                architecture=args.model,
                appliance=appliance,
                data=data,
                data_pipeline=data_pipeline
            )
            metrics = pipeline.run(epochs=args.epochs)
            results[args.model][appliance] = metrics
        
        # Save and visualize results
        results_file = os.path.join('results', f'{args.model}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_file}")
        
        # Summary
        avg_mae = np.mean([results[args.model][app]['MAE'] for app in results[args.model].keys()])
        avg_f1 = np.mean([results[args.model][app]['F1_Score'] for app in results[args.model].keys()])
        avg_energy = np.mean([results[args.model][app]['Energy_Accuracy'] for app in results[args.model].keys()])
        
        print(f"\n{'='*60}")
        print(f"AVERAGE RESULTS FOR {args.model.upper()}")
        print(f"{'='*60}")
        print(f"Average MAE:             {avg_mae:.2f} W")
        print(f"Average F1-Score:        {avg_f1:.4f}")
        print(f"Average Energy Accuracy: {avg_energy:.2f}%")
        print(f"{'='*60}\n")
    
    print("\nNILM System execution completed successfully!")
    print("Check the 'results' directory for visualizations and detailed metrics.")


if __name__ == '__main__':
    main()
