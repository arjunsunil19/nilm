"""Feature engineering utilities for NILM system."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_8_channel_features(
    mains_power: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Compute 8-channel engineered features from mains power.
    
    Channels:
    1. Standardized mains power (RobustScaler)
    2. First derivative (power transitions)
    3. Second derivative (switching acceleration)
    4. Rolling mean (smoothed signal)
    5. Rolling standard deviation (local variance)
    6. Rolling range (local dynamics)
    7. Rolling skewness (asymmetry)
    8. Rolling kurtosis (tail behavior)
    
    Args:
        mains_power: Raw mains power values (1D array)
        window_size: Window size for rolling statistics
        
    Returns:
        Array of shape (n_samples, 8) containing all features
    """
    n_samples = len(mains_power)
    features = np.zeros((n_samples, 8), dtype=np.float32)
    
    # Convert to pandas Series for easier rolling operations
    power_series = pd.Series(mains_power)
    
    # Channel 1: Standardized mains power
    scaler = RobustScaler()
    features[:, 0] = scaler.fit_transform(mains_power.reshape(-1, 1)).flatten()
    
    # Channel 2: First derivative
    first_deriv = np.gradient(mains_power)
    features[:, 1] = first_deriv
    
    # Channel 3: Second derivative
    second_deriv = np.gradient(first_deriv)
    features[:, 2] = second_deriv
    
    # Channel 4: Rolling mean
    rolling_mean = power_series.rolling(window=window_size, center=True, min_periods=1).mean()
    features[:, 3] = rolling_mean.values
    
    # Channel 5: Rolling standard deviation
    rolling_std = power_series.rolling(window=window_size, center=True, min_periods=1).std()
    features[:, 4] = rolling_std.fillna(0).values
    
    # Channel 6: Rolling range (max - min)
    rolling_max = power_series.rolling(window=window_size, center=True, min_periods=1).max()
    rolling_min = power_series.rolling(window=window_size, center=True, min_periods=1).min()
    features[:, 5] = (rolling_max - rolling_min).values
    
    # Channel 7: Rolling skewness
    rolling_skew = power_series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: skew(x) if len(x) > 2 else 0, raw=True
    )
    features[:, 6] = rolling_skew.fillna(0).values
    
    # Channel 8: Rolling kurtosis
    rolling_kurt = power_series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: kurtosis(x) if len(x) > 3 else 0, raw=True
    )
    features[:, 7] = rolling_kurt.fillna(0).values
    
    # Handle any remaining NaN or inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Computed 8-channel features. Shape: {features.shape}")
    return features


def create_target_labels(
    appliance_power: np.ndarray,
    thresholds: list,
    appliance_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create target labels for power regression and state classification.
    
    Args:
        appliance_power: Raw appliance power values
        thresholds: Power thresholds for state classification
        appliance_type: 'binary' or 'multi_state'
        
    Returns:
        Tuple of (power_targets, state_labels)
    """
    power_targets = appliance_power.astype(np.float32)
    
    if appliance_type == 'binary':
        # Binary classification: OFF (0) or ON (1)
        state_labels = (appliance_power > thresholds[0]).astype(np.int32)
    else:
        # Multi-state classification
        state_labels = np.zeros_like(appliance_power, dtype=np.int32)
        for i, threshold in enumerate(thresholds):
            state_labels[appliance_power > threshold] = i + 1
    
    return power_targets, state_labels


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Normalize features using RobustScaler (already done in feature engineering).
    
    This function is kept for consistency, but features are already scaled.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Tuple of normalized features and the scaler
    """
    # Features are already scaled in compute_8_channel_features
    # This function returns them as-is for consistency
    scaler = RobustScaler()
    
    logger.info("Features are already normalized in feature engineering step")
    return X_train, X_val, X_test, scaler
