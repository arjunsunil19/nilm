"""
Data Preprocessor for NILM

Utilities for preprocessing energy consumption data for NILM models.
Includes normalization, windowing, and data augmentation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Data preprocessor for NILM.
    
    Handles:
    - Normalization (StandardScaler, MinMaxScaler)
    - Windowing (creating input windows)
    - Train/test splitting
    - Data augmentation
    
    Attributes:
        window_size: Size of input windows
        stride: Stride between consecutive windows
        normalize_method: Normalization method ('standard', 'minmax', 'appliance')
    """
    
    # Appliance-specific normalization parameters
    APPLIANCE_PARAMS = {
        "kettle": {"max": 3100, "mean": 700, "std": 1000, "on_threshold": 2000},
        "microwave": {"max": 3000, "mean": 500, "std": 800, "on_threshold": 200},
        "fridge": {"max": 400, "mean": 200, "std": 400, "on_threshold": 50},
        "dishwasher": {"max": 2500, "mean": 700, "std": 1000, "on_threshold": 10},
        "washing_machine": {"max": 2500, "mean": 400, "std": 700, "on_threshold": 20},
    }
    
    # Aggregate mains normalization parameters
    MAINS_PARAMS = {"mean": 522, "std": 814, "max": 10000}
    
    def __init__(
        self,
        window_size: int = 599,
        stride: int = 1,
        normalize_method: str = "appliance"
    ):
        """
        Initialize the DataPreprocessor.
        
        Args:
            window_size: Size of input windows
            stride: Stride between consecutive windows
            normalize_method: 'standard', 'minmax', or 'appliance'
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize_method = normalize_method
        
        self.scaler_X = None
        self.scaler_y = None
    
    def create_windows(
        self,
        aggregate: np.ndarray,
        appliance: np.ndarray,
        target_type: str = "midpoint"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data.
        
        Args:
            aggregate: Aggregate power time series
            appliance: Appliance power time series
            target_type: 'midpoint' (Seq2Point) or 'sequence' (Seq2Seq)
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Ensure 1D arrays
        aggregate = np.array(aggregate).flatten()
        appliance = np.array(appliance).flatten()
        
        num_windows = (len(aggregate) - self.window_size) // self.stride + 1
        
        X = np.zeros((num_windows, self.window_size, 1), dtype=np.float32)
        
        if target_type == "midpoint":
            y = np.zeros((num_windows, 1), dtype=np.float32)
        else:  # sequence
            y = np.zeros((num_windows, self.window_size, 1), dtype=np.float32)
        
        midpoint = self.window_size // 2
        
        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_size
            
            X[i, :, 0] = aggregate[start:end]
            
            if target_type == "midpoint":
                y[i, 0] = appliance[start + midpoint]
            else:
                y[i, :, 0] = appliance[start:end]
        
        return X, y
    
    def normalize(
        self,
        data: np.ndarray,
        appliance_name: Optional[str] = None,
        is_aggregate: bool = True,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data: Data to normalize
            appliance_name: Name of appliance (for appliance-specific normalization)
            is_aggregate: Whether data is aggregate (True) or appliance (False)
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized data
        """
        if self.normalize_method == "appliance" and appliance_name:
            # Use appliance-specific parameters
            if is_aggregate:
                params = self.MAINS_PARAMS
            else:
                params = self.APPLIANCE_PARAMS.get(
                    appliance_name,
                    {"mean": 0, "std": 1}
                )
            
            normalized = (data - params["mean"]) / params["std"]
            return normalized
        
        elif self.normalize_method == "standard":
            if is_aggregate:
                if fit or self.scaler_X is None:
                    self.scaler_X = StandardScaler()
                    return self.scaler_X.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
                else:
                    return self.scaler_X.transform(data.reshape(-1, 1)).reshape(data.shape)
            else:
                if fit or self.scaler_y is None:
                    self.scaler_y = StandardScaler()
                    return self.scaler_y.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
                else:
                    return self.scaler_y.transform(data.reshape(-1, 1)).reshape(data.shape)
        
        elif self.normalize_method == "minmax":
            if is_aggregate:
                if fit or self.scaler_X is None:
                    self.scaler_X = MinMaxScaler()
                    return self.scaler_X.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
                else:
                    return self.scaler_X.transform(data.reshape(-1, 1)).reshape(data.shape)
            else:
                if fit or self.scaler_y is None:
                    self.scaler_y = MinMaxScaler()
                    return self.scaler_y.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
                else:
                    return self.scaler_y.transform(data.reshape(-1, 1)).reshape(data.shape)
        
        return data
    
    def denormalize(
        self,
        data: np.ndarray,
        appliance_name: Optional[str] = None,
        is_aggregate: bool = False
    ) -> np.ndarray:
        """
        Denormalize data back to original scale.
        
        Args:
            data: Normalized data
            appliance_name: Name of appliance
            is_aggregate: Whether data is aggregate or appliance
            
        Returns:
            Denormalized data
        """
        if self.normalize_method == "appliance" and appliance_name:
            if is_aggregate:
                params = self.MAINS_PARAMS
            else:
                params = self.APPLIANCE_PARAMS.get(
                    appliance_name,
                    {"mean": 0, "std": 1}
                )
            
            return data * params["std"] + params["mean"]
        
        elif self.normalize_method in ["standard", "minmax"]:
            scaler = self.scaler_X if is_aggregate else self.scaler_y
            if scaler is not None:
                return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)
        
        return data
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Input features
            y: Target values
            train_ratio: Ratio of training data
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        n_samples = len(X)
        
        if shuffle:
            np.random.seed(random_state)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        train_size = int(n_samples * train_ratio)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_factor: float = 0.05,
        shift_range: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with noise and shifts.
        
        Args:
            X: Input features
            y: Target values
            noise_factor: Standard deviation of Gaussian noise
            shift_range: Maximum shift in samples
            
        Returns:
            Augmented (X, y) arrays
        """
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor * np.std(X), X.shape)
        X_augmented = X + noise
        
        # Random time shifts
        if shift_range > 0:
            shift = np.random.randint(-shift_range, shift_range + 1)
            if shift > 0:
                X_augmented = np.roll(X_augmented, shift, axis=1)
                X_augmented[:, :shift, :] = X_augmented[:, shift:shift+1, :]
            elif shift < 0:
                X_augmented = np.roll(X_augmented, shift, axis=1)
                X_augmented[:, shift:, :] = X_augmented[:, shift-1:shift, :]
        
        # Ensure non-negative values
        X_augmented = np.clip(X_augmented, 0, None)
        
        return X_augmented, y
    
    def prepare_data(
        self,
        aggregate: np.ndarray,
        appliance: np.ndarray,
        appliance_name: str,
        train_ratio: float = 0.8,
        target_type: str = "midpoint"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Combines normalization, windowing, and train/test splitting.
        
        Args:
            aggregate: Aggregate power time series
            appliance: Appliance power time series
            appliance_name: Name of the appliance
            train_ratio: Ratio of training data
            target_type: 'midpoint' or 'sequence'
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Normalize
        aggregate_norm = self.normalize(aggregate, appliance_name, is_aggregate=True)
        appliance_norm = self.normalize(appliance, appliance_name, is_aggregate=False)
        
        # Create windows
        X, y = self.create_windows(aggregate_norm, appliance_norm, target_type)
        
        # Split
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, train_ratio)
        
        return X_train, X_test, y_train, y_test
