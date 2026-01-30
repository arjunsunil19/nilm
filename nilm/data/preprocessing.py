"""
Data Preprocessing for NILM

Utilities for preprocessing power consumption data for NILM models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class DataPreprocessor:
    """
    Preprocessor for NILM data.
    
    Handles normalization, windowing, and data cleaning
    for NILM datasets.
    """
    
    def __init__(
        self,
        window_size: int = 599,
        stride: int = 1,
        normalize: bool = True,
        clip_outliers: bool = True,
        outlier_percentile: float = 99.5,
        fill_nan_method: str = "interpolate",
    ):
        """
        Initialize preprocessor.
        
        Args:
            window_size: Size of sliding window
            stride: Stride for sliding window
            normalize: Whether to normalize data
            clip_outliers: Whether to clip outliers
            outlier_percentile: Percentile for outlier clipping
            fill_nan_method: Method for filling NaN values
                           ("interpolate", "zero", "mean", "forward")
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.clip_outliers = clip_outliers
        self.outlier_percentile = outlier_percentile
        self.fill_nan_method = fill_nan_method
        
        # Statistics (computed during fit)
        self.stats: Dict = {}
        
    def fit(
        self,
        aggregate: np.ndarray,
        appliances: Optional[np.ndarray] = None,
    ) -> "DataPreprocessor":
        """
        Compute statistics for normalization.
        
        Args:
            aggregate: Aggregate power data
            appliances: Individual appliance power data
            
        Returns:
            self
        """
        # Clean data first
        aggregate = self._fill_nan(aggregate.copy())
        
        # Compute aggregate statistics
        self.stats["aggregate"] = {
            "mean": float(np.mean(aggregate)),
            "std": float(np.std(aggregate)),
            "min": float(np.min(aggregate)),
            "max": float(np.max(aggregate)),
        }
        
        # Handle zero std
        if self.stats["aggregate"]["std"] == 0:
            self.stats["aggregate"]["std"] = 1.0
            
        # Compute appliance statistics
        if appliances is not None:
            appliances = self._fill_nan(appliances.copy())
            n_appliances = appliances.shape[1] if appliances.ndim > 1 else 1
            
            self.stats["appliances"] = []
            for i in range(n_appliances):
                app_data = appliances[:, i] if appliances.ndim > 1 else appliances
                self.stats["appliances"].append({
                    "mean": float(np.mean(app_data)),
                    "std": float(np.std(app_data)),
                    "min": float(np.min(app_data)),
                    "max": float(np.max(app_data)),
                    "threshold": float(np.percentile(app_data[app_data > 0], 50)) if np.any(app_data > 0) else 0,
                })
                if self.stats["appliances"][i]["std"] == 0:
                    self.stats["appliances"][i]["std"] = 1.0
                    
        return self
    
    def transform(
        self,
        aggregate: np.ndarray,
        appliances: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform data using computed statistics.
        
        Args:
            aggregate: Aggregate power data
            appliances: Individual appliance power data (optional)
            
        Returns:
            Transformed data (aggregate only or tuple of aggregate and appliances)
        """
        # Clean data
        aggregate = self._fill_nan(aggregate.copy())
        
        # Clip outliers
        if self.clip_outliers:
            threshold = np.percentile(aggregate, self.outlier_percentile)
            aggregate = np.clip(aggregate, 0, threshold)
        
        # Normalize
        if self.normalize:
            aggregate = (aggregate - self.stats["aggregate"]["mean"]) / self.stats["aggregate"]["std"]
            
        if appliances is None:
            return aggregate
            
        appliances = self._fill_nan(appliances.copy())
        
        return aggregate, appliances
    
    def inverse_transform(
        self,
        data: np.ndarray,
        is_aggregate: bool = True,
    ) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            is_aggregate: Whether data is aggregate power
            
        Returns:
            Denormalized data
        """
        if not self.normalize:
            return data
            
        if is_aggregate:
            return data * self.stats["aggregate"]["std"] + self.stats["aggregate"]["mean"]
        else:
            # For appliance data, we don't normalize, so just return as is
            return data
    
    def fit_transform(
        self,
        aggregate: np.ndarray,
        appliances: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit and transform in one step.
        
        Args:
            aggregate: Aggregate power data
            appliances: Individual appliance power data
            
        Returns:
            Transformed data
        """
        self.fit(aggregate, appliances)
        return self.transform(aggregate, appliances)
    
    def _fill_nan(self, data: np.ndarray) -> np.ndarray:
        """Fill NaN values in data."""
        if not np.any(np.isnan(data)):
            return data
            
        if self.fill_nan_method == "zero":
            data = np.nan_to_num(data, nan=0.0)
            
        elif self.fill_nan_method == "mean":
            mean_val = np.nanmean(data)
            data = np.nan_to_num(data, nan=mean_val)
            
        elif self.fill_nan_method == "interpolate":
            if data.ndim == 1:
                data = self._interpolate_1d(data)
            else:
                for i in range(data.shape[1]):
                    data[:, i] = self._interpolate_1d(data[:, i])
                    
        elif self.fill_nan_method == "forward":
            if data.ndim == 1:
                data = self._forward_fill_1d(data)
            else:
                for i in range(data.shape[1]):
                    data[:, i] = self._forward_fill_1d(data[:, i])
                    
        return data
    
    @staticmethod
    def _interpolate_1d(arr: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in 1D array."""
        nan_mask = np.isnan(arr)
        if not np.any(nan_mask):
            return arr
            
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) == 0:
            return np.zeros_like(arr)
            
        nan_indices = np.where(nan_mask)[0]
        arr[nan_indices] = np.interp(nan_indices, valid_indices, arr[valid_indices])
        return arr
    
    @staticmethod
    def _forward_fill_1d(arr: np.ndarray) -> np.ndarray:
        """Forward fill NaN values in 1D array."""
        nan_mask = np.isnan(arr)
        if not np.any(nan_mask):
            return arr
            
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) == 0:
            return np.zeros_like(arr)
            
        # Forward fill
        first_valid_value = arr[valid_indices[0]]
        for i in range(len(arr)):
            if nan_mask[i]:
                if i > 0:
                    arr[i] = arr[i - 1]
                else:
                    # Use first valid value for leading NaNs
                    arr[i] = first_valid_value
                    
        return arr
    
    def create_windows(
        self,
        aggregate: np.ndarray,
        appliances: Optional[np.ndarray] = None,
        output_type: str = "seq2point",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Create sliding windows from data.
        
        Args:
            aggregate: Aggregate power data
            appliances: Individual appliance power data
            output_type: "seq2point" or "seq2seq"
            
        Returns:
            Windows of shape (n_windows, window_size) or 
            tuple of windows and targets
        """
        n_windows = (len(aggregate) - self.window_size) // self.stride + 1
        
        # Create aggregate windows
        agg_windows = np.zeros((n_windows, self.window_size), dtype=np.float32)
        
        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            agg_windows[i] = aggregate[start:end]
            
        if appliances is None:
            return agg_windows
        
        # Ensure appliances is 2D
        if appliances.ndim == 1:
            appliances = appliances.reshape(-1, 1)
            
        # Create target windows/points
        if output_type == "seq2point":
            # Target is middle point
            mid_offset = self.window_size // 2
            n_appliances = appliances.shape[1]
            targets = np.zeros((n_windows, n_appliances), dtype=np.float32)
            
            for i in range(n_windows):
                mid_idx = i * self.stride + mid_offset
                targets[i] = appliances[mid_idx]
        else:
            # Seq2Seq: target is full window
            n_appliances = appliances.shape[1]
            targets = np.zeros((n_windows, self.window_size, n_appliances), dtype=np.float32)
            
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size
                targets[i] = appliances[start:end]
                
        return agg_windows, targets
    
    def get_stats(self) -> Dict:
        """Get computed statistics."""
        return self.stats.copy()
    
    def save_stats(self, path: str) -> None:
        """Save statistics to file."""
        import json
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
            
    def load_stats(self, path: str) -> None:
        """Load statistics from file."""
        import json
        with open(path, "r") as f:
            self.stats = json.load(f)


def resample_data(
    data: np.ndarray,
    timestamps: np.ndarray,
    target_freq: str = "1S",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data to target frequency.
    
    Args:
        data: Power consumption data
        timestamps: Timestamps for data
        target_freq: Target frequency (e.g., "1S", "6S", "1min")
        
    Returns:
        Resampled data and timestamps
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for resampling. Install with: pip install pandas")
    
    df = pd.DataFrame(data, index=pd.to_datetime(timestamps))
    resampled = df.resample(target_freq).mean()
    
    # Forward fill any gaps
    resampled = resampled.ffill()
    
    return resampled.values, resampled.index.values


def compute_on_off_states(
    power: np.ndarray,
    threshold: float = 10.0,
    min_on_duration: int = 1,
    min_off_duration: int = 1,
) -> np.ndarray:
    """
    Convert power readings to binary on/off states.
    
    Args:
        power: Power consumption data
        threshold: Power threshold for "on" state
        min_on_duration: Minimum consecutive samples for on state
        min_off_duration: Minimum consecutive samples for off state
        
    Returns:
        Binary on/off state array
    """
    # Initial thresholding
    states = (power > threshold).astype(np.int32)
    
    # Remove short on/off periods
    i = 0
    while i < len(states):
        # Find start of current run
        current_state = states[i]
        j = i
        while j < len(states) and states[j] == current_state:
            j += 1
            
        run_length = j - i
        
        # Check if run is too short
        if current_state == 1 and run_length < min_on_duration:
            states[i:j] = 0
        elif current_state == 0 and run_length < min_off_duration:
            states[i:j] = 1
            
        i = j
        
    return states
