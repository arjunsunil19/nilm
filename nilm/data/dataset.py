"""
NILM Dataset

PyTorch Dataset implementations for NILM data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class NILMDataset(Dataset):
    """
    PyTorch Dataset for NILM data.
    
    Handles windowing and preprocessing of power consumption data
    for training and evaluation of NILM models.
    """
    
    def __init__(
        self,
        aggregate: np.ndarray,
        appliances: Optional[np.ndarray] = None,
        window_size: int = 599,
        stride: int = 1,
        output_type: str = "seq2point",
        normalize: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        appliance_stats: Optional[Dict] = None,
    ):
        """
        Initialize NILM Dataset.
        
        Args:
            aggregate: Aggregate power consumption array of shape (n_samples,)
            appliances: Individual appliance power array of shape (n_samples, n_appliances)
                       or None for inference
            window_size: Size of the sliding window
            stride: Stride for sliding window
            output_type: "seq2point" (single output) or "seq2seq" (sequence output)
            normalize: Whether to normalize the data
            mean: Pre-computed mean for normalization
            std: Pre-computed std for normalization
            appliance_stats: Dictionary of appliance normalization stats
        """
        self.aggregate = aggregate.astype(np.float32)
        self.appliances = appliances.astype(np.float32) if appliances is not None else None
        self.window_size = window_size
        self.stride = stride
        self.output_type = output_type
        self.normalize = normalize
        
        # Compute or use provided normalization stats
        if normalize:
            self.mean = mean if mean is not None else np.mean(aggregate)
            self.std = std if std is not None else np.std(aggregate)
            if self.std == 0:
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0
            
        self.appliance_stats = appliance_stats
        
        # Calculate number of windows
        self.n_windows = max(0, (len(aggregate) - window_size) // stride + 1)
        
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If appliances is None: (aggregate_window,)
            Otherwise: (aggregate_window, target)
            where target depends on output_type
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Get aggregate window
        agg_window = self.aggregate[start_idx:end_idx].copy()
        
        # Normalize
        if self.normalize:
            agg_window = (agg_window - self.mean) / self.std
            
        x = torch.from_numpy(agg_window).unsqueeze(-1).float()  # (window, 1)
        
        # If no appliance data, return only aggregate
        if self.appliances is None:
            return x
            
        # Get target
        if self.output_type == "seq2point":
            mid_idx = start_idx + self.window_size // 2
            target = self.appliances[mid_idx].copy()
        else:  # seq2seq
            target = self.appliances[start_idx:end_idx].copy()
            
        y = torch.from_numpy(target).float()
        
        return x, y
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            "mean": self.mean,
            "std": self.std,
            "n_samples": len(self),
            "window_size": self.window_size,
            "stride": self.stride,
        }


class NILMDatasetFromDict(Dataset):
    """
    NILM Dataset that loads data from a dictionary structure.
    
    Useful for working with REDD, UK-DALE, and other common NILM datasets.
    """
    
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        aggregate_key: str = "aggregate",
        appliance_keys: Optional[List[str]] = None,
        window_size: int = 599,
        stride: int = 1,
        output_type: str = "seq2point",
        normalize: bool = True,
    ):
        """
        Initialize from dictionary.
        
        Args:
            data_dict: Dictionary with aggregate and appliance power data
            aggregate_key: Key for aggregate power in dict
            appliance_keys: List of keys for appliance power data
            window_size: Size of sliding window
            stride: Stride for sliding window
            output_type: "seq2point" or "seq2seq"
            normalize: Whether to normalize data
        """
        self.data_dict = data_dict
        aggregate = data_dict[aggregate_key]
        
        if appliance_keys:
            self.appliance_keys = appliance_keys
            appliances = np.stack([data_dict[k] for k in appliance_keys], axis=1)
        else:
            self.appliance_keys = []
            appliances = None
            
        # Create base dataset
        self._dataset = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=window_size,
            stride=stride,
            output_type=output_type,
            normalize=normalize,
        )
        
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int):
        return self._dataset[idx]
    
    def get_stats(self) -> Dict:
        stats = self._dataset.get_stats()
        stats["appliance_keys"] = self.appliance_keys
        return stats


class SyntheticNILMDataset(Dataset):
    """
    Synthetic NILM dataset for testing and demonstration.
    
    Generates synthetic power consumption patterns for multiple appliances.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_appliances: int = 5,
        window_size: int = 599,
        noise_level: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            n_samples: Number of time steps to generate
            n_appliances: Number of appliances
            window_size: Size of sliding window
            noise_level: Level of noise to add
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.n_samples = n_samples
        self.n_appliances = n_appliances
        self.window_size = window_size
        
        # Generate synthetic appliance patterns
        self.appliances = self._generate_appliances(n_samples, n_appliances)
        
        # Aggregate with noise
        self.aggregate = np.sum(self.appliances, axis=1)
        noise = np.random.randn(n_samples) * noise_level * np.std(self.aggregate)
        self.aggregate = self.aggregate + noise
        self.aggregate = np.maximum(self.aggregate, 0)  # Power can't be negative
        
        # Create dataset
        self._dataset = NILMDataset(
            aggregate=self.aggregate,
            appliances=self.appliances,
            window_size=window_size,
            stride=1,
            normalize=True,
        )
        
    def _generate_appliances(
        self, 
        n_samples: int, 
        n_appliances: int
    ) -> np.ndarray:
        """Generate synthetic appliance power patterns."""
        appliances = np.zeros((n_samples, n_appliances), dtype=np.float32)
        
        # Typical power levels for different appliances (in watts)
        power_levels = [100, 500, 1000, 2000, 3000][:n_appliances]
        
        for i in range(n_appliances):
            power = power_levels[i] if i < len(power_levels) else np.random.uniform(50, 3000)
            
            # Generate on/off patterns
            state = np.zeros(n_samples)
            t = 0
            while t < n_samples:
                # Random on duration
                if np.random.random() > 0.5:
                    on_duration = np.random.randint(50, 500)
                    end_t = min(t + on_duration, n_samples)
                    state[t:end_t] = 1
                    t = end_t
                else:
                    t += np.random.randint(100, 1000)
                    
            # Add some variability to power when on
            variability = np.random.randn(n_samples) * 0.05 * power
            appliances[:, i] = state * (power + variability)
            appliances[:, i] = np.maximum(appliances[:, i], 0)
            
        return appliances
        
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int):
        return self._dataset[idx]
    
    def get_stats(self) -> Dict:
        return self._dataset.get_stats()
