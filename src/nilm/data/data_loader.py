"""
Data Loader for NILM

Utilities for loading energy consumption data from various sources
including NILMTK datasets (UK-DALE, REDD, etc.) and CSV files.
"""

import os
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DataLoader:
    """
    Data loader for NILM datasets.
    
    Supports loading data from:
    - NILMTK HDF5 files (UK-DALE, REDD, etc.)
    - CSV files with timestamp and power columns
    - NumPy arrays
    
    Attributes:
        data_path: Path to the data file or directory
        appliances: List of target appliances
    """
    
    # Standard appliance parameters (mean, std) based on UK-DALE dataset
    APPLIANCE_PARAMS = {
        "kettle": {"mean": 700, "std": 1000, "threshold": 2000, "min_off": 0, "min_on": 12},
        "microwave": {"mean": 500, "std": 800, "threshold": 200, "min_off": 30, "min_on": 12},
        "fridge": {"mean": 200, "std": 400, "threshold": 50, "min_off": 12, "min_on": 60},
        "dishwasher": {"mean": 700, "std": 1000, "threshold": 10, "min_off": 1800, "min_on": 1800},
        "washing_machine": {"mean": 400, "std": 700, "threshold": 20, "min_off": 160, "min_on": 1800},
    }
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        appliances: Optional[List[str]] = None
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to data file (HDF5/CSV) or directory
            appliances: List of appliance names to load
        """
        self.data_path = data_path
        self.appliances = appliances or list(self.APPLIANCE_PARAMS.keys())
    
    def load_from_csv(
        self,
        aggregate_path: str,
        appliance_paths: Dict[str, str],
        timestamp_col: str = "timestamp",
        power_col: str = "power",
        resample: str = "6S"
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Load data from CSV files.
        
        Args:
            aggregate_path: Path to aggregate power CSV
            appliance_paths: Dictionary mapping appliance names to CSV paths
            timestamp_col: Name of the timestamp column
            power_col: Name of the power column
            resample: Resampling frequency (e.g., '6S' for 6 seconds)
            
        Returns:
            Tuple of (aggregate_power, dict of appliance_power series)
        """
        # Load aggregate
        aggregate_df = pd.read_csv(aggregate_path)
        aggregate_df[timestamp_col] = pd.to_datetime(aggregate_df[timestamp_col])
        aggregate_df.set_index(timestamp_col, inplace=True)
        aggregate = aggregate_df[power_col].resample(resample).mean().fillna(method="ffill")
        
        # Load appliances
        appliance_data = {}
        for appliance, path in appliance_paths.items():
            if appliance in self.appliances:
                app_df = pd.read_csv(path)
                app_df[timestamp_col] = pd.to_datetime(app_df[timestamp_col])
                app_df.set_index(timestamp_col, inplace=True)
                appliance_data[appliance] = app_df[power_col].resample(resample).mean().fillna(method="ffill")
        
        return aggregate, appliance_data
    
    def load_from_nilmtk(
        self,
        hdf5_path: str,
        building: int = 1,
        appliances: Optional[List[str]] = None,
        sample_period: int = 6
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Load data from NILMTK HDF5 format.
        
        Requires nilmtk to be installed.
        
        Args:
            hdf5_path: Path to HDF5 file
            building: Building number to load
            appliances: List of appliances (uses self.appliances if None)
            sample_period: Sampling period in seconds
            
        Returns:
            Tuple of (aggregate_power, dict of appliance_power series)
        """
        try:
            from nilmtk import DataSet
        except ImportError:
            raise ImportError(
                "nilmtk is required to load HDF5 files. "
                "Install with: pip install nilmtk"
            )
        
        appliances = appliances or self.appliances
        
        dataset = DataSet(hdf5_path)
        elec = dataset.buildings[building].elec
        
        # Load aggregate (mains)
        aggregate = elec.mains().power_series_all_data(sample_period=sample_period)
        
        # Load individual appliances
        appliance_data = {}
        for appliance in appliances:
            try:
                app_elec = elec[appliance]
                appliance_data[appliance] = app_elec.power_series_all_data(
                    sample_period=sample_period
                )
            except KeyError:
                print(f"Warning: Appliance '{appliance}' not found in building {building}")
        
        dataset.store.close()
        
        return aggregate, appliance_data
    
    def load_from_numpy(
        self,
        aggregate: np.ndarray,
        appliance_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load data from NumPy arrays.
        
        Args:
            aggregate: Aggregate power array
            appliance_data: Dictionary of appliance name to power array
            
        Returns:
            Tuple of (aggregate, appliance_data)
        """
        return aggregate, appliance_data
    
    def get_appliance_params(self, appliance: str) -> Dict:
        """
        Get standard parameters for an appliance.
        
        Args:
            appliance: Appliance name
            
        Returns:
            Dictionary with mean, std, threshold, min_off, min_on
        """
        if appliance in self.APPLIANCE_PARAMS:
            return self.APPLIANCE_PARAMS[appliance]
        else:
            # Return default parameters
            return {
                "mean": 0,
                "std": 1,
                "threshold": 10,
                "min_off": 12,
                "min_on": 12
            }
    
    def generate_synthetic_data(
        self,
        num_samples: int = 10000,
        window_size: int = 599,
        appliance: str = "kettle",
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data for testing.
        
        Creates synthetic aggregate and appliance power data
        for testing model architectures without real data.
        
        Args:
            num_samples: Number of samples to generate
            window_size: Size of each window
            appliance: Target appliance name (for parameters)
            noise_level: Amount of noise to add
            
        Returns:
            Tuple of (X, y) arrays
        """
        params = self.get_appliance_params(appliance)
        
        # Generate appliance power patterns
        y = np.zeros((num_samples, 1))
        for i in range(num_samples):
            if np.random.random() > 0.7:  # 30% chance of appliance being on
                y[i] = np.random.normal(params["mean"], params["std"] * 0.3)
                y[i] = np.clip(y[i], 0, params["mean"] * 2)
        
        # Generate aggregate (appliance + other loads + noise)
        base_load = np.random.uniform(100, 500, (num_samples, window_size, 1))
        
        # Add appliance contribution to middle of window
        X = base_load.copy()
        mid = window_size // 2
        for i in range(num_samples):
            if y[i] > 0:
                # Add appliance pattern around midpoint
                spread = np.random.randint(10, 50)
                start = max(0, mid - spread)
                end = min(window_size, mid + spread)
                X[i, start:end, 0] += y[i, 0]
        
        # Add noise
        X += np.random.normal(0, noise_level * 100, X.shape)
        X = np.clip(X, 0, None)
        
        return X.astype(np.float32), y.astype(np.float32)
