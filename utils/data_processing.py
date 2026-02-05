"""Data processing utilities for NILM system."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from Excel or CSV file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        if file_path.endswith('.xlsx'):
            logger.info(f"Loading Excel file: {file_path}")
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def handle_missing_values(df: pd.DataFrame, method: str = 'both') -> pd.DataFrame:
    """
    Handle missing values using forward/backward fill.
    
    Args:
        df: Input DataFrame
        method: Fill method ('forward', 'backward', or 'both')
        
    Returns:
        DataFrame with filled missing values
    """
    df_filled = df.copy()
    
    if method == 'forward':
        df_filled = df_filled.fillna(method='ffill')
    elif method == 'backward':
        df_filled = df_filled.fillna(method='bfill')
    elif method == 'both':
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError(f"Invalid fill method: {method}")
    
    # Fill any remaining NaN with 0
    df_filled = df_filled.fillna(0)
    
    logger.info(f"Missing values handled using {method} fill")
    return df_filled


def clip_negative_values(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Clip negative power values to zero.
    
    Args:
        df: Input DataFrame
        columns: List of columns to clip (if None, clips all numeric columns)
        
    Returns:
        DataFrame with clipped values
    """
    df_clipped = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            df_clipped[col] = df_clipped[col].clip(lower=0)
    
    logger.info(f"Negative values clipped for {len(columns)} columns")
    return df_clipped


def compute_statistics(df: pd.DataFrame, columns: list) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to compute statistics for
        
    Returns:
        Dictionary containing statistics for each column
    """
    stats = {}
    
    for col in columns:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'total_energy': float(df[col].sum()),
                'on_time': int((df[col] > 0).sum()),
                'off_time': int((df[col] == 0).sum())
            }
    
    logger.info(f"Statistics computed for {len(stats)} columns")
    return stats


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    midpoint: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence-to-point windows for training.
    
    Args:
        X: Input features (samples, channels)
        y: Target values (samples, targets)
        window_size: Size of the input window
        midpoint: Position of the target in the window
        
    Returns:
        Tuple of windowed inputs and corresponding targets
    """
    n_samples = X.shape[0] - window_size + 1
    n_channels = X.shape[1]
    n_targets = y.shape[1] if len(y.shape) > 1 else 1
    
    X_seq = np.zeros((n_samples, window_size, n_channels), dtype=np.float32)
    y_seq = np.zeros((n_samples, n_targets), dtype=np.float32) if n_targets > 1 else np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X_seq[i] = X[i:i + window_size]
        if len(y.shape) > 1:
            y_seq[i] = y[i + midpoint]
        else:
            y_seq[i] = y[i + midpoint]
    
    logger.info(f"Created sequences: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")
    return X_seq, y_seq


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets (time-series aware).
    
    Args:
        X: Input features
        y: Target values
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = X.shape[0]
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test
