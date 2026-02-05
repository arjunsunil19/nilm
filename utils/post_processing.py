"""Post-processing utilities for NILM system."""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from typing import List
import logging

logger = logging.getLogger(__name__)


def apply_median_filter(
    predictions: np.ndarray,
    filter_size: int = 5
) -> np.ndarray:
    """
    Apply median filtering to smooth predictions.
    
    Args:
        predictions: Raw predictions
        filter_size: Size of the median filter kernel
        
    Returns:
        Filtered predictions
    """
    if filter_size <= 1:
        return predictions
    
    filtered = median_filter(predictions, size=filter_size, mode='reflect')
    return filtered


def apply_gaussian_smoothing(
    predictions: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian smoothing to predictions.
    
    Args:
        predictions: Raw predictions
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed predictions
    """
    if sigma <= 0:
        return predictions
    
    smoothed = gaussian_filter1d(predictions, sigma=sigma, mode='reflect')
    return smoothed


def apply_noise_floor_threshold(
    predictions: np.ndarray,
    noise_floor: float
) -> np.ndarray:
    """
    Apply noise floor thresholding.
    
    Args:
        predictions: Raw predictions
        noise_floor: Minimum power threshold
        
    Returns:
        Thresholded predictions
    """
    thresholded = predictions.copy()
    thresholded[thresholded < noise_floor] = 0
    return thresholded


def apply_physical_bounds(
    predictions: np.ndarray,
    min_power: float = 0.0,
    max_power: float = 10000.0
) -> np.ndarray:
    """
    Apply physical bounds to predictions.
    
    Args:
        predictions: Raw predictions
        min_power: Minimum allowable power
        max_power: Maximum allowable power
        
    Returns:
        Clipped predictions
    """
    clipped = np.clip(predictions, min_power, max_power)
    return clipped


def refine_states_from_power(
    power_predictions: np.ndarray,
    thresholds: List[float],
    appliance_type: str
) -> np.ndarray:
    """
    Refine state predictions based on power values.
    
    Args:
        power_predictions: Predicted power values
        thresholds: Power thresholds for state classification
        appliance_type: 'binary' or 'multi_state'
        
    Returns:
        Refined state labels
    """
    if appliance_type == 'binary':
        states = (power_predictions > thresholds[0]).astype(np.int32)
    else:
        states = np.zeros_like(power_predictions, dtype=np.int32)
        for i, threshold in enumerate(thresholds):
            states[power_predictions > threshold] = i + 1
    
    return states


def post_process_predictions(
    power_predictions: np.ndarray,
    state_predictions: np.ndarray,
    thresholds: List[float],
    appliance_type: str,
    noise_floor: float = 5.0,
    median_filter_size: int = 5,
    gaussian_sigma: float = 1.0,
    max_power: float = 10000.0
) -> tuple:
    """
    Apply full post-processing pipeline to predictions.
    
    Args:
        power_predictions: Raw power predictions
        state_predictions: Raw state predictions
        thresholds: Power thresholds
        appliance_type: 'binary' or 'multi_state'
        noise_floor: Minimum power threshold
        median_filter_size: Size of median filter
        gaussian_sigma: Sigma for Gaussian smoothing
        max_power: Maximum allowable power
        
    Returns:
        Tuple of (processed_power, processed_states)
    """
    # Apply median filtering
    power_filtered = apply_median_filter(power_predictions, median_filter_size)
    
    # Apply Gaussian smoothing
    power_smoothed = apply_gaussian_smoothing(power_filtered, gaussian_sigma)
    
    # Apply noise floor threshold
    power_thresholded = apply_noise_floor_threshold(power_smoothed, noise_floor)
    
    # Apply physical bounds
    power_final = apply_physical_bounds(power_thresholded, 0.0, max_power)
    
    # Refine states based on final power predictions
    states_final = refine_states_from_power(power_final, thresholds, appliance_type)
    
    logger.info(f"Post-processing applied: median={median_filter_size}, gaussian={gaussian_sigma}, noise_floor={noise_floor}")
    
    return power_final, states_final
