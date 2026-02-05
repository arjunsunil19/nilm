"""Evaluation metrics for NILM system."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_relative_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Relative MAE as percentage."""
    mae = mean_absolute_error(y_true, y_pred)
    mean_true = np.mean(y_true)
    if mean_true == 0:
        return 0.0
    return float((mae / mean_true) * 100)


def calculate_energy_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Energy Accuracy as percentage.
    
    Energy Accuracy = 1 - |sum(y_pred) - sum(y_true)| / sum(y_true)
    """
    total_true = np.sum(y_true)
    total_pred = np.sum(y_pred)
    
    if total_true == 0:
        return 0.0
    
    accuracy = 1 - (abs(total_pred - total_true) / total_true)
    return float(max(0, accuracy * 100))  # Clamp to [0, 100]


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True state labels
        y_pred: Predicted state labels
        num_classes: Number of classes
        
    Returns:
        Dictionary containing classification metrics
    """
    # Ensure integer labels
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Calculate metrics
    average_method = 'binary' if num_classes == 2 else 'weighted'
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, average=average_method, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=average_method, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average_method, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred))
    }
    
    return metrics


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def calculate_all_metrics(
    y_true_power: np.ndarray,
    y_pred_power: np.ndarray,
    y_true_state: np.ndarray,
    y_pred_state: np.ndarray,
    num_classes: int,
    appliance_name: str
) -> Dict[str, float]:
    """
    Calculate all metrics for a single appliance.
    
    Args:
        y_true_power: True power values
        y_pred_power: Predicted power values
        y_true_state: True state labels
        y_pred_state: Predicted state labels
        num_classes: Number of classes for the appliance
        appliance_name: Name of the appliance
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Power regression metrics
    metrics['mae'] = calculate_mae(y_true_power, y_pred_power)
    metrics['rmse'] = calculate_rmse(y_true_power, y_pred_power)
    metrics['relative_mae'] = calculate_relative_mae(y_true_power, y_pred_power)
    metrics['energy_accuracy'] = calculate_energy_accuracy(y_true_power, y_pred_power)
    
    # Classification metrics
    class_metrics = calculate_classification_metrics(y_true_state, y_pred_state, num_classes)
    metrics.update(class_metrics)
    
    logger.info(f"Metrics for {appliance_name}:")
    logger.info(f"  MAE: {metrics['mae']:.2f}W, RMSE: {metrics['rmse']:.2f}W")
    logger.info(f"  Energy Accuracy: {metrics['energy_accuracy']:.2f}%")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def aggregate_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across all appliances.
    
    Args:
        metrics_dict: Dictionary of metrics for each appliance
        
    Returns:
        Dictionary of averaged metrics
    """
    if not metrics_dict:
        return {}
    
    # Get all metric keys from first appliance
    metric_keys = list(next(iter(metrics_dict.values())).keys())
    
    aggregated = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in metrics_dict.values()]
        aggregated[f'avg_{key}'] = float(np.mean(values))
        aggregated[f'std_{key}'] = float(np.std(values))
    
    logger.info("Aggregated metrics across all appliances:")
    logger.info(f"  Avg MAE: {aggregated.get('avg_mae', 0):.2f}W")
    logger.info(f"  Avg F1-Score: {aggregated.get('avg_f1_score', 0):.4f}")
    logger.info(f"  Avg Energy Accuracy: {aggregated.get('avg_energy_accuracy', 0):.2f}%")
    
    return aggregated
