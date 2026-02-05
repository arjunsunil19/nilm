"""
Metrics Calculator for NILM

Standard metrics for evaluating NILM model performance.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class MetricsCalculator:
    """
    Calculator for NILM evaluation metrics.
    
    Implements standard metrics used in NILM research:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error)
    - SAE (Signal Aggregate Error)
    - NDE (Normalized Disaggregation Error)
    - F1 Score (for on/off detection)
    
    Reference:
        Kolter, J. Z., & Johnson, M. J. (2011). REDD: A public data set for
        energy disaggregation research.
    """
    
    def __init__(self, appliance_name: Optional[str] = None):
        """
        Initialize MetricsCalculator.
        
        Args:
            appliance_name: Name of appliance (for threshold-based metrics)
        """
        self.appliance_name = appliance_name
        
        # Default on/off thresholds per appliance
        self.thresholds = {
            "kettle": 2000,
            "microwave": 200,
            "fridge": 50,
            "dishwasher": 10,
            "washing_machine": 20,
        }
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def sae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Signal Aggregate Error (relative error in total energy).
        
        SAE = |sum(y_pred) - sum(y_true)| / sum(y_true)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            SAE value (0 to infinity, lower is better)
        """
        total_true = np.sum(y_true)
        if total_true == 0:
            return float("inf") if np.sum(y_pred) > 0 else 0.0
        
        return float(np.abs(np.sum(y_pred) - total_true) / total_true)
    
    def nde(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Normalized Disaggregation Error.
        
        NDE = sum(|y_pred - y_true|^2) / sum(y_true^2)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            NDE value (0 to infinity, lower is better)
        """
        denominator = np.sum(y_true ** 2)
        if denominator == 0:
            return float("inf") if np.sum(y_pred ** 2) > 0 else 0.0
        
        return float(np.sum((y_pred - y_true) ** 2) / denominator)
    
    def eac(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Energy Accuracy (1 - SAE, capped at 0).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            EAC value (0 to 1, higher is better)
        """
        return max(0, 1 - self.sae(y_true, y_pred))
    
    def f1_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate F1 score for on/off detection.
        
        Args:
            y_true: Ground truth power values
            y_pred: Predicted power values
            threshold: On/off threshold (uses appliance default if None)
            
        Returns:
            Dictionary with precision, recall, f1, and accuracy
        """
        if threshold is None:
            threshold = self.thresholds.get(self.appliance_name, 10)
        
        # Convert to binary (on/off)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate confusion matrix elements
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        }
    
    def relative_error_total_energy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate relative error in total energy.
        
        RE = (sum(y_pred) - sum(y_true)) / max(sum(y_true), sum(y_pred))
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Relative error (-1 to 1)
        """
        sum_true = np.sum(y_true)
        sum_pred = np.sum(y_pred)
        max_sum = max(sum_true, sum_pred)
        
        if max_sum == 0:
            return 0.0
        
        return float((sum_pred - sum_true) / max_sum)
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            threshold: On/off threshold for F1 calculation
            
        Returns:
            Dictionary of all metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure non-negative predictions
        y_pred = np.clip(y_pred, 0, None)
        
        f1_metrics = self.f1_score(y_true, y_pred, threshold)
        
        return {
            "mae": self.mae(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "sae": self.sae(y_true, y_pred),
            "nde": self.nde(y_true, y_pred),
            "eac": self.eac(y_true, y_pred),
            "relative_error": self.relative_error_total_energy(y_true, y_pred),
            **f1_metrics
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted table.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        print("\n" + "=" * 50)
        print("NILM Evaluation Metrics")
        print("=" * 50)
        
        print("\nRegression Metrics:")
        print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f} W")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f} W")
        print(f"  SAE:  {metrics.get('sae', 'N/A'):.4f}")
        print(f"  NDE:  {metrics.get('nde', 'N/A'):.4f}")
        print(f"  EAC:  {metrics.get('eac', 'N/A'):.4f}")
        
        print("\nClassification Metrics (On/Off):")
        print(f"  F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}")
        print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
        
        print("=" * 50 + "\n")
