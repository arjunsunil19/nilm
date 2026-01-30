"""
NILM Metrics

Evaluation metrics for Non-Intrusive Load Monitoring.
"""

from typing import Optional, Tuple

import numpy as np


class NILMMetrics:
    """
    Collection of metrics for evaluating NILM models.
    
    Includes standard regression metrics, classification metrics,
    and NILM-specific metrics from the literature.
    """
    
    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(predictions - targets)))
    
    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean Squared Error.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            MSE value
        """
        return float(np.mean((predictions - targets) ** 2))
    
    @staticmethod
    def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    @staticmethod
    def signal_aggregate_error(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Signal Aggregate Error (SAE).
        
        Measures the relative error in total energy estimation.
        
        SAE = |sum(predictions) - sum(targets)| / sum(targets)
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            SAE value
        """
        total_pred = np.sum(predictions)
        total_true = np.sum(targets)
        
        if total_true == 0:
            return float("nan") if total_pred != 0 else 0.0
            
        return float(np.abs(total_pred - total_true) / total_true)
    
    @staticmethod
    def normalized_disaggregation_error(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Normalized Disaggregation Error (NDE).
        
        NDE = sqrt(sum((predictions - targets)^2) / sum(targets^2))
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            NDE value
        """
        num = np.sum((predictions - targets) ** 2)
        denom = np.sum(targets ** 2)
        
        if denom == 0:
            return float("nan") if num != 0 else 0.0
            
        return float(np.sqrt(num / denom))
    
    @staticmethod
    def energy_accuracy(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Energy Accuracy (EAC).
        
        Measures accuracy of total energy estimation.
        EAC = 1 - SAE
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            EAC value (higher is better)
        """
        sae = NILMMetrics.signal_aggregate_error(predictions, targets)
        if np.isnan(sae):
            return float("nan")
        return float(max(0, 1 - sae))
    
    @staticmethod
    def nep(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Normalized Error in assigned Power (NEP).
        
        NEP = sum(|predictions - targets|) / sum(targets)
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            NEP value
        """
        total_true = np.sum(targets)
        
        if total_true == 0:
            return float("nan")
            
        return float(np.sum(np.abs(predictions - targets)) / total_true)
    
    @staticmethod
    def f1_score(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        F1 Score for binary on/off classification.
        
        Args:
            predictions: Binary predictions (0 or 1)
            targets: Binary ground truth (0 or 1)
            
        Returns:
            F1 score
        """
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
            
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
            
        if precision + recall == 0:
            return 0.0
            
        return float(2 * precision * recall / (precision + recall))
    
    @staticmethod
    def precision(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Precision for binary classification.
        
        Args:
            predictions: Binary predictions
            targets: Binary ground truth
            
        Returns:
            Precision value
        """
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        
        if tp + fp == 0:
            return 0.0
            
        return float(tp / (tp + fp))
    
    @staticmethod
    def recall(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Recall for binary classification.
        
        Args:
            predictions: Binary predictions
            targets: Binary ground truth
            
        Returns:
            Recall value
        """
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        tp = np.sum((predictions == 1) & (targets == 1))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        if tp + fn == 0:
            return 0.0
            
        return float(tp / (tp + fn))
    
    @staticmethod
    def accuracy(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Accuracy for binary classification.
        
        Args:
            predictions: Binary predictions
            targets: Binary ground truth
            
        Returns:
            Accuracy value
        """
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        return float(np.mean(predictions == targets))
    
    @staticmethod
    def matthews_correlation_coefficient(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Matthews Correlation Coefficient (MCC).
        
        A balanced measure for binary classification that works well
        even with imbalanced classes.
        
        Args:
            predictions: Binary predictions
            targets: Binary ground truth
            
        Returns:
            MCC value between -1 and 1
        """
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        tp = np.sum((predictions == 1) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        numerator = float((tp * tn) - (fp * fn))
        
        # Handle edge cases where one or more confusion matrix sums are zero
        denom_product = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom_product == 0:
            # When all predictions or all targets are the same class,
            # MCC is undefined. Return 0.0 by convention.
            return 0.0
        
        denominator = np.sqrt(denom_product)
        return float(numerator / denominator)
    
    @staticmethod
    def proportion_of_total_energy(
        predictions: np.ndarray,
        aggregate: np.ndarray,
    ) -> float:
        """
        Proportion of Total Energy correctly assigned.
        
        Args:
            predictions: Predicted appliance power
            aggregate: Aggregate power
            
        Returns:
            Proportion (0 to 1)
        """
        total_agg = np.sum(aggregate)
        total_pred = np.sum(np.maximum(predictions, 0))  # Only positive values
        
        if total_agg == 0:
            return float("nan")
            
        return float(min(total_pred / total_agg, 1.0))
    
    @staticmethod
    def compute_all_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Compute all available metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            threshold: Optional threshold for binary classification
            
        Returns:
            Dictionary with all metric values
        """
        metrics = NILMMetrics()
        
        results = {
            "mae": metrics.mae(predictions, targets),
            "mse": metrics.mse(predictions, targets),
            "rmse": metrics.rmse(predictions, targets),
            "sae": metrics.signal_aggregate_error(predictions, targets),
            "nde": metrics.normalized_disaggregation_error(predictions, targets),
            "eac": metrics.energy_accuracy(predictions, targets),
            "nep": metrics.nep(predictions, targets),
        }
        
        # Add classification metrics if threshold is provided
        if threshold is not None:
            pred_binary = (predictions > threshold).astype(int)
            true_binary = (targets > threshold).astype(int)
            
            results["f1"] = metrics.f1_score(pred_binary, true_binary)
            results["precision"] = metrics.precision(pred_binary, true_binary)
            results["recall"] = metrics.recall(pred_binary, true_binary)
            results["accuracy"] = metrics.accuracy(pred_binary, true_binary)
            results["mcc"] = metrics.matthews_correlation_coefficient(pred_binary, true_binary)
            
        return results
