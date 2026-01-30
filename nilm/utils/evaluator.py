"""
NILM Model Evaluator

Evaluation utilities and metrics for NILM models.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nilm.models.base import BaseNILMModel
from nilm.utils.metrics import NILMMetrics


class Evaluator:
    """
    Evaluator class for NILM models.
    
    Computes comprehensive metrics for energy disaggregation
    including regression metrics, classification metrics,
    and NILM-specific metrics.
    """
    
    def __init__(
        self,
        model: BaseNILMModel,
        device: Optional[str] = None,
        appliance_names: Optional[List[str]] = None,
        appliance_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained NILM model
            device: Device to use for evaluation
            appliance_names: Names of appliances
            appliance_thresholds: Power thresholds for on/off classification
        """
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.appliance_names = appliance_names or []
        self.appliance_thresholds = appliance_thresholds or {}
        
        self.metrics = NILMMetrics()
        
    def predict(
        self,
        data_loader: DataLoader,
        return_targets: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader with test data
            return_targets: Whether to return ground truth
            
        Returns:
            Tuple of (predictions, targets) or just predictions
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                    targets.append(y.numpy())
                else:
                    x = batch
                    y = None
                    
                x = x.to(self.device)
                outputs = self.model(x)
                predictions.append(outputs.cpu().numpy())
                
        predictions = np.concatenate(predictions, axis=0)
        
        if return_targets and len(targets) > 0:
            targets = np.concatenate(targets, axis=0)
            return predictions, targets
        else:
            return predictions, None
            
    def evaluate(
        self,
        data_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader with test data
            verbose: Whether to print results
            
        Returns:
            Dictionary of metrics
        """
        predictions, targets = self.predict(data_loader, return_targets=True)
        
        if targets is None:
            raise ValueError("No targets available for evaluation")
            
        # Compute overall metrics
        results = {}
        
        # MAE and MSE
        results["mae"] = float(np.mean(np.abs(predictions - targets)))
        results["mse"] = float(np.mean((predictions - targets) ** 2))
        results["rmse"] = float(np.sqrt(results["mse"]))
        
        # Relative error
        total_energy = np.sum(np.abs(targets))
        if total_energy > 0:
            results["relative_error"] = float(np.sum(np.abs(predictions - targets)) / total_energy)
        else:
            results["relative_error"] = float("nan")
            
        # Per-appliance metrics
        if len(self.appliance_names) > 0 and predictions.ndim > 1:
            for i, name in enumerate(self.appliance_names):
                pred_app = predictions[:, i] if predictions.ndim > 1 else predictions
                true_app = targets[:, i] if targets.ndim > 1 else targets
                
                results[f"{name}_mae"] = float(np.mean(np.abs(pred_app - true_app)))
                results[f"{name}_rmse"] = float(np.sqrt(np.mean((pred_app - true_app) ** 2)))
                
                # Energy accuracy
                results[f"{name}_eac"] = self.metrics.energy_accuracy(pred_app, true_app)
                
                # F1 score for on/off classification
                if name in self.appliance_thresholds:
                    threshold = self.appliance_thresholds[name]
                    pred_on = (pred_app > threshold).astype(int)
                    true_on = (true_app > threshold).astype(int)
                    
                    results[f"{name}_f1"] = self.metrics.f1_score(pred_on, true_on)
                    results[f"{name}_precision"] = self.metrics.precision(pred_on, true_on)
                    results[f"{name}_recall"] = self.metrics.recall(pred_on, true_on)
                    
        # Aggregate NILM-specific metrics
        results["sae"] = self.metrics.signal_aggregate_error(predictions, targets)
        results["nde"] = self.metrics.normalized_disaggregation_error(predictions, targets)
        
        if verbose:
            self._print_results(results)
            
        return results
    
    def _print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 50)
        print("NILM Evaluation Results")
        print("=" * 50)
        
        print("\nOverall Metrics:")
        print(f"  MAE:  {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  SAE:  {results['sae']:.4f}")
        print(f"  NDE:  {results['nde']:.4f}")
        
        if len(self.appliance_names) > 0:
            print("\nPer-Appliance Metrics:")
            for name in self.appliance_names:
                print(f"\n  {name}:")
                print(f"    MAE:  {results.get(f'{name}_mae', 'N/A'):.4f}")
                print(f"    RMSE: {results.get(f'{name}_rmse', 'N/A'):.4f}")
                print(f"    EAC:  {results.get(f'{name}_eac', 'N/A'):.4f}")
                
                if f"{name}_f1" in results:
                    print(f"    F1:   {results[f'{name}_f1']:.4f}")
                    
        print("\n" + "=" * 50)
        
    def compare_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        n_samples: int = 5,
    ) -> None:
        """
        Compare predictions with targets for specific samples.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            sample_indices: Specific indices to compare
            n_samples: Number of random samples if indices not provided
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)
            
        print("\nSample Predictions vs Targets:")
        print("-" * 40)
        
        for idx in sample_indices:
            print(f"\nSample {idx}:")
            
            if predictions.ndim == 1:
                print(f"  Predicted: {predictions[idx]:.2f}")
                print(f"  Target:    {targets[idx]:.2f}")
                print(f"  Error:     {abs(predictions[idx] - targets[idx]):.2f}")
            else:
                for i, name in enumerate(self.appliance_names or [f"App_{i}" for i in range(predictions.shape[1])]):
                    print(f"  {name}: Pred={predictions[idx, i]:.2f}, True={targets[idx, i]:.2f}, "
                          f"Error={abs(predictions[idx, i] - targets[idx, i]):.2f}")


def evaluate_models(
    models: Dict[str, BaseNILMModel],
    test_loader: DataLoader,
    device: str = "cuda",
    appliance_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate and compare multiple models.
    
    Args:
        models: Dictionary of model name to model
        test_loader: Test data loader
        device: Device to use
        appliance_names: Names of appliances
        
    Returns:
        Dictionary of model name to results
    """
    all_results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluator = Evaluator(model, device=device, appliance_names=appliance_names)
        results = evaluator.evaluate(test_loader, verbose=False)
        all_results[name] = results
        
    # Print comparison
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'SAE':>10} {'NDE':>10}")
    print("-" * 70)
    
    for name, results in all_results.items():
        print(f"{name:<20} {results['mae']:>10.4f} {results['rmse']:>10.4f} "
              f"{results['sae']:>10.4f} {results['nde']:>10.4f}")
              
    return all_results
