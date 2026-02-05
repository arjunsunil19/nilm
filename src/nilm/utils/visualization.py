"""
Visualization utilities for NILM.

Provides plotting functions for visualizing predictions, training history,
and model comparisons.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Visualizer:
    """
    Visualization utilities for NILM results.
    
    Provides methods for:
    - Plotting predictions vs ground truth
    - Visualizing training history
    - Comparing multiple models
    - Creating publication-ready figures
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the Visualizer.
        
        Args:
            figsize: Default figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )
        
        self.figsize = figsize
        
        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "NILM Predictions",
        appliance_name: str = "Appliance",
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        show_aggregate: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot predictions against ground truth.
        
        Args:
            y_true: Ground truth power values
            y_pred: Predicted power values
            title: Plot title
            appliance_name: Name of the appliance
            start_idx: Start index for plotting
            end_idx: End index for plotting (None for all)
            show_aggregate: Aggregate power to show (optional)
            save_path: Path to save the figure (optional)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        end_idx = end_idx or len(y_true)
        y_true = y_true[start_idx:end_idx]
        y_pred = y_pred[start_idx:end_idx]
        
        num_subplots = 2 if show_aggregate is None else 3
        fig, axes = plt.subplots(num_subplots, 1, figsize=(self.figsize[0], self.figsize[1] * num_subplots // 2))
        
        if num_subplots == 2:
            axes = [axes[0], axes[1]]
        
        # Time axis
        x = np.arange(len(y_true))
        
        ax_idx = 0
        
        # Plot aggregate if provided
        if show_aggregate is not None:
            show_aggregate = np.array(show_aggregate).flatten()[start_idx:end_idx]
            axes[ax_idx].plot(x, show_aggregate, "b-", alpha=0.7, label="Aggregate")
            axes[ax_idx].set_ylabel("Power (W)")
            axes[ax_idx].set_title("Aggregate Power")
            axes[ax_idx].legend()
            ax_idx += 1
        
        # Plot ground truth and predictions
        axes[ax_idx].plot(x, y_true, "b-", alpha=0.7, label="Ground Truth", linewidth=1.5)
        axes[ax_idx].plot(x, y_pred, "r--", alpha=0.7, label="Prediction", linewidth=1.5)
        axes[ax_idx].set_ylabel("Power (W)")
        axes[ax_idx].set_title(f"{appliance_name} - Prediction vs Ground Truth")
        axes[ax_idx].legend()
        ax_idx += 1
        
        # Plot difference
        diff = y_pred - y_true
        axes[ax_idx].fill_between(x, 0, diff, alpha=0.5, color="green", where=diff >= 0, label="Over-estimate")
        axes[ax_idx].fill_between(x, 0, diff, alpha=0.5, color="red", where=diff < 0, label="Under-estimate")
        axes[ax_idx].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[ax_idx].set_ylabel("Error (W)")
        axes[ax_idx].set_xlabel("Time step")
        axes[ax_idx].set_title("Prediction Error")
        axes[ax_idx].legend()
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Training history dictionary with 'loss', 'val_loss', etc.
            title: Plot title
            save_path: Path to save the figure
        """
        # Handle both keras History objects and dictionaries
        if hasattr(history, "history"):
            history = history.history
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot loss
        epochs = range(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], "b-", label="Training Loss")
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        
        # Plot MAE if available
        if "mae" in history:
            axes[1].plot(epochs, history["mae"], "b-", label="Training MAE")
            if "val_mae" in history:
                axes[1].plot(epochs, history["val_mae"], "r-", label="Validation MAE")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].set_title("Training and Validation MAE")
            axes[1].legend()
        else:
            # Plot learning rate if available
            if "lr" in history:
                axes[1].plot(epochs, history["lr"], "g-")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Learning Rate")
                axes[1].set_title("Learning Rate Schedule")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            results: Dictionary mapping model names to their metrics
            metrics: List of metrics to compare (default: MAE, RMSE, F1)
            title: Plot title
            save_path: Path to save the figure
        """
        metrics = metrics or ["mae", "rmse", "f1"]
        models = list(results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = [results[model].get(metric, 0) for model in models]
            bars = ax.bar(models, values, color=plt.cm.tab10(range(len(models))))
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Comparison")
            ax.tick_params(axis="x", rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f"{value:.3f}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha="center", va="bottom")
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_power_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Power Distribution",
        bins: int = 50,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of power values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            bins: Number of histogram bins
            save_path: Path to save the figure
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram comparison
        axes[0].hist(y_true, bins=bins, alpha=0.5, label="Ground Truth", density=True)
        axes[0].hist(y_pred, bins=bins, alpha=0.5, label="Prediction", density=True)
        axes[0].set_xlabel("Power (W)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Power Distribution")
        axes[0].legend()
        
        # Scatter plot (predicted vs actual)
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=1)
        max_val = max(np.max(y_true), np.max(y_pred))
        axes[1].plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
        axes[1].set_xlabel("Ground Truth (W)")
        axes[1].set_ylabel("Prediction (W)")
        axes[1].set_title("Predicted vs Actual")
        axes[1].legend()
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
