"""Visualization utilities for NILM system."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import os

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: Dict,
    save_path: str
) -> None:
    """
    Plot training history (loss, MAE, accuracy, learning rate).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history.get('loss', []), label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot MAE
    if 'mae' in history or 'power_output_mae' in history:
        mae_key = 'mae' if 'mae' in history else 'power_output_mae'
        val_mae_key = 'val_mae' if 'val_mae' in history else 'val_power_output_mae'
        axes[0, 1].plot(history.get(mae_key, []), label='Train MAE', linewidth=2)
        axes[0, 1].plot(history.get(val_mae_key, []), label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (Watts)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history or 'state_output_accuracy' in history:
        acc_key = 'accuracy' if 'accuracy' in history else 'state_output_accuracy'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_state_output_accuracy'
        axes[1, 0].plot(history.get(acc_key, []), label='Train Accuracy', linewidth=2)
        axes[1, 0].plot(history.get(val_acc_key, []), label='Val Accuracy', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")


def plot_disaggregation(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    save_path: str,
    max_samples: int = 1000
) -> None:
    """
    Plot actual vs predicted disaggregation for all appliances.
    
    Args:
        y_true: Dictionary of actual power values for each appliance
        y_pred: Dictionary of predicted power values for each appliance
        save_path: Path to save the plot
        max_samples: Maximum number of samples to plot
    """
    n_appliances = len(y_true)
    fig, axes = plt.subplots(n_appliances, 1, figsize=(15, 4 * n_appliances))
    
    if n_appliances == 1:
        axes = [axes]
    
    for idx, (appliance, true_values) in enumerate(y_true.items()):
        pred_values = y_pred.get(appliance, np.zeros_like(true_values))
        
        # Limit samples for visualization
        n_samples = min(len(true_values), max_samples)
        x = np.arange(n_samples)
        
        axes[idx].plot(x, true_values[:n_samples], label='Actual', linewidth=1.5, alpha=0.7)
        axes[idx].plot(x, pred_values[:n_samples], label='Predicted', linewidth=1.5, alpha=0.7)
        axes[idx].set_xlabel('Time (samples)')
        axes[idx].set_ylabel('Power (W)')
        axes[idx].set_title(f'{appliance} - Actual vs Predicted')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Disaggregation plot saved to {save_path}")


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    state_names: Dict[str, List[str]],
    save_path: str
) -> None:
    """
    Plot confusion matrices for all appliances.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices for each appliance
        state_names: Dictionary of state names for each appliance
        save_path: Path to save the plot
    """
    n_appliances = len(confusion_matrices)
    n_cols = min(3, n_appliances)
    n_rows = (n_appliances + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    if n_appliances == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (appliance, cm) in enumerate(confusion_matrices.items()):
        labels = state_names.get(appliance, [f'State {i}' for i in range(cm.shape[0])])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f'{appliance} - Confusion Matrix')
    
    # Hide unused subplots
    for idx in range(n_appliances, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrices plot saved to {save_path}")


def plot_energy_comparison(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """
    Plot energy comparison bar chart for all appliances.
    
    Args:
        y_true: Dictionary of actual power values for each appliance
        y_pred: Dictionary of predicted power values for each appliance
        save_path: Path to save the plot
    """
    appliances = list(y_true.keys())
    true_energy = [np.sum(y_true[app]) for app in appliances]
    pred_energy = [np.sum(y_pred[app]) for app in appliances]
    
    x = np.arange(len(appliances))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, true_energy, width, label='Actual', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_energy, width, label='Predicted', alpha=0.8)
    
    ax.set_xlabel('Appliance')
    ax.set_ylabel('Total Energy (Wh)')
    ax.set_title('Energy Consumption Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(appliances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Energy comparison plot saved to {save_path}")


def plot_power_distributions(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """
    Plot power distribution histograms for all appliances.
    
    Args:
        y_true: Dictionary of actual power values for each appliance
        y_pred: Dictionary of predicted power values for each appliance
        save_path: Path to save the plot
    """
    n_appliances = len(y_true)
    n_cols = min(3, n_appliances)
    n_rows = (n_appliances + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    if n_appliances == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (appliance, true_values) in enumerate(y_true.items()):
        pred_values = y_pred.get(appliance, np.zeros_like(true_values))
        
        # Filter out zeros for better visualization
        true_nonzero = true_values[true_values > 0]
        pred_nonzero = pred_values[pred_values > 0]
        
        axes[idx].hist(true_nonzero, bins=50, alpha=0.5, label='Actual', density=True)
        axes[idx].hist(pred_nonzero, bins=50, alpha=0.5, label='Predicted', density=True)
        axes[idx].set_xlabel('Power (W)')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{appliance} - Power Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_appliances, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Power distribution plot saved to {save_path}")


def save_predictions_to_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    appliance_name: str,
    save_dir: str
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        appliance_name: Name of the appliance
        save_dir: Directory to save the CSV
    """
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'absolute_error': np.abs(y_true - y_pred)
    })
    
    filepath = os.path.join(save_dir, f'{appliance_name}_predictions.csv')
    df.to_csv(filepath, index=False)
    logger.info(f"Predictions saved to {filepath}")
