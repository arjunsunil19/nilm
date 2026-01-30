"""
NILM Model Trainer

Comprehensive training utilities for NILM models.
"""

import os
import time
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from nilm.models.base import BaseNILMModel


class Trainer:
    """
    Trainer class for NILM models.
    
    Handles the training loop, validation, checkpointing,
    and learning rate scheduling.
    """
    
    def __init__(
        self,
        model: BaseNILMModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: Optional[str] = "plateau",
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        gradient_clip: Optional[float] = 1.0,
    ):
        """
        Initialize trainer.
        
        Args:
            model: NILM model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer type ("adam", "adamw", "sgd")
            lr: Learning rate
            weight_decay: Weight decay for regularization
            scheduler: LR scheduler type ("plateau", "cosine", "step", None)
            criterion: Loss function (default: MSELoss)
            device: Device to use ("cuda", "cpu", or None for auto)
            checkpoint_dir: Directory for saving checkpoints
            early_stopping_patience: Patience for early stopping
            gradient_clip: Max gradient norm for clipping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler(scheduler)
        
        # Loss function
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # Training settings
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip = gradient_clip
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        
        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
    def _create_optimizer(
        self, 
        optimizer_type: str, 
        lr: float, 
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
        if optimizer_type.lower() == "adam":
            return Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "adamw":
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
            
    def _create_scheduler(
        self, 
        scheduler_type: Optional[str]
    ) -> Optional[Any]:
        """Create learning rate scheduler."""
        if scheduler_type is None:
            return None
            
        if scheduler_type.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5,
            )
        elif scheduler_type.lower() == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=50)
        elif scheduler_type.lower() == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
            
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return float("nan")
            
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches
    
    def train(
        self,
        epochs: int,
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            verbose: Whether to print progress
            callback: Optional callback function called after each epoch
            
        Returns:
            Training history
        """
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch += 1
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                if self.checkpoint_dir:
                    self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if verbose:
                print(f"Epoch {self.current_epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.6f} - "
                      f"Val Loss: {val_loss:.6f} - "
                      f"LR: {current_lr:.2e} - "
                      f"Time: {epoch_time:.1f}s")
            
            # Callback
            if callback is not None:
                callback(self, epoch, train_loss, val_loss)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {self.current_epoch}")
                break
                
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            
        return self.history
    
    def save_checkpoint(self, filename: str) -> str:
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            
        Returns:
            Path to saved checkpoint
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set")
            
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_info": self.model.get_model_info(),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
    def get_model(self) -> BaseNILMModel:
        """Get the trained model."""
        return self.model


class MultiApplianceTrainer(Trainer):
    """
    Trainer specialized for multi-appliance NILM.
    
    Extends base trainer with per-appliance metrics and
    weighted loss computation.
    """
    
    def __init__(
        self,
        model: BaseNILMModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        appliance_names: Optional[List[str]] = None,
        appliance_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize multi-appliance trainer.
        
        Args:
            model: NILM model
            train_loader: Training data loader
            val_loader: Validation data loader
            appliance_names: Names of appliances
            appliance_weights: Weight for each appliance in loss
            **kwargs: Additional arguments for base Trainer
        """
        super().__init__(model, train_loader, val_loader, **kwargs)
        
        self.appliance_names = appliance_names or []
        self.appliance_weights = appliance_weights
        
        # Per-appliance history
        for name in self.appliance_names:
            self.history[f"val_loss_{name}"] = []
            
    def validate(self) -> float:
        """Validate with per-appliance metrics."""
        if self.val_loader is None:
            return float("nan")
            
        self.model.eval()
        total_loss = 0.0
        per_appliance_loss = [0.0] * len(self.appliance_names)
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                
                # Per-appliance loss
                for i in range(len(self.appliance_names)):
                    app_loss = nn.functional.mse_loss(outputs[:, i], y[:, i])
                    per_appliance_loss[i] += app_loss.item()
                    
                n_batches += 1
                
        # Update per-appliance history
        for i, name in enumerate(self.appliance_names):
            self.history[f"val_loss_{name}"].append(per_appliance_loss[i] / n_batches)
            
        return total_loss / n_batches
