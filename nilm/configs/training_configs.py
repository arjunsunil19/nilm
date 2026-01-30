"""
Training Configuration

Dataclass for training settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for training NILM models."""
    
    # Data settings
    window_size: int = 599
    stride: int = 1
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Batch settings
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimizer settings
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Scheduler settings
    scheduler: str = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Training settings
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    
    # Regularization
    dropout: float = 0.2
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 10
    verbose: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = False
    
    # Random seed
    seed: Optional[int] = 42
    
    def validate(self) -> None:
        """Validate configuration settings."""
        assert 0 < self.train_split < 1, "train_split must be between 0 and 1"
        assert 0 <= self.val_split < 1, "val_split must be between 0 and 1"
        assert 0 <= self.test_split < 1, "test_split must be between 0 and 1"
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Splits must sum to 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    
    # Experiment info
    experiment_name: str = "nilm_experiment"
    description: str = ""
    
    # Model configuration
    model_type: str = "bilstm"
    model_config: dict = field(default_factory=dict)
    
    # Training configuration
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data configuration
    data_path: Optional[str] = None
    appliance_names: List[str] = field(default_factory=list)
    
    # Output
    output_dir: str = "./experiments"
    save_predictions: bool = True
    save_model: bool = True
