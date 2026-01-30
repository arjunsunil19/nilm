"""
Model Configuration Classes

Dataclasses for configuring NILM models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BiLSTMConfig:
    """Configuration for BiLSTM model."""
    
    # Model architecture
    window_size: int = 599
    num_appliances: int = 1
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    output_type: str = "seq2point"  # "seq2point" or "seq2seq"
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "num_appliances": self.num_appliances,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_type": self.output_type,
        }


@dataclass
class CNNConfig:
    """Configuration for CNN model."""
    
    # Model architecture
    window_size: int = 599
    num_appliances: int = 1
    input_dim: int = 1
    num_filters: List[int] = field(default_factory=lambda: [30, 30, 40, 50, 50])
    kernel_sizes: List[int] = field(default_factory=lambda: [10, 8, 6, 5, 5])
    dropout: float = 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "num_appliances": self.num_appliances,
            "input_dim": self.input_dim,
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "dropout": self.dropout,
        }


@dataclass
class Seq2PointConfig:
    """Configuration for Seq2Point model."""
    
    # Model architecture
    window_size: int = 599
    num_appliances: int = 1
    input_dim: int = 1
    num_filters: List[int] = field(default_factory=lambda: [30, 30, 40, 50, 50])
    kernel_sizes: List[int] = field(default_factory=lambda: [10, 8, 6, 5, 5])
    dropout: float = 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "num_appliances": self.num_appliances,
            "input_dim": self.input_dim,
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "dropout": self.dropout,
        }


@dataclass
class Seq2SeqConfig:
    """Configuration for Seq2Seq model."""
    
    # Model architecture
    window_size: int = 599
    num_appliances: int = 1
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    use_attention: bool = True
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "num_appliances": self.num_appliances,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
        }


def get_default_config(model_type: str) -> Any:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: Type of model ("bilstm", "cnn", "seq2point", "seq2seq")
        
    Returns:
        Configuration dataclass
    """
    configs = {
        "bilstm": BiLSTMConfig,
        "cnn": CNNConfig,
        "seq2point": Seq2PointConfig,
        "seq2seq": Seq2SeqConfig,
    }
    
    model_type = model_type.lower()
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(configs.keys())}")
    
    return configs[model_type]()
