"""
Base NILM Model

Abstract base class for all NILM models providing common interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class BaseNILMModel(nn.Module, ABC):
    """
    Abstract base class for NILM models.
    
    All NILM models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        **kwargs
    ):
        """
        Initialize the base NILM model.
        
        Args:
            window_size: Size of the input window (sequence length)
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features (default: 1 for aggregate power)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.window_size = window_size
        self.num_appliances = num_appliances
        self.input_dim = input_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            Output tensor with disaggregated power for each appliance
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.__class__.__name__,
            "window_size": self.window_size,
            "num_appliances": self.num_appliances,
            "input_dim": self.input_dim,
            "num_parameters": self.count_parameters(),
        }
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path: str) -> None:
        """
        Save model state to file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_info": self.get_model_info(),
        }, path)
        
    def load_model(self, path: str) -> None:
        """
        Load model state from file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["model_state_dict"])
