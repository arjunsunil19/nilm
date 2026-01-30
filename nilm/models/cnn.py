"""
CNN Model for NILM

Convolutional Neural Network architecture for energy disaggregation.
Inspired by the architecture proposed in Neural NILM paper.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from nilm.models.base import BaseNILMModel


class CNN(BaseNILMModel):
    """
    Convolutional Neural Network for Non-Intrusive Load Monitoring.
    
    Architecture based on Neural NILM paper with modifications:
        - Multiple 1D convolutional layers
        - Batch normalization
        - Dropout for regularization
        - Fully connected output layers
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.2,
        **kwargs
    ):
        """
        Initialize CNN model.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        if num_filters is None:
            num_filters = [30, 30, 40, 50, 50]
        if kernel_sizes is None:
            kernel_sizes = [10, 8, 6, 5, 5]
            
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        
        # Build convolutional layers with proper "same" padding
        conv_layers = []
        in_channels = input_dim
        
        for i, (filters, kernel) in enumerate(zip(num_filters, kernel_sizes)):
            # Calculate padding for "same" output
            padding = kernel // 2
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(filters),
            ])
            if i < len(num_filters) - 1:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = filters
            
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate feature size after convolutions
        # With proper "same" padding, sequence length is approximately preserved
        # But some kernels cause slight changes, so we compute dynamically
        self._compute_feature_size(window_size, input_dim, num_filters, kernel_sizes)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_appliances)
        )
    
    def _compute_feature_size(self, window_size, input_dim, num_filters, kernel_sizes):
        """Compute the feature size after conv layers."""
        # Simulate forward pass to get actual size
        seq_len = window_size
        for kernel in kernel_sizes:
            padding = kernel // 2
            # output_length = (input_length + 2*padding - kernel) // stride + 1
            seq_len = (seq_len + 2 * padding - kernel) // 1 + 1
        
        self.feature_size = num_filters[-1] * seq_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
               or (batch_size, window_size)
               
        Returns:
            Output tensor of shape (batch_size, num_appliances)
        """
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # Reshape for conv1d: (batch, channels, seq)
        x = x.permute(0, 2, 1)
        
        # Convolutional layers
        x = self.conv(x)  # (batch, filters, seq)
        
        # Fully connected layers
        output = self.fc(x)
        
        return output


class DilatedCNN(BaseNILMModel):
    """
    Dilated CNN for NILM with larger receptive field.
    
    Uses dilated convolutions to capture long-range dependencies
    without increasing the number of parameters.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 8,
        dropout: float = 0.2,
        **kwargs
    ):
        """
        Initialize Dilated CNN model.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            num_filters: Number of filters in each layer
            kernel_size: Kernel size for convolutions
            num_layers: Number of dilated conv layers
            dropout: Dropout rate
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters)
        )
        
        # Dilated convolutional layers with residual connections
        self.dilated_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            self.dilated_layers.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size, 
                         dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.Dropout(dropout)
            ))
        
        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_appliances)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = x.permute(0, 2, 1)  # (batch, channels, seq)
        
        # Initial projection
        x = self.input_conv(x)
        
        # Dilated layers with residual connections
        for layer in self.dilated_layers:
            residual = x
            x = layer(x) + residual
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, filters)
        
        # Output
        output = self.fc(x)
        
        return output
