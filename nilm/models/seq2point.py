"""
Sequence-to-Point Model for NILM

The Seq2Point model predicts the power consumption at the midpoint
of the input window, which has been shown to be effective for NILM.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from nilm.models.base import BaseNILMModel


class Seq2Point(BaseNILMModel):
    """
    Sequence-to-Point model for Non-Intrusive Load Monitoring.
    
    This model takes a window of aggregate power readings and predicts
    the power consumption at the center point of the window.
    
    Architecture follows the original Seq2Point paper with:
        - Multiple 1D convolutional layers
        - Dense layers for final prediction
        - Outputs single point prediction
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
        Initialize Seq2Point model.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        # Default architecture from original paper
        if num_filters is None:
            num_filters = [30, 30, 40, 50, 50]
        if kernel_sizes is None:
            kernel_sizes = [10, 8, 6, 5, 5]
            
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        
        # Build convolutional layers
        layers = []
        in_channels = input_dim
        
        for filters, kernel in zip(num_filters, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel, padding="same"),
                nn.ReLU(),
                nn.BatchNorm1d(filters),
            ])
            in_channels = filters
            
        self.conv = nn.Sequential(*layers)
        
        # Flatten and dense layers
        self.flatten = nn.Flatten()
        
        # Calculate flattened size
        self.flattened_size = num_filters[-1] * window_size
        
        self.dense = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_appliances)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Seq2Point.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
               or (batch_size, window_size)
               
        Returns:
            Output tensor of shape (batch_size, num_appliances)
            representing power at the center point
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # Reshape for conv1d
        x = x.permute(0, 2, 1)  # (batch, channels, seq)
        
        # Convolutional layers
        x = self.conv(x)
        
        # Flatten and dense
        x = self.flatten(x)
        output = self.dense(x)
        
        return output


class Seq2PointWithAttention(BaseNILMModel):
    """
    Seq2Point model enhanced with attention mechanism.
    
    Adds self-attention layers to better capture relationships
    between different parts of the input sequence.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        num_filters: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        **kwargs
    ):
        """
        Initialize Seq2Point with attention.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            num_filters: Number of filters in conv layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.num_filters = num_filters
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        # Convolutional feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(num_filters)
        
        # Point selection - focus on center
        self.point_attention = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.Tanh(),
            nn.Linear(num_filters // 2, 1),
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_appliances)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = x.permute(0, 2, 1)  # (batch, channels, seq)
        
        # Conv features
        x = self.conv(x)  # (batch, filters, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, filters)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)
        
        # Point attention to focus on relevant timesteps
        point_weights = torch.softmax(self.point_attention(x), dim=1)
        x = torch.sum(point_weights * x, dim=1)  # (batch, filters)
        
        # Output
        output = self.fc(x)
        
        return output
