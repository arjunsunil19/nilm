"""
BiLSTM Model for NILM

Bidirectional LSTM architecture for energy disaggregation.
This model processes the input sequence in both forward and backward
directions, capturing temporal dependencies from both past and future context.
"""

from typing import Optional

import torch
import torch.nn as nn

from nilm.models.base import BaseNILMModel


class BiLSTM(BaseNILMModel):
    """
    Bidirectional LSTM model for Non-Intrusive Load Monitoring.
    
    Architecture:
        - Input layer with optional linear projection
        - Multiple bidirectional LSTM layers with dropout
        - Fully connected layers for output
        
    The model takes aggregate power readings and outputs
    individual appliance power consumption.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_type: str = "seq2point",
        **kwargs
    ):
        """
        Initialize BiLSTM model.
        
        Args:
            window_size: Size of input window (sequence length)
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features (default: 1)
            hidden_dim: Hidden dimension of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate between layers
            output_type: Type of output - "seq2point" (single output) or 
                        "seq2seq" (sequence output)
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_type = output_type
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        if output_type == "seq2point":
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_appliances)
            )
        else:  # seq2seq
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_appliances)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BiLSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
               or (batch_size, window_size) if input_dim=1
               
        Returns:
            If output_type="seq2point": 
                Tensor of shape (batch_size, num_appliances)
            If output_type="seq2seq":
                Tensor of shape (batch_size, window_size, num_appliances)
        """
        # Ensure input has 3 dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Ensure float32 dtype
        x = x.float()
            
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq, hidden)
        
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        if self.output_type == "seq2point":
            # Apply attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
            
            # Apply dropout
            context = self.dropout_layer(context)
            
            # Output layer
            output = self.output_layer(context)  # (batch, num_appliances)
        else:
            # Seq2Seq: apply output layer to each timestep
            output = self.output_layer(lstm_out)  # (batch, seq, num_appliances)
        
        return output
    
    def get_model_info(self):
        """Get model configuration info."""
        info = super().get_model_info()
        info.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_type": self.output_type,
        })
        return info


class BiLSTMWithConv(BaseNILMModel):
    """
    BiLSTM model with convolutional feature extraction.
    
    This variant adds 1D convolutional layers before the BiLSTM
    to extract local features from the input signal.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_filters: int = 32,
        kernel_size: int = 5,
        **kwargs
    ):
        """
        Initialize BiLSTM with convolutional layers.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            hidden_dim: Hidden dimension of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            conv_filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, conv_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters * 2),
            nn.Dropout(dropout)
        )
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_filters * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_appliances)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_appliances)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Ensure float32 dtype
        x = x.float()
            
        # Reshape for conv1d: (batch, input_dim, seq)
        x = x.permute(0, 2, 1)
        
        # Convolutional feature extraction
        x = self.conv_layers(x)  # (batch, filters*2, seq)
        
        # Reshape back for LSTM: (batch, seq, filters*2)
        x = x.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Take the output from the middle of the sequence
        mid_idx = lstm_out.size(1) // 2
        mid_output = lstm_out[:, mid_idx, :]
        
        # Output
        output = self.fc(mid_output)
        
        return output
