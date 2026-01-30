"""
Sequence-to-Sequence Model for NILM

The Seq2Seq model outputs a sequence of predictions matching
the input sequence length, enabling dense predictions.
"""

from typing import Optional

import torch
import torch.nn as nn

from nilm.models.base import BaseNILMModel


class Seq2Seq(BaseNILMModel):
    """
    Sequence-to-Sequence model for Non-Intrusive Load Monitoring.
    
    This model outputs power consumption predictions for each timestep
    in the input sequence, using an encoder-decoder architecture.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        **kwargs
    ):
        """
        Initialize Seq2Seq model.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            hidden_dim: Hidden dimension for encoder/decoder
            num_layers: Number of layers in encoder/decoder
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_attention = use_attention
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        decoder_input_dim = hidden_dim * 2 if use_attention else hidden_dim * 2
        self.decoder = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_appliances)
        )
        
        # Initial decoder hidden state projection
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Seq2Seq.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
               or (batch_size, window_size)
               
        Returns:
            Output tensor of shape (batch_size, window_size, num_appliances)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs: (batch, seq, hidden*2)
        
        # Reshape hidden and cell states for decoder
        # Combine forward and backward states
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        hidden = self.hidden_projection(hidden)  # (layers, batch, hidden)
        
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        cell = self.cell_projection(cell)  # (layers, batch, hidden)
        
        # Decode
        outputs = []
        decoder_input = encoder_outputs[:, :1, :]  # Start with first encoder output
        
        for t in range(seq_len):
            if self.use_attention:
                # Compute attention weights
                decoder_hidden = hidden[-1].unsqueeze(1)  # (batch, 1, hidden)
                decoder_hidden_expanded = decoder_hidden.expand(-1, seq_len, -1)
                
                attention_input = torch.cat([
                    encoder_outputs, 
                    decoder_hidden_expanded
                ], dim=-1)  # (batch, seq, hidden*3)
                
                attention_scores = self.attention(attention_input)  # (batch, seq, 1)
                attention_weights = torch.softmax(attention_scores, dim=1)
                
                context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
                decoder_input = context  # (batch, 1, hidden*2)
            else:
                decoder_input = encoder_outputs[:, t:t+1, :]
            
            # Decoder step
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Project to output
            output = self.output_layer(decoder_output)  # (batch, 1, num_appliances)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, seq, num_appliances)
        
        return outputs


class UNetSeq2Seq(BaseNILMModel):
    """
    U-Net style Seq2Seq model for NILM.
    
    Uses skip connections between encoder and decoder layers
    for better gradient flow and feature preservation.
    """
    
    def __init__(
        self,
        window_size: int,
        num_appliances: int,
        input_dim: int = 1,
        base_filters: int = 32,
        dropout: float = 0.2,
        **kwargs
    ):
        """
        Initialize U-Net Seq2Seq model.
        
        Args:
            window_size: Size of input window
            num_appliances: Number of appliances to disaggregate
            input_dim: Number of input features
            base_filters: Base number of filters (doubles at each level)
            dropout: Dropout rate
        """
        super().__init__(window_size, num_appliances, input_dim)
        
        self.base_filters = base_filters
        self.dropout_rate = dropout
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_dim, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose1d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        # Output
        self.output_conv = nn.Conv1d(base_filters, num_appliances, 1)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = x.permute(0, 2, 1)  # (batch, channels, seq)
        
        # Pad to make divisible by 8
        orig_len = x.size(2)
        pad_len = (8 - orig_len % 8) % 8
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len))
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = self.output_conv(d1)  # (batch, appliances, seq)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :, :orig_len]
        
        output = output.permute(0, 2, 1)  # (batch, seq, appliances)
        
        return output
