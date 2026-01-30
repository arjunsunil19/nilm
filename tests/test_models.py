"""
Test NILM Models

Unit tests for all NILM model implementations.
"""

import pytest
import torch
import numpy as np

from nilm.models import BiLSTM, CNN, Seq2Point, Seq2Seq
from nilm.models.bilstm import BiLSTMWithConv
from nilm.models.cnn import DilatedCNN
from nilm.models.seq2point import Seq2PointWithAttention
from nilm.models.seq2seq import UNetSeq2Seq


class TestBiLSTM:
    """Tests for BiLSTM model."""
    
    def test_forward_pass_seq2point(self):
        """Test forward pass with seq2point output."""
        model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
            num_layers=1,
            output_type="seq2point",
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 99, 1)
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        
    def test_forward_pass_seq2seq(self):
        """Test forward pass with seq2seq output."""
        model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
            num_layers=1,
            output_type="seq2seq",
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 99, 1)
        output = model(x)
        
        assert output.shape == (batch_size, 99, 3)
        
    def test_2d_input(self):
        """Test with 2D input (no channel dimension)."""
        model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
        )
        
        x = torch.randn(8, 99)  # No channel dimension
        output = model(x)
        
        assert output.shape == (8, 3)
        
    def test_model_info(self):
        """Test get_model_info method."""
        model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
        )
        
        info = model.get_model_info()
        
        assert info["model_name"] == "BiLSTM"
        assert info["window_size"] == 99
        assert info["num_appliances"] == 3
        assert info["hidden_dim"] == 64
        assert info["num_parameters"] > 0


class TestBiLSTMWithConv:
    """Tests for BiLSTM with convolutional layers."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = BiLSTMWithConv(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
            conv_filters=32,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)


class TestCNN:
    """Tests for CNN model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = CNN(
            window_size=99,
            num_appliances=3,
            num_filters=[16, 32, 32],
            kernel_sizes=[5, 5, 5],
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)
        
    def test_default_config(self):
        """Test with default configuration."""
        model = CNN(
            window_size=99,
            num_appliances=3,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)


class TestDilatedCNN:
    """Tests for Dilated CNN model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DilatedCNN(
            window_size=99,
            num_appliances=3,
            num_filters=32,
            num_layers=4,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)


class TestSeq2Point:
    """Tests for Seq2Point model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = Seq2Point(
            window_size=99,
            num_appliances=3,
            num_filters=[16, 32, 32],
            kernel_sizes=[5, 5, 5],
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)
        
    def test_with_attention(self):
        """Test Seq2Point with attention."""
        model = Seq2PointWithAttention(
            window_size=99,
            num_appliances=3,
            num_filters=32,
            num_heads=4,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 3)


class TestSeq2Seq:
    """Tests for Seq2Seq model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = Seq2Seq(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
            num_layers=1,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 99, 3)
        
    def test_without_attention(self):
        """Test without attention mechanism."""
        model = Seq2Seq(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
            use_attention=False,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        assert output.shape == (8, 99, 3)


class TestUNetSeq2Seq:
    """Tests for U-Net style Seq2Seq model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = UNetSeq2Seq(
            window_size=96,  # Divisible by 8
            num_appliances=3,
            base_filters=16,
        )
        
        x = torch.randn(8, 96, 1)
        output = model(x)
        
        assert output.shape == (8, 96, 3)
        
    def test_non_divisible_size(self):
        """Test with window size not divisible by 8."""
        model = UNetSeq2Seq(
            window_size=99,  # Not divisible by 8
            num_appliances=3,
            base_filters=16,
        )
        
        x = torch.randn(8, 99, 1)
        output = model(x)
        
        # Should still output correct shape due to padding
        assert output.shape == (8, 99, 3)


class TestModelSaveLoad:
    """Tests for model save/load functionality."""
    
    def test_save_load_model(self, tmp_path):
        """Test saving and loading a model."""
        model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
        )
        
        # Set to eval mode for consistent output
        model.eval()
        
        # Get original output
        x = torch.randn(4, 99, 1)
        with torch.no_grad():
            original_output = model(x)
        
        # Save model
        save_path = tmp_path / "model.pt"
        model.save_model(str(save_path))
        
        # Create new model and load
        new_model = BiLSTM(
            window_size=99,
            num_appliances=3,
            hidden_dim=64,
        )
        new_model.load_model(str(save_path))
        new_model.eval()
        
        # Check outputs match
        with torch.no_grad():
            loaded_output = new_model(x)
        assert torch.allclose(original_output, loaded_output)


class TestGradients:
    """Tests for gradient flow."""
    
    @pytest.mark.parametrize("model_class,kwargs", [
        (BiLSTM, {"hidden_dim": 32, "num_layers": 1}),
        (CNN, {"num_filters": [16, 16], "kernel_sizes": [3, 3]}),
        (Seq2Point, {"num_filters": [16, 16], "kernel_sizes": [3, 3]}),
    ])
    def test_gradients_flow(self, model_class, kwargs):
        """Test that gradients flow through the model."""
        model = model_class(
            window_size=49,
            num_appliances=2,
            **kwargs,
        )
        
        x = torch.randn(4, 49, 1, requires_grad=True)
        y = torch.randn(4, 2)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
