"""
Test NILM Utilities

Unit tests for training, evaluation, and metrics.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from nilm.models import BiLSTM
from nilm.data import SyntheticNILMDataset
from nilm.utils import Trainer, Evaluator
from nilm.utils.metrics import NILMMetrics


class TestNILMMetrics:
    """Tests for NILMMetrics class."""
    
    def test_mae(self):
        """Test Mean Absolute Error."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        
        mae = NILMMetrics.mae(predictions, targets)
        assert mae == 0.5
        
    def test_mse(self):
        """Test Mean Squared Error."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        
        mse = NILMMetrics.mse(predictions, targets)
        assert mse == 1.0
        
    def test_rmse(self):
        """Test Root Mean Squared Error."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        
        rmse = NILMMetrics.rmse(predictions, targets)
        assert rmse == 1.0
        
    def test_signal_aggregate_error(self):
        """Test Signal Aggregate Error."""
        predictions = np.array([100, 200, 300])
        targets = np.array([100, 200, 400])  # Sum = 700
        
        sae = NILMMetrics.signal_aggregate_error(predictions, targets)
        # |600 - 700| / 700 = 100/700 ≈ 0.143
        np.testing.assert_allclose(sae, 100/700, rtol=1e-5)
        
    def test_sae_with_zero_targets(self):
        """Test SAE when targets sum to zero."""
        predictions = np.array([0, 0, 1])
        targets = np.array([0, 0, 0])
        
        sae = NILMMetrics.signal_aggregate_error(predictions, targets)
        assert np.isnan(sae)
        
    def test_normalized_disaggregation_error(self):
        """Test Normalized Disaggregation Error."""
        predictions = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        
        nde = NILMMetrics.normalized_disaggregation_error(predictions, targets)
        assert nde == 0.0
        
    def test_energy_accuracy(self):
        """Test Energy Accuracy."""
        predictions = np.array([100, 200, 300])
        targets = np.array([100, 200, 300])
        
        eac = NILMMetrics.energy_accuracy(predictions, targets)
        assert eac == 1.0
        
    def test_f1_score(self):
        """Test F1 Score."""
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 0])
        
        # TP=1, FP=1, FN=0, TN=2
        # Precision = 1/2 = 0.5
        # Recall = 1/1 = 1.0
        # F1 = 2 * 0.5 * 1.0 / 1.5 = 0.667
        f1 = NILMMetrics.f1_score(predictions, targets)
        np.testing.assert_allclose(f1, 2/3, rtol=1e-5)
        
    def test_precision(self):
        """Test Precision."""
        predictions = np.array([1, 1, 1, 0])
        targets = np.array([1, 1, 0, 0])
        
        # TP=2, FP=1 -> Precision = 2/3
        precision = NILMMetrics.precision(predictions, targets)
        np.testing.assert_allclose(precision, 2/3, rtol=1e-5)
        
    def test_recall(self):
        """Test Recall."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 1, 1, 0])
        
        # TP=2, FN=1 -> Recall = 2/3
        recall = NILMMetrics.recall(predictions, targets)
        np.testing.assert_allclose(recall, 2/3, rtol=1e-5)
        
    def test_accuracy(self):
        """Test Accuracy."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 1, 1, 0])
        
        # 3 correct out of 4
        accuracy = NILMMetrics.accuracy(predictions, targets)
        assert accuracy == 0.75
        
    def test_mcc(self):
        """Test Matthews Correlation Coefficient."""
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 0])
        
        mcc = NILMMetrics.matthews_correlation_coefficient(predictions, targets)
        # Should be between -1 and 1
        assert -1 <= mcc <= 1
        
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        predictions = np.random.rand(100)
        targets = np.random.rand(100)
        
        results = NILMMetrics.compute_all_metrics(predictions, targets, threshold=0.5)
        
        # Check that all expected keys are present
        expected_keys = ["mae", "mse", "rmse", "sae", "nde", "eac", "nep",
                        "f1", "precision", "recall", "accuracy", "mcc"]
        for key in expected_keys:
            assert key in results


class TestTrainer:
    """Tests for Trainer class."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup for training tests."""
        dataset = SyntheticNILMDataset(
            n_samples=500,
            n_appliances=2,
            window_size=49,
            seed=42,
        )
        
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        model = BiLSTM(
            window_size=49,
            num_appliances=2,
            hidden_dim=32,
            num_layers=1,
        )
        
        return model, train_loader, val_loader
    
    def test_trainer_creation(self, setup_training):
        """Test trainer creation."""
        model, train_loader, val_loader = setup_training
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        
    def test_train_epoch(self, setup_training):
        """Test single training epoch."""
        model, train_loader, val_loader = setup_training
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            device="cpu",
        )
        
        loss = trainer.train_epoch()
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        
    def test_validate(self, setup_training):
        """Test validation."""
        model, train_loader, val_loader = setup_training
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
        )
        
        val_loss = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert not np.isnan(val_loss)
        
    def test_train_multiple_epochs(self, setup_training):
        """Test training for multiple epochs."""
        model, train_loader, val_loader = setup_training
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
            early_stopping_patience=5,
        )
        
        history = trainer.train(epochs=2, verbose=False)
        
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        
    def test_checkpoint_save_load(self, setup_training, tmp_path):
        """Test checkpoint saving and loading."""
        model, train_loader, val_loader = setup_training
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
            checkpoint_dir=str(tmp_path),
        )
        
        # Train a bit
        trainer.train(epochs=1, verbose=False)
        
        # Save checkpoint
        path = trainer.save_checkpoint("test_checkpoint.pt")
        assert (tmp_path / "test_checkpoint.pt").exists()
        
        # Load checkpoint
        trainer.load_checkpoint(path)
        assert trainer.current_epoch == 1


class TestEvaluator:
    """Tests for Evaluator class."""
    
    @pytest.fixture
    def setup_evaluation(self):
        """Setup for evaluation tests."""
        dataset = SyntheticNILMDataset(
            n_samples=200,
            n_appliances=2,
            window_size=49,
            seed=42,
        )
        
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        model = BiLSTM(
            window_size=49,
            num_appliances=2,
            hidden_dim=32,
            num_layers=1,
        )
        
        return model, test_loader
        
    def test_predict(self, setup_evaluation):
        """Test prediction generation."""
        model, test_loader = setup_evaluation
        
        evaluator = Evaluator(
            model=model,
            device="cpu",
            appliance_names=["App1", "App2"],
        )
        
        predictions, targets = evaluator.predict(test_loader)
        
        assert predictions.shape[0] == len(test_loader.dataset)
        assert predictions.shape[1] == 2
        assert targets is not None
        
    def test_evaluate(self, setup_evaluation):
        """Test full evaluation."""
        model, test_loader = setup_evaluation
        
        evaluator = Evaluator(
            model=model,
            device="cpu",
            appliance_names=["App1", "App2"],
        )
        
        results = evaluator.evaluate(test_loader, verbose=False)
        
        assert "mae" in results
        assert "rmse" in results
        assert "sae" in results
        assert "nde" in results
