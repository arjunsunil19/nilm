"""
Test NILM Data Module

Unit tests for datasets and preprocessing.
"""

import pytest
import numpy as np
import torch

from nilm.data import NILMDataset, DataPreprocessor
from nilm.data.dataset import SyntheticNILMDataset, NILMDatasetFromDict


class TestNILMDataset:
    """Tests for NILMDataset class."""
    
    def test_basic_creation(self):
        """Test basic dataset creation."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        appliances = np.random.randn(1000, 3).astype(np.float32) + 100
        
        dataset = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=99,
            stride=1,
        )
        
        assert len(dataset) > 0
        
    def test_getitem_returns_correct_shapes(self):
        """Test that __getitem__ returns correct shapes."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        appliances = np.random.randn(1000, 3).astype(np.float32) + 100
        
        dataset = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=99,
            stride=1,
        )
        
        x, y = dataset[0]
        
        assert x.shape == (99, 1)
        assert y.shape == (3,)
        
    def test_seq2seq_output(self):
        """Test seq2seq output type."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        appliances = np.random.randn(1000, 3).astype(np.float32) + 100
        
        dataset = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=99,
            stride=1,
            output_type="seq2seq",
        )
        
        x, y = dataset[0]
        
        assert x.shape == (99, 1)
        assert y.shape == (99, 3)
        
    def test_stride(self):
        """Test different stride values."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        appliances = np.random.randn(1000, 3).astype(np.float32) + 100
        
        dataset_s1 = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=99,
            stride=1,
        )
        
        dataset_s10 = NILMDataset(
            aggregate=aggregate,
            appliances=appliances,
            window_size=99,
            stride=10,
        )
        
        # Larger stride should result in fewer samples
        assert len(dataset_s10) < len(dataset_s1)
        
    def test_normalization(self):
        """Test normalization."""
        aggregate = np.random.randn(1000).astype(np.float32) * 100 + 500
        
        dataset = NILMDataset(
            aggregate=aggregate,
            window_size=99,
            normalize=True,
        )
        
        # Check that normalization stats are computed
        stats = dataset.get_stats()
        assert "mean" in stats
        assert "std" in stats
        
    def test_no_appliances(self):
        """Test dataset without appliance data (inference mode)."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        
        dataset = NILMDataset(
            aggregate=aggregate,
            appliances=None,
            window_size=99,
        )
        
        x = dataset[0]
        assert x.shape == (99, 1)


class TestSyntheticNILMDataset:
    """Tests for SyntheticNILMDataset class."""
    
    def test_creation(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticNILMDataset(
            n_samples=1000,
            n_appliances=3,
            window_size=99,
            seed=42,
        )
        
        assert len(dataset) > 0
        
    def test_reproducibility(self):
        """Test that seed produces reproducible data."""
        dataset1 = SyntheticNILMDataset(
            n_samples=100,
            n_appliances=2,
            window_size=49,
            seed=42,
        )
        
        dataset2 = SyntheticNILMDataset(
            n_samples=100,
            n_appliances=2,
            window_size=49,
            seed=42,
        )
        
        x1, y1 = dataset1[0]
        x2, y2 = dataset2[0]
        
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)
        
    def test_aggregate_is_sum_of_appliances(self):
        """Test that aggregate is approximately sum of appliances."""
        dataset = SyntheticNILMDataset(
            n_samples=1000,
            n_appliances=3,
            window_size=99,
            noise_level=0.0,  # No noise
            seed=42,
        )
        
        # Without noise, aggregate should equal sum of appliances
        agg_sum = np.sum(dataset.aggregate)
        app_sum = np.sum(dataset.appliances)
        
        # Should be equal (no noise)
        np.testing.assert_allclose(agg_sum, app_sum, rtol=1e-5)


class TestNILMDatasetFromDict:
    """Tests for NILMDatasetFromDict class."""
    
    def test_creation_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            "aggregate": np.random.randn(1000).astype(np.float32) + 500,
            "fridge": np.random.randn(1000).astype(np.float32) + 100,
            "microwave": np.random.randn(1000).astype(np.float32) + 50,
        }
        
        dataset = NILMDatasetFromDict(
            data_dict=data_dict,
            aggregate_key="aggregate",
            appliance_keys=["fridge", "microwave"],
            window_size=99,
        )
        
        assert len(dataset) > 0
        
        x, y = dataset[0]
        assert x.shape == (99, 1)
        assert y.shape == (2,)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        aggregate = np.random.randn(1000).astype(np.float32) * 100 + 500
        appliances = np.random.randn(1000, 3).astype(np.float32) * 50 + 100
        
        preprocessor = DataPreprocessor(
            window_size=99,
            normalize=True,
        )
        
        agg_transformed, app_transformed = preprocessor.fit_transform(aggregate, appliances)
        
        # Check shapes preserved
        assert agg_transformed.shape == aggregate.shape
        assert app_transformed.shape == appliances.shape
        
        # Check normalization (aggregate should have mean ~0)
        assert abs(np.mean(agg_transformed)) < 0.5
        
    def test_inverse_transform(self):
        """Test inverse_transform method."""
        aggregate = np.random.randn(1000).astype(np.float32) * 100 + 500
        
        preprocessor = DataPreprocessor(
            normalize=True,
            clip_outliers=False,  # Disable clipping for exact reconstruction
        )
        
        transformed = preprocessor.fit_transform(aggregate)
        reconstructed = preprocessor.inverse_transform(transformed)
        
        np.testing.assert_allclose(aggregate, reconstructed, rtol=1e-5)
        
    def test_fill_nan_interpolate(self):
        """Test NaN filling with interpolation."""
        aggregate = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
        
        preprocessor = DataPreprocessor(
            fill_nan_method="interpolate",
            normalize=False,
        )
        
        transformed = preprocessor.fit_transform(aggregate)
        
        # NaN should be filled
        assert not np.any(np.isnan(transformed))
        # Interpolated value should be 3
        np.testing.assert_allclose(transformed[2], 3.0, rtol=1e-5)
        
    def test_fill_nan_zero(self):
        """Test NaN filling with zeros."""
        aggregate = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
        
        preprocessor = DataPreprocessor(
            fill_nan_method="zero",
            normalize=False,
        )
        
        transformed = preprocessor.fit_transform(aggregate)
        
        assert not np.any(np.isnan(transformed))
        assert transformed[2] == 0.0
        
    def test_clip_outliers(self):
        """Test outlier clipping."""
        aggregate = np.array([100, 200, 10000, 300, 400], dtype=np.float32)
        
        preprocessor = DataPreprocessor(
            clip_outliers=True,
            outlier_percentile=90,
            normalize=False,
        )
        
        transformed = preprocessor.fit_transform(aggregate)
        
        # The outlier (10000) should be clipped
        assert np.max(transformed) < 10000
        
    def test_create_windows(self):
        """Test window creation."""
        aggregate = np.arange(100, dtype=np.float32)
        appliances = np.arange(100, dtype=np.float32).reshape(-1, 1)
        
        preprocessor = DataPreprocessor(
            window_size=10,
            stride=5,
            normalize=False,
        )
        
        agg_windows, targets = preprocessor.create_windows(
            aggregate, appliances, output_type="seq2point"
        )
        
        # Check shapes
        expected_n_windows = (100 - 10) // 5 + 1
        assert agg_windows.shape == (expected_n_windows, 10)
        assert targets.shape == (expected_n_windows, 1)
        
    def test_save_load_stats(self, tmp_path):
        """Test saving and loading stats."""
        aggregate = np.random.randn(1000).astype(np.float32) + 500
        
        preprocessor = DataPreprocessor(normalize=True)
        preprocessor.fit(aggregate)
        
        # Save stats
        stats_path = tmp_path / "stats.json"
        preprocessor.save_stats(str(stats_path))
        
        # Load stats in new preprocessor
        new_preprocessor = DataPreprocessor(normalize=True)
        new_preprocessor.load_stats(str(stats_path))
        
        assert new_preprocessor.stats == preprocessor.stats
