#!/usr/bin/env python3
"""Pytest tests for connectivity analysis."""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import pytest
import numpy as np
import mne
from pathlib import Path

from run_connectivity import (
    load_epoch_data,
    compute_connectivity,
    _permute_pdc,
)


# Test data paths
BIDS_ROOT = "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(bipolar)"
TEST_SUBJECT = "D0019"
TEST_DESCRIPTION = "perception"
TEST_BAND = "highgamma"
TEST_DATATYPE = "epoch(band)(sig)(effective)"


class TestLoadEpochData:
    """Tests for load_epoch_data function."""
    
    def test_load_epoch_data_returns_epochs(self):
        """Test that load_epoch_data returns MNE Epochs object."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description=TEST_DESCRIPTION,
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        
        assert isinstance(epochs, mne.BaseEpochs), "Should return MNE Epochs object"
        
    def test_load_epoch_data_has_correct_shape(self):
        """Test that loaded data has expected dimensions."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description=TEST_DESCRIPTION,
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        
        data = epochs.get_data()
        assert data.ndim == 3, "Data should be 3D (n_epochs, n_channels, n_times)"
        assert data.shape[0] > 0, "Should have at least one epoch"
        assert data.shape[1] > 0, "Should have at least one channel"
        assert data.shape[2] > 0, "Should have at least one time point"
        
    def test_load_epoch_data_has_channel_names(self):
        """Test that epochs have channel names."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description=TEST_DESCRIPTION,
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        
        assert len(epochs.ch_names) > 0, "Should have channel names"
        print(f"Channel names: {epochs.ch_names}")
        
    def test_load_epoch_data_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent subject."""
        with pytest.raises(FileNotFoundError):
            load_epoch_data(
                bids_root=BIDS_ROOT,
                subject="NONEXISTENT",
                description=TEST_DESCRIPTION,
                band=TEST_BAND,
                datatype=TEST_DATATYPE,
            )


class TestPermutePDC:
    """Tests for _permute_pdc function."""
    
    def test_permute_pdc_output_shape(self):
        """Test that _permute_pdc returns correct shape."""
        n_epochs, n_channels, n_times = 10, 5, 64
        segment = np.random.randn(n_epochs, n_channels, n_times)
        
        pdc = _permute_pdc(seed=42, segment=segment, p=4, nfft=32)
        
        assert pdc.shape == (n_channels, n_channels), \
            f"PDC shape should be ({n_channels}, {n_channels}), got {pdc.shape}"
            
    def test_permute_pdc_reproducibility(self):
        """Test that same seed produces same result."""
        n_epochs, n_channels, n_times = 10, 5, 64
        segment = np.random.randn(n_epochs, n_channels, n_times)
        
        pdc1 = _permute_pdc(seed=42, segment=segment, p=4, nfft=32)
        pdc2 = _permute_pdc(seed=42, segment=segment, p=4, nfft=32)
        
        np.testing.assert_array_almost_equal(pdc1, pdc2, 
            err_msg="Same seed should produce same PDC")


class TestComputeConnectivity:
    """Tests for compute_connectivity function."""
    
    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs for testing."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description=TEST_DESCRIPTION,
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        return epochs
    
    def test_compute_connectivity_returns_dict(self, sample_epochs):
        """Test that compute_connectivity returns a dictionary."""
        results = compute_connectivity(
            epochs=sample_epochs,
            window=0.5,
            step=0.5,  # Large step for faster test
            model_order=4,
            n_permutations=2,  # Few permutations for faster test
            n_jobs=1,
        )
        
        assert isinstance(results, dict), "Should return dictionary"
        
    def test_compute_connectivity_has_required_keys(self, sample_epochs):
        """Test that results contain all required keys."""
        results = compute_connectivity(
            epochs=sample_epochs,
            window=0.5,
            step=0.5,
            model_order=4,
            n_permutations=2,
            n_jobs=1,
        )
        
        required_keys = [
            'coef', 'rescov', 'pdc', 'pvals', 'null',
            'start_samples', 'end_samples', 'time_points',
            'fs', 'tmin', 'tmax', 'window', 'step',
            'model_order', 'nfft', 'ch_names'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"
            
    def test_compute_connectivity_pdc_shape(self, sample_epochs):
        """Test that PDC has correct shape."""
        results = compute_connectivity(
            epochs=sample_epochs,
            window=0.5,
            step=0.5,
            model_order=4,
            n_permutations=2,
            n_jobs=1,
        )
        
        n_channels = len(sample_epochs.ch_names)
        n_windows = len(results['time_points'])
        
        assert results['pdc'].shape == (n_windows, n_channels, n_channels), \
            f"PDC shape mismatch"
            
    def test_compute_connectivity_pvals_range(self, sample_epochs):
        """Test that p-values are in valid range [0, 1]."""
        results = compute_connectivity(
            epochs=sample_epochs,
            window=0.5,
            step=0.5,
            model_order=4,
            n_permutations=2,
            n_jobs=1,
        )
        
        assert np.all(results['pvals'] >= 0), "P-values should be >= 0"
        assert np.all(results['pvals'] <= 1), "P-values should be <= 1"


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_perception(self):
        """Test full pipeline for perception condition."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description="perception",
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        
        results = compute_connectivity(
            epochs=epochs,
            window=0.5,
            step=0.5,
            model_order=4,
            n_permutations=2,
            n_jobs=1,
        )
        
        # Basic sanity checks
        assert results['pdc'].shape[0] > 0, "Should have at least one time window"
        print(f"Perception: {results['pdc'].shape[0]} windows, "
              f"{len(results['ch_names'])} channels")
        
    def test_full_pipeline_production(self):
        """Test full pipeline for production condition."""
        epochs = load_epoch_data(
            bids_root=BIDS_ROOT,
            subject=TEST_SUBJECT,
            description="production",
            band=TEST_BAND,
            datatype=TEST_DATATYPE,
        )
        
        results = compute_connectivity(
            epochs=epochs,
            window=0.5,
            step=0.5,
            model_order=4,
            n_permutations=2,
            n_jobs=1,
        )
        
        assert results['pdc'].shape[0] > 0, "Should have at least one time window"
        print(f"Production: {results['pdc'].shape[0]} windows, "
              f"{len(results['ch_names'])} channels")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
