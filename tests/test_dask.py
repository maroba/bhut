"""
Dask integration tests for bhut.

Tests the Dask backend functionality including array handling,
chunking strategies, and consistency with NumPy backend.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.requires_dask
class TestDaskBackend:
    """Test Dask backend functionality."""

    def test_dask_availability(self):
        """Test Dask availability detection."""
        try:
            from bhut.backends.dask_ import DASK_AVAILABLE
            assert isinstance(DASK_AVAILABLE, bool)
        except ImportError:
            pytest.skip("Dask backend module not available")

    def test_dask_basic_functionality(self, random_system):
        """Test basic Dask backend functionality."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = random_system
        
        # Convert to Dask arrays
        pos_da = da.from_array(positions, chunks=(5, 3))
        masses_da = da.from_array(masses, chunks=5)
        
        # Test acceleration computation
        acc = bhut.accelerations(pos_da, masses_da, backend="dask")
        
        # Should return a Dask array
        assert hasattr(acc, 'compute')
        
        # Compute result
        acc_computed = acc.compute()
        assert acc_computed.shape == (10, 3)
        assert np.all(np.isfinite(acc_computed))

    def test_dask_numpy_consistency(self, simple_system):
        """Test that Dask and NumPy backends give consistent results."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = simple_system
        
        # NumPy result
        acc_numpy = bhut.accelerations(positions, masses, backend="numpy")
        
        # Dask result
        pos_da = da.from_array(positions, chunks=(2, 3))
        masses_da = da.from_array(masses, chunks=2)
        acc_dask = bhut.accelerations(pos_da, masses_da, backend="dask")
        acc_dask_computed = acc_dask.compute()
        
        # Should be very close
        np.testing.assert_allclose(acc_numpy, acc_dask_computed, rtol=1e-12)

    def test_dask_chunking_strategies(self, random_system):
        """Test different Dask chunking strategies."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = random_system
        
        chunking_strategies = [
            (5, 3),    # Chunk particles
            (10, 3),   # No chunking in particles
            (3, 3),    # Smaller chunks
        ]
        
        results = []
        for chunks in chunking_strategies:
            pos_da = da.from_array(positions, chunks=chunks)
            masses_da = da.from_array(masses, chunks=chunks[0])
            
            acc = bhut.accelerations(pos_da, masses_da, backend="dask")
            acc_computed = acc.compute()
            results.append(acc_computed)
            
            assert acc_computed.shape == (10, 3)
            assert np.all(np.isfinite(acc_computed))
        
        # All chunking strategies should give very similar results
        ref_result = results[0]
        for result in results[1:]:
            np.testing.assert_allclose(result, ref_result, rtol=1e-12)
