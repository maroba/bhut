"""
Dask integration tests for bhut.

Tests the Dask backend functionality including array handling,
chunking strategies, and consistency with NumPy backend.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.requires_dask
class TestDaskIntegration:
    """Test Dask array integration with bhut."""

    def test_dask_availability(self):
        """Test Dask availability detection."""
        try:
            from bhut.backends.dask_ import DASK_AVAILABLE
            assert isinstance(DASK_AVAILABLE, bool)
        except ImportError:
            pytest.skip("Dask backend module not available")

    def test_dask_basic_functionality(self):
class TestDaskBackend:
    """Test Dask backend functionality."""

    def test_dask_backend_detection(self) -> None:
        """Test that Dask arrays are properly detected."""
        # Create Dask array
        positions_da = da.from_array(np.random.randn(100, 3), chunks=(50, 3))
        masses_da = da.ones(100, chunks=50)
        
        # Test backend detection
        from bhut.backends.base import get_namespace
        xp = get_namespace(positions_da, backend="auto")
        
        # Should detect as Dask
        from bhut.backends.dask_ import DaskArrayNamespace
        assert isinstance(xp, DaskArrayNamespace)

    def test_dask_vs_numpy_basic_consistency(self) -> None:
        """Test that Dask and NumPy backends give consistent results."""
        # Create test data
        np.random.seed(42)  # For reproducibility
        N = 200
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays with chunking
        chunk_size = 50
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations with NumPy
        acc_numpy = bhut.accelerations(
            positions_np, masses_np, 
            softening=0.01, theta=0.5, backend="numpy"
        )
        
        # Compute accelerations with Dask
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.01, theta=0.5, backend="dask"
        )
        
        # Materialize Dask result
        acc_dask_computed = acc_dask.compute()
        
        # Check consistency
        np.testing.assert_allclose(
            acc_numpy, acc_dask_computed, 
            rtol=1e-12, atol=1e-15,
            err_msg="Dask and NumPy results should be identical"
        )
        
        # Check shapes and dtypes
        assert acc_dask_computed.shape == acc_numpy.shape
        assert acc_dask_computed.dtype == acc_numpy.dtype

    def test_dask_chunking_preservation(self) -> None:
        """Test that Dask arrays preserve chunking structure."""
        # Create test data with specific chunking
        N = 150
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays with specific chunking
        chunk_size = 40
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.01, backend="dask"
        )
        
        # Check that chunking is preserved
        assert isinstance(acc_dask, DaskArray)
        assert acc_dask.chunks[0] == positions_da.chunks[0]  # Same row chunking
        assert acc_dask.chunks[1] == (3,) * len(acc_dask.chunks[1])  # Column chunking
        
        # Check that computation works
        result = acc_dask.compute()
        assert result.shape == (N, 3)
        assert np.isfinite(result).all()

    def test_dask_large_chunking(self) -> None:
        """Test with larger chunks typical of distributed computing."""
        # Create test data
        N = 500
        np.random.seed(123)
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.random.uniform(0.5, 2.0, N).astype(np.float64)
        
        # Create Dask arrays with large chunks (~10k elements as specified)
        chunk_size = 100  # Each chunk has 100 particles
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute with both backends
        acc_numpy = bhut.accelerations(
            positions_np, masses_np,
            softening=0.05, theta=0.7, backend="numpy"
        )
        
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.05, theta=0.7, backend="dask"
        )
        
        # Verify consistency
        acc_dask_computed = acc_dask.compute()
        np.testing.assert_allclose(
            acc_numpy, acc_dask_computed,
            rtol=1e-10, atol=1e-12
        )

    def test_dask_2d_case(self) -> None:
        """Test Dask backend with 2D positions."""
        # Create 2D test data
        N = 80
        positions_np = np.random.randn(N, 2).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays
        chunk_size = 25
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 2))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations in 2D
        acc_numpy = bhut.accelerations(
            positions_np, masses_np,
            softening=0.02, dim=2, backend="numpy"
        )
        
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.02, dim=2, backend="dask"
        )
        
        # Verify consistency
        acc_dask_computed = acc_dask.compute()
        np.testing.assert_allclose(acc_numpy, acc_dask_computed, rtol=1e-12)
        
        # Check 2D output shape
        assert acc_dask_computed.shape == (N, 2)

    def test_dask_auto_backend_detection(self) -> None:
        """Test that 'auto' backend correctly detects Dask arrays."""
        # Create test data
        N = 60
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays
        positions_da = da.from_array(positions_np, chunks=(30, 3))
        masses_da = da.from_array(masses_np, chunks=30)
        
        # Use auto backend - should detect Dask
        acc_auto = bhut.accelerations(
            positions_da, masses_da,
            softening=0.01, backend="auto"  # Should auto-detect Dask
        )
        
        # Should return Dask array
        assert isinstance(acc_auto, DaskArray)
        
        # Compare with explicit NumPy computation
        acc_numpy = bhut.accelerations(
            positions_np, masses_np,
            softening=0.01, backend="numpy"
        )
        
        np.testing.assert_allclose(acc_auto.compute(), acc_numpy, rtol=1e-12)

    def test_dask_tree_api_consistency(self) -> None:
        """Test that Tree API works consistently with Dask arrays."""
        # Create test data
        N = 120
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays
        positions_da = da.from_array(positions_np, chunks=(40, 3))
        masses_da = da.from_array(masses_np, chunks=40)
        
        # Test with Tree API - NumPy
        tree_numpy = bhut.Tree(positions_np, masses_np, backend="numpy")
        tree_numpy.build()
        acc_tree_numpy = tree_numpy.accelerations(softening=0.01)
        
        # Note: Tree API currently materializes Dask arrays for tree building
        # This is the expected behavior since tree construction doesn't parallelize well
        tree_dask = bhut.Tree(positions_da, masses_da, backend="dask")
        tree_dask.build()
        acc_tree_dask = tree_dask.accelerations(softening=0.01)
        
        # Results should be identical (both use materialized arrays)
        np.testing.assert_allclose(acc_tree_numpy, acc_tree_dask, rtol=1e-14)

    def test_dask_different_chunk_sizes(self) -> None:
        """Test Dask computation with various chunk sizes."""
        # Create test data
        N = 200
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Reference result with NumPy
        acc_ref = bhut.accelerations(
            positions_np, masses_np,
            softening=0.01, backend="numpy"
        )
        
        # Test different chunk sizes
        chunk_sizes = [20, 50, 75, 100]
        
        for chunk_size in chunk_sizes:
            positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
            masses_da = da.from_array(masses_np, chunks=chunk_size)
            
            acc_dask = bhut.accelerations(
                positions_da, masses_da,
                softening=0.01, backend="dask"
            )
            
            acc_computed = acc_dask.compute()
            
            np.testing.assert_allclose(
                acc_ref, acc_computed, rtol=1e-12,
                err_msg=f"Failed for chunk_size={chunk_size}"
            )

    def test_dask_error_handling(self) -> None:
        """Test error handling with Dask arrays."""
        # Create test data
        N = 50
        positions_da = da.from_array(np.random.randn(N, 3), chunks=(25, 3))
        masses_da = da.ones(N, chunks=25)
        
        # Test invalid softening
        with pytest.raises(ValueError, match="non-negative"):
            bhut.accelerations(positions_da, masses_da, softening=-0.1, backend="dask")
        
        # Test invalid theta
        with pytest.raises(ValueError, match="non-negative"):
            bhut.accelerations(positions_da, masses_da, theta=-0.1, backend="dask")
        
        # Test mismatched shapes
        bad_masses = da.ones(N-5, chunks=20)
        with pytest.raises(ValueError, match="particles"):
            bhut.accelerations(positions_da, bad_masses, backend="dask")

    def test_dask_performance_chunking(self) -> None:
        """Test that Dask computation respects chunking for performance."""
        # Create test data with large chunks
        N = 300
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays with chunk size close to 10k as specified
        # For N=300, use chunk_size=100 to get reasonable chunks
        chunk_size = 100
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.01, theta=0.5, backend="dask"
        )
        
        # Verify that result maintains chunking structure
        assert isinstance(acc_dask, DaskArray)
        
        # Check that number of chunks is as expected
        expected_n_chunks = int(np.ceil(N / chunk_size))
        assert len(acc_dask.chunks[0]) == expected_n_chunks
        
        # Verify computation produces valid results
        result = acc_dask.compute()
        assert result.shape == (N, 3)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()


@pytest.mark.skipif(DASK_AVAILABLE, reason="Testing behavior when Dask is not available")
class TestDaskNotAvailable:
    """Test behavior when Dask is not available."""

    def test_dask_backend_error_when_not_available(self) -> None:
        """Test that requesting Dask backend gives helpful error when not available."""
        positions = np.random.randn(10, 3)
        masses = np.ones(10)
        
        with pytest.raises(ValueError, match="Dask backend not yet implemented"):
            bhut.accelerations(positions, masses, backend="dask")


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")  
class TestDaskIntegration:
    """Integration tests for Dask functionality."""

    def test_dask_full_workflow(self) -> None:
        """Test complete workflow with Dask arrays."""
        # Create realistic test scenario
        N = 400
        np.random.seed(456)
        
        # Create initial positions in a sphere
        theta = np.random.uniform(0, 2*np.pi, N)
        phi = np.random.uniform(0, np.pi, N)
        r = np.random.uniform(0.1, 1.0, N)
        
        positions_np = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ]).astype(np.float64)
        
        masses_np = np.random.uniform(0.1, 2.0, N).astype(np.float64)
        
        # Create Dask arrays with medium-sized chunks
        chunk_size = 80
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations with both backends
        acc_numpy = bhut.accelerations(
            positions_np, masses_np,
            softening=0.05, theta=0.6, G=1.0, backend="numpy"
        )
        
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.05, theta=0.6, G=1.0, backend="dask"
        )
        
        # Verify results
        acc_dask_computed = acc_dask.compute()
        
        # Should be nearly identical
        np.testing.assert_allclose(acc_numpy, acc_dask_computed, rtol=1e-10)
        
        # Check physical properties
        assert np.isfinite(acc_dask_computed).all()
        assert acc_dask_computed.shape == (N, 3)
        
        # Check that accelerations point generally inward for sphere
        # (center of mass should attract particles)
        com = np.average(positions_np, weights=masses_np, axis=0)
        to_com = com[np.newaxis, :] - positions_np
        
        # Most accelerations should have positive dot product with direction to COM
        dots = np.sum(acc_dask_computed * to_com, axis=1)
        positive_fraction = np.mean(dots > 0)
        assert positive_fraction > 0.7  # Most particles should be attracted to center

    def test_dask_memory_efficiency(self) -> None:
        """Test that Dask computation doesn't load all data at once."""
        # Create moderately large dataset
        N = 600
        positions_np = np.random.randn(N, 3).astype(np.float64)
        masses_np = np.ones(N, dtype=np.float64)
        
        # Create Dask arrays with small chunks to test memory efficiency
        chunk_size = 50
        positions_da = da.from_array(positions_np, chunks=(chunk_size, 3))
        masses_da = da.from_array(masses_np, chunks=chunk_size)
        
        # Compute accelerations
        acc_dask = bhut.accelerations(
            positions_da, masses_da,
            softening=0.01, backend="dask"
        )
        
        # Result should be a Dask array (not computed yet)
        assert isinstance(acc_dask, DaskArray)
        
        # Should have appropriate chunking
        assert len(acc_dask.chunks[0]) == int(np.ceil(N / chunk_size))
        
        # Computing should work
        result = acc_dask.compute()
        assert result.shape == (N, 3)
        assert np.isfinite(result).all()
