"""
Unit tests for array namespace backends.

Tests NumPy and Dask backend implementations and selection logic.
"""

import numpy as np
import pytest

import bhut
from bhut.backends.base import get_namespace
from bhut.backends.numpy_ import NumpyArrayNamespace


@pytest.mark.unit
class TestNumpyBackend:
    """Test NumPy array namespace backend."""

    def test_basic_operations(self):
        """Test basic array operations."""
        xp = NumpyArrayNamespace()
        
        # Test array creation
        arr = np.array([1, 2, 3, 4])
        zeros = xp.zeros((3, 2))
        assert zeros.shape == (3, 2)
        assert np.all(zeros == 0)
        
        # Test mathematical operations
        result = xp.sum(arr)
        assert result == 10
        
        sqrt_result = xp.sqrt(np.array([4, 9, 16]))
        np.testing.assert_allclose(sqrt_result, [2, 3, 4])
        
        # Test where operation
        where_result = xp.where(arr > 2, arr, 0)
        np.testing.assert_array_equal(where_result, [0, 0, 3, 4])

    def test_array_creation_methods(self):
        """Test array creation methods."""
        xp = NumpyArrayNamespace()
        
        # Test zeros with different dtypes
        zeros_float = xp.zeros((2, 3), dtype=np.float32)
        assert zeros_float.shape == (2, 3)
        assert zeros_float.dtype == np.float32
        
        # Test arange
        arange_result = xp.arange(5)
        np.testing.assert_array_equal(arange_result, [0, 1, 2, 3, 4])
        
        # Test arange with parameters
        arange_step = xp.arange(2, 10, 2)
        np.testing.assert_array_equal(arange_step, [2, 4, 6, 8])

    def test_mathematical_functions(self):
        """Test mathematical functions."""
        xp = NumpyArrayNamespace()
        
        arr = np.array([1.0, 4.0, 9.0, 16.0])
        
        # Test sqrt
        sqrt_result = xp.sqrt(arr)
        np.testing.assert_allclose(sqrt_result, [1, 2, 3, 4])
        
        # Test sum with axis
        arr_2d = np.array([[1, 2], [3, 4]])
        sum_axis0 = xp.sum(arr_2d, axis=0)
        np.testing.assert_array_equal(sum_axis0, [4, 6])
        
        sum_axis1 = xp.sum(arr_2d, axis=1) 
        np.testing.assert_array_equal(sum_axis1, [3, 7])

    def test_indexing_operations(self):
        """Test array indexing and manipulation."""
        xp = NumpyArrayNamespace()
        
        arr = np.array([3, 1, 4, 1, 5])
        
        # Test argsort
        indices = xp.argsort(arr)
        expected_indices = [1, 3, 0, 2, 4]  # Sorted: [1, 1, 3, 4, 5]
        np.testing.assert_array_equal(indices, expected_indices)
        
        # Test sort
        sorted_arr = xp.sort(arr)
        np.testing.assert_array_equal(sorted_arr, [1, 1, 3, 4, 5])
        
        # Test take
        result = xp.take(arr, [0, 2, 4])
        np.testing.assert_array_equal(result, [3, 4, 5])

    def test_concatenation(self):
        """Test array concatenation."""
        xp = NumpyArrayNamespace()
        
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        
        # Concatenate along axis 0
        result0 = xp.concatenate([arr1, arr2], axis=0)
        expected0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(result0, expected0)
        
        # Concatenate along axis 1
        result1 = xp.concatenate([arr1, arr2], axis=1)
        expected1 = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result1, expected1)


@pytest.mark.unit
class TestBackendSelection:
    """Test backend selection and auto-detection."""

    def test_numpy_backend_detection(self):
        """Test that NumPy arrays are detected correctly."""
        arrays = [
            np.array([1, 2, 3]),
            np.random.randn(5, 3).astype(np.float32),
            np.random.randint(0, 10, (4, 2)).astype(np.int64),
            np.zeros((2, 2, 2)),
        ]
        
        for arr in arrays:
            xp = get_namespace(arr, "auto")
            assert isinstance(xp, NumpyArrayNamespace)

    def test_explicit_backend_selection(self, random_system):
        """Test explicit backend selection."""
        positions, masses = random_system
        
        # Test explicit numpy backend
        acc_numpy = bhut.accelerations(positions, masses, backend="numpy")
        acc_auto = bhut.accelerations(positions, masses, backend="auto")
        
        # Should give identical results for numpy arrays
        np.testing.assert_allclose(acc_numpy, acc_auto, rtol=1e-15)

    def test_backend_consistency(self, hierarchical_system):
        """Test that backend choice doesn't affect results."""
        positions, masses = hierarchical_system
        
        # All should give same results for numpy arrays
        backends = ["auto", "numpy"]
        results = {}
        
        for backend in backends:
            acc = bhut.accelerations(positions, masses, backend=backend)
            results[backend] = acc
            assert acc.shape == (3, 3)
            assert np.all(np.isfinite(acc))
        
        # All results should be identical
        ref_result = results["auto"]
        for backend, acc in results.items():
            np.testing.assert_allclose(acc, ref_result, rtol=1e-15)


@pytest.mark.unit
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


@pytest.mark.unit
class TestBackendErrorHandling:
    """Test error handling in backend operations."""

    def test_invalid_backend_specification(self, random_system):
        """Test error handling for invalid backend specification."""
        positions, masses = random_system
        
        with pytest.raises(ValueError, match="backend"):
            bhut.accelerations(positions, masses, backend="invalid_backend")

    def test_mismatched_array_types(self):
        """Test error handling for mismatched array types."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        # NumPy positions, Dask masses
        positions = np.random.randn(5, 3).astype(np.float64)
        masses_da = da.from_array(np.ones(5), chunks=3)
        
        # Should handle mixed types appropriately
        # (specific behavior depends on implementation)
        try:
            acc = bhut.accelerations(positions, masses_da, backend="auto")
            assert acc.shape == (5, 3)
        except (ValueError, TypeError):
            # Some mixed-type combinations might not be supported
            pass

    def test_array_namespace_errors(self):
        """Test array namespace compatibility."""
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        masses = [1.0, 1.0]  # Different type from numpy array
        
        # The API actually handles this gracefully by converting to numpy arrays
        acc = bhut.accelerations(positions, masses)
        assert acc.shape == (2, 3)
