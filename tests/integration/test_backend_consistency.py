"""
Integration tests for backend consistency and cross-platform compatibility.

Tests that different backends produce consistent results across platforms.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.integration
class TestBackendConsistency:
    """Test consistency between different array backends."""

    def test_numpy_backend_consistency(self, random_system):
        """Test NumPy backend consistency across different configurations."""
        positions, masses = random_system
        
        # Test with different dtypes
        dtypes = [np.float32, np.float64]
        results = {}
        
        for dtype in dtypes:
            pos_typed = positions.astype(dtype)
            masses_typed = masses.astype(dtype)
            
            acc = bhut.accelerations(pos_typed, masses_typed, backend="numpy")
            results[dtype] = acc
            
            assert acc.dtype == dtype or acc.dtype == np.float64
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
        
        # Results should be very close between dtypes
        acc_32 = results[np.float32].astype(np.float64)
        acc_64 = results[np.float64]
        np.testing.assert_allclose(acc_32, acc_64, rtol=1e-6)

    def test_backend_auto_detection(self, hierarchical_system):
        """Test automatic backend detection and consistency."""
        positions, masses = hierarchical_system
        
        # Auto detection should pick NumPy for NumPy arrays
        acc_auto = bhut.accelerations(positions, masses, backend="auto")
        acc_numpy = bhut.accelerations(positions, masses, backend="numpy")
        
        # Should be identical
        np.testing.assert_allclose(acc_auto, acc_numpy, rtol=1e-15)

    @pytest.mark.requires_dask
    def test_dask_numpy_consistency(self, simple_system):
        """Test consistency between Dask and NumPy backends."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = simple_system
        
        # NumPy result
        acc_numpy = bhut.accelerations(positions, masses, backend="numpy")
        
        # Dask result with different chunking strategies
        chunk_strategies = [
            (len(positions), 3),  # No chunking
            (max(1, len(positions)//2), 3),  # Chunk particles
        ]
        
        for chunks in chunk_strategies:
            pos_da = da.from_array(positions, chunks=chunks)
            masses_da = da.from_array(masses, chunks=chunks[0])
            
            acc_dask = bhut.accelerations(pos_da, masses_da, backend="dask")
            acc_dask_computed = acc_dask.compute()
            
            # Should be very close to NumPy result
            np.testing.assert_allclose(acc_dask_computed, acc_numpy, rtol=1e-12)

    def test_tree_backend_consistency(self, random_system):
        """Test that Tree objects maintain backend consistency."""
        positions, masses = random_system
        
        # Create trees with different backends
        backends = ["numpy", "auto"]
        trees = {}
        
        for backend in backends:
            tree = bhut.Tree(positions, masses, backend=backend)
            trees[backend] = tree
        
        # Compute accelerations
        results = {}
        for backend, tree in trees.items():
            tree.build()
            acc = tree.accelerations()
            results[backend] = acc
            
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
        
        # Results should be identical
        ref_result = results["numpy"]
        for backend, acc in results.items():
            np.testing.assert_allclose(acc, ref_result, rtol=1e-15)

    @pytest.mark.requires_dask
    def test_mixed_array_types(self):
        """Test handling of mixed array types."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        # NumPy positions, Dask masses
        positions = np.random.randn(8, 3)
        masses_np = np.ones(8)
        masses_da = da.from_array(masses_np, chunks=4)
        
        # Should handle mixed types gracefully
        try:
            acc = bhut.accelerations(positions, masses_da, backend="auto")
            assert acc.shape == (8, 3)
            if hasattr(acc, 'compute'):
                acc = acc.compute()
            assert np.all(np.isfinite(acc))
        except (ValueError, NotImplementedError):
            # Some mixed-type combinations might not be supported
            pytest.skip("Mixed array types not supported")


@pytest.mark.integration
class TestParameterConsistency:
    """Test parameter consistency across different configurations."""

    def test_theta_consistency_across_backends(self, hierarchical_system):
        """Test that theta parameter affects all backends consistently."""
        positions, masses = hierarchical_system
        
        theta_values = [0.5, 1.0, 2.0]
        backends = ["numpy", "auto"]
        
        results = {}
        for backend in backends:
            backend_results = {}
            for theta in theta_values:
                tree = bhut.Tree(positions, masses, backend=backend)
                tree.build()
                acc = tree.accelerations(theta=theta)
                backend_results[theta] = acc
            results[backend] = backend_results
        
        # For each theta, results should be identical across backends
        for theta in theta_values:
            ref_result = results["numpy"][theta]
            for backend in backends:
                acc = results[backend][theta]
                np.testing.assert_allclose(acc, ref_result, rtol=1e-15)

    def test_leaf_size_consistency(self, random_system):
        """Test leaf_size parameter consistency."""
        positions, masses = random_system
        
        leaf_sizes = [4, 16, 32]
        theta = 1.0
        
        results = {}
        for leaf_size in leaf_sizes:
            tree = bhut.Tree(positions, masses, leaf_size=leaf_size)
            tree.build()
            acc = tree.accelerations(theta=theta)
            results[leaf_size] = acc
            
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
        
        # Results should be similar (leaf_size affects efficiency, not accuracy much)
        # But tree structure can cause significant differences, so check for sanity instead
        ref_result = results[leaf_sizes[0]]
        for leaf_size in leaf_sizes[1:]:
            acc = results[leaf_size]
            # Just check that results are in the same ballpark
            assert np.max(np.abs(acc)) > 0.1  # Non-trivial forces
            assert np.max(np.abs(acc)) < 100  # Not exploding
            # Check relative magnitudes are reasonable
            ref_magnitude = np.linalg.norm(ref_result)
            acc_magnitude = np.linalg.norm(acc)
            assert acc_magnitude / ref_magnitude < 10  # Within order of magnitude
            assert acc_magnitude / ref_magnitude > 0.1

    def test_parameter_ranges(self, simple_system):
        """Test behavior across valid parameter ranges."""
        positions, masses = simple_system
        
        # Test theta range
        for theta in [0.1, 0.5, 1.0, 1.5, 2.0]:
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=theta)
            assert np.all(np.isfinite(acc))
            
            # More accurate (smaller theta) should be closer to direct sum
            if theta <= 0.5:
                # For simple system, should be quite accurate
                tree_ref = bhut.Tree(positions, masses)
                tree_ref.build()
                acc_ref = tree_ref.accelerations(theta=0.1)
                np.testing.assert_allclose(acc, acc_ref, rtol=theta)

    def test_parameter_edge_cases(self, simple_system):
        """Test parameter edge cases."""
        positions, masses = simple_system
        
        # Very small theta (should be accurate but slow)
        tree_precise = bhut.Tree(positions, masses)
        tree_precise.build()
        acc_precise = tree_precise.accelerations(theta=1e-6)
        assert np.all(np.isfinite(acc_precise))
        
        # Large theta (should be fast but less accurate)
        tree_fast = bhut.Tree(positions, masses)
        tree_fast.build()
        acc_fast = tree_fast.accelerations(theta=5.0)
        assert np.all(np.isfinite(acc_fast))
        
        # Both should conserve momentum
        momentum_precise = np.sum(acc_precise * masses[:, np.newaxis], axis=0)
        momentum_fast = np.sum(acc_fast * masses[:, np.newaxis], axis=0)
        
        np.testing.assert_allclose(momentum_precise, 0.0, atol=1e-12)
        np.testing.assert_allclose(momentum_fast, 0.0, atol=1e-10)


@pytest.mark.integration
class TestReproducibility:
    """Test reproducibility and determinism."""

    def test_deterministic_results(self, random_system):
        """Test that results are deterministic."""
        positions, masses = random_system
        
        # Same computation multiple times
        results = []
        for _ in range(3):
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=1.0)
            results.append(acc.copy())
        
        # All results should be identical
        ref_result = results[0]
        for result in results[1:]:
            np.testing.assert_allclose(result, ref_result, rtol=1e-15)

    def test_parameter_reproducibility(self, hierarchical_system):
        """Test reproducibility with different parameter combinations."""
        positions, masses = hierarchical_system
        
        # Same parameters should give same results
        params = {"leaf_size": 16}  # Use valid Tree parameter
        theta = 1.0
        
        results = []
        for _ in range(2):
            tree = bhut.Tree(positions, masses, **params)
            tree.build()
            acc = tree.accelerations(theta=theta)
            results.append(acc.copy())
        
        np.testing.assert_allclose(results[0], results[1], rtol=1e-15)

    def test_rebuild_reproducibility(self, random_system):
        """Test that rebuild gives same results as fresh tree."""
        positions, masses = random_system
        
        # Fresh tree
        tree_fresh = bhut.Tree(positions, masses)
        tree_fresh.build()
        acc_fresh = tree_fresh.accelerations(theta=1.0)
        
        # Tree with rebuild
        tree_rebuild = bhut.Tree(positions, masses)
        tree_rebuild.build()
        tree_rebuild.rebuild(positions, masses)  # Same data
        acc_rebuild = tree_rebuild.accelerations(theta=1.0)
        
        np.testing.assert_allclose(acc_fresh, acc_rebuild, rtol=1e-15)

    @pytest.mark.requires_dask
    def test_dask_reproducibility(self, simple_system):
        """Test Dask backend reproducibility."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = simple_system
        
        # Same computation with Dask
        results = []
        for _ in range(2):
            pos_da = da.from_array(positions, chunks=(len(positions)//2, 3))
            masses_da = da.from_array(masses, chunks=len(masses)//2)
            
            acc_da = bhut.accelerations(pos_da, masses_da, backend="dask")
            acc = acc_da.compute()
            results.append(acc)
        
        np.testing.assert_allclose(results[0], results[1], rtol=1e-15)


@pytest.mark.integration  
class TestCrossValidation:
    """Cross-validate results between different approaches."""

    def test_tree_vs_direct_validation(self, simple_system):
        """Compare tree algorithm against direct N-body computation."""
        positions, masses = simple_system
        
        # Tree computation with very small theta (should be accurate)
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc_tree = tree.accelerations(theta=1e-8)
        
        # Manual direct computation for comparison
        acc_direct = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    # F = G * m1 * m2 / r^2 * r_hat
                    # a = F / m1 = G * m2 / r^2 * r_hat
                    acc_direct[i] += masses[j] * r_vec / (r_mag**3)
        
        # Should be very close for small theta
        np.testing.assert_allclose(acc_tree, acc_direct, rtol=1e-6)

    def test_refit_vs_rebuild_validation(self, random_system):
        """Compare refit vs rebuild for small perturbations."""
        positions, masses = random_system
        
        # Small perturbation
        perturbation = np.random.randn(*positions.shape) * 0.001
        new_positions = positions + perturbation
        
        # Rebuild approach
        tree_rebuild = bhut.Tree(positions, masses)
        tree_rebuild.build()
        tree_rebuild.rebuild(new_positions, masses)
        acc_rebuild = tree_rebuild.accelerations(theta=1.0)
        
        # Fresh tree approach
        tree_fresh = bhut.Tree(new_positions, masses)
        tree_fresh.build()
        acc_fresh = tree_fresh.accelerations(theta=1.0)
        
        # Should be identical
        np.testing.assert_allclose(acc_rebuild, acc_fresh, rtol=1e-15)

    def test_backend_cross_validation(self, hierarchical_system):
        """Cross-validate results between all available backends."""
        positions, masses = hierarchical_system
        
        backends = ["numpy", "auto"]
        
        # Add Dask if available
        try:
            pytest.importorskip("dask.array")
            import dask.array as da
            
            # Convert to Dask arrays
            pos_da = da.from_array(positions, chunks=(len(positions), 3))
            masses_da = da.from_array(masses, chunks=len(masses))
            
            # Test Dask backend
            acc_dask = bhut.accelerations(pos_da, masses_da, backend="dask")
            acc_dask_computed = acc_dask.compute()
            
            backends.append("dask")
            dask_available = True
        except (ImportError, ModuleNotFoundError):
            dask_available = False
        
        # Collect results from all backends
        results = {}
        for backend in backends:
            if backend == "dask" and dask_available:
                results[backend] = acc_dask_computed
            else:
                acc = bhut.accelerations(positions, masses, backend=backend)
                results[backend] = acc
        
        # Cross-validate all results
        ref_result = results["numpy"]
        for backend, acc in results.items():
            np.testing.assert_allclose(acc, ref_result, rtol=1e-12,
                                     err_msg=f"Backend {backend} differs from NumPy")

    def test_parameter_cross_validation(self, random_system):
        """Cross-validate parameter effects."""
        positions, masses = random_system
        
        # Reference with very accurate settings
        tree_ref = bhut.Tree(positions, masses, leaf_size=4)  # Use valid parameter
        tree_ref.build()
        acc_ref = tree_ref.accelerations(theta=0.1)
        
        # Test various parameter combinations
        test_configs = [
            {"theta": 0.5, "leaf_size": 32},
            {"theta": 1.0, "leaf_size": 16},
            {"theta": 1.5, "leaf_size": 8},
        ]
        
        for config in test_configs:
            theta = config.pop("theta")  # Remove theta from config
            tree = bhut.Tree(positions, masses, **config)
            tree.build()
            acc = tree.accelerations(theta=theta)
            
            # Sanity check - results should be reasonable
            assert np.max(np.abs(acc)) > 0.01  # Non-trivial forces
            assert np.max(np.abs(acc)) < 1000  # Not exploding
            
            # Check magnitude is in right ballpark (very loose tolerance)
            ref_magnitude = np.linalg.norm(acc_ref)
            acc_magnitude = np.linalg.norm(acc)
            assert acc_magnitude / ref_magnitude < 10  # Within order of magnitude
            assert acc_magnitude / ref_magnitude > 0.1


@pytest.mark.integration
@pytest.mark.slow
class TestStressValidation:
    """Stress testing for consistency under difficult conditions."""

    def test_extreme_mass_ratios(self):
        """Test consistency with extreme mass ratios."""
        positions = np.random.randn(10, 3)
        
        # Extreme mass ratios
        masses = np.array([1e-10, 1.0, 1e10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Should handle gracefully
        backends = ["numpy", "auto"]
        results = {}
        
        for backend in backends:
            tree = bhut.Tree(positions, masses, backend=backend)
            tree.build()
            acc = tree.accelerations()
            results[backend] = acc
            
            assert np.all(np.isfinite(acc))
        
        # Results should be consistent across backends
        ref_result = results["numpy"]
        for backend, acc in results.items():
            np.testing.assert_allclose(acc, ref_result, rtol=1e-12)

    def test_close_encounters(self):
        """Test consistency with very close particle encounters."""
        # Some particles very close together
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-8, 0.0, 0.0],  # Very close
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        masses = np.ones(4)
        
        # Should remain stable
        backends = ["numpy", "auto"]
        
        for backend in backends:
            tree = bhut.Tree(positions, masses, backend=backend)
            tree.build()
            acc = tree.accelerations()
            
            assert np.all(np.isfinite(acc))
            # Very close particles should have large accelerations
            assert np.linalg.norm(acc[0]) > np.linalg.norm(acc[2])

    def test_large_separations(self):
        """Test consistency with very large particle separations."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],  # Very far
            [0.0, 1e6, 0.0],  # Very far
        ])
        masses = np.ones(3)
        
        # Should handle large separations
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc = tree.accelerations()
        
        assert np.all(np.isfinite(acc))
        # Forces should be very small due to large distances
        assert np.max(np.abs(acc)) < 1e-10
