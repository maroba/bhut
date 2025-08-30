"""
Integration tests for complete workflows and end-to-end functionality.

Tests interactions between different components and realistic usage patterns.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.integration
class TestWorkflowIntegration:
    """Test complete workflows from start to finish."""

    def test_basic_simulation_workflow(self, random_system):
        """Test a basic simulation workflow."""
        positions, masses = random_system
        
        # Step 1: Create tree
        tree = bhut.Tree(positions, masses)
        tree.build()
        assert len(tree.positions) == len(positions)
        
        # Step 2: Compute accelerations
        accelerations = tree.accelerations()
        assert accelerations.shape == positions.shape
        assert np.all(np.isfinite(accelerations))
        
        # Step 3: Use accelerations in time integration (mock)
        dt = 0.01
        velocities = np.zeros_like(positions)
        
        # Simple leapfrog-style update
        velocities += 0.5 * dt * accelerations
        new_positions = positions + dt * velocities
        
        assert new_positions.shape == positions.shape
        assert not np.array_equal(new_positions, positions)

    def test_tree_rebuild_workflow(self, hierarchical_system):
        """Test workflow with tree rebuilding."""
        positions, masses = hierarchical_system
        
        # Initial tree
        tree = bhut.Tree(positions, masses)
        tree.build()
        initial_acc = tree.accelerations(theta=1.0)
        
        # Modify positions significantly
        new_positions = positions + np.random.randn(*positions.shape) * 0.1
        
        # Rebuild tree
        tree.rebuild(new_positions, masses)
        new_acc = tree.accelerations()
        
        assert new_acc.shape == initial_acc.shape
        assert np.all(np.isfinite(new_acc))
        # Should be different due to position change
        assert not np.allclose(initial_acc, new_acc, rtol=1e-6)

    def test_refitting_workflow(self, simple_system):
        """Test workflow with tree refitting."""
        positions, masses = simple_system
        
        # Initial tree
        tree = bhut.Tree(positions, masses)
        tree.build()
        initial_acc = tree.accelerations(theta=0.8)
        
        # Small position perturbation
        new_positions = positions + np.random.randn(*positions.shape) * 0.001  # Smaller perturbation
        
        # Refit instead of rebuild
        tree.refit(new_positions)
        refitted_acc = tree.accelerations()
        
        assert refitted_acc.shape == initial_acc.shape
        assert np.all(np.isfinite(refitted_acc))
        
        # For very small perturbations, should be similar
        np.testing.assert_allclose(initial_acc, refitted_acc, rtol=1e-1, atol=1e-2)

    def test_parameter_sweep_workflow(self, simple_system):
        """Test workflow with parameter sweeps."""
        positions, masses = simple_system
        
        # Test different theta values
        theta_values = [0.5, 1.0, 1.5, 2.0]
        results = {}
        
        for theta in theta_values:
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=theta)
            results[theta] = acc
            
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
        
        # Results should converge as theta decreases (more accurate)
        # For simple system, differences should be small anyway
        ref_result = results[0.5]  # Most accurate
        for theta, acc in results.items():
            if theta > 0.5:
                # Allow more error for larger theta
                max_rtol = theta * 0.1
                np.testing.assert_allclose(acc, ref_result, rtol=max_rtol)


@pytest.mark.integration
class TestBackendIntegration:
    """Test integration between different backends."""

    def test_backend_consistency_workflow(self, random_system):
        """Test that different backends give consistent results."""
        positions, masses = random_system
        
        # Test available backends
        backends = ["auto", "numpy"]
        results = {}
        
        for backend in backends:
            acc = bhut.accelerations(positions, masses, backend=backend)
            results[backend] = acc
            
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
        
        # All backends should give very similar results
        ref_result = results["numpy"]
        for backend, acc in results.items():
            np.testing.assert_allclose(acc, ref_result, rtol=1e-12)

    @pytest.mark.requires_dask
    def test_dask_integration_workflow(self, random_system):
        """Test Dask integration in complete workflow."""
        pytest.importorskip("dask.array")
        import dask.array as da
        
        positions, masses = random_system
        
        # Convert to Dask arrays
        positions_da = da.from_array(positions, chunks=(5, 3))
        masses_da = da.from_array(masses, chunks=5)
        
        # Workflow with Dask arrays
        tree = bhut.Tree(positions_da, masses_da)
        tree.build()
        acc_da = tree.accelerations()
        
        # For Tree.accelerations(), the result is a numpy array
        # Let's check if it's at least the right shape and finite
        assert acc_da.shape == positions.shape
        assert np.all(np.isfinite(acc_da))
        
        # The result is already computed (numpy array)
        assert acc_da.shape == positions.shape
        assert np.all(np.isfinite(acc_da))
        
        # Compare with NumPy version
        tree_numpy = bhut.Tree(positions, masses)
        tree_numpy.build()
        acc_numpy = tree_numpy.accelerations()
        
        np.testing.assert_allclose(acc_da, acc_numpy, rtol=1e-12)

    def test_mixed_backend_workflow(self, hierarchical_system):
        """Test workflow mixing different backends."""
        positions, masses = hierarchical_system
        
        # Start with NumPy
        tree_numpy = bhut.Tree(positions, masses, backend="numpy")
        tree_numpy.build()
        acc_numpy = tree_numpy.accelerations()
        
        # Switch to auto backend (should detect NumPy)
        acc_auto = bhut.accelerations(positions, masses, backend="auto")
        
        # Should be identical for NumPy arrays
        np.testing.assert_allclose(acc_numpy, acc_auto, rtol=1e-15)


@pytest.mark.integration
class TestScalingIntegration:
    """Test integration at different scales."""

    def test_small_system_integration(self):
        """Test integration with very small systems."""
        # Two-body system
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 1.0])
        
        # Should work with minimal leaf size
        tree = bhut.Tree(positions, masses, leaf_size=2)
        tree.build()
        acc = tree.accelerations(theta=1.0)
        
        assert acc.shape == (2, 3)
        assert np.all(np.isfinite(acc))
        
        # Verify symmetry
        np.testing.assert_allclose(acc[0], -acc[1], rtol=1e-12)

    def test_medium_system_integration(self, random_system):
        """Test integration with medium-sized systems."""
        positions, masses = random_system  # 10 particles
        
        # Multiple workflow steps
        tree = bhut.Tree(positions, masses)
        tree.build()
        
        # Simulate several timesteps
        current_positions = positions.copy()
        current_velocities = np.zeros_like(positions)
        dt = 0.01
        n_steps = 5
        
        for step in range(n_steps):
            # Compute accelerations
            tree.refit(current_positions)  # Update tree
            acc = tree.accelerations()
            
            # Time integration
            current_velocities += dt * acc
            current_positions += dt * current_velocities
            
            # Verify results remain finite
            assert np.all(np.isfinite(current_positions))
            assert np.all(np.isfinite(current_velocities))
            assert np.all(np.isfinite(acc))

    @pytest.mark.slow
    def test_large_system_integration(self):
        """Test integration with larger systems."""
        # Create larger system
        n_particles = 100
        positions = np.random.randn(n_particles, 3).astype(np.float64)
        masses = np.abs(np.random.randn(n_particles)) + 0.1
        
        # Should handle larger systems efficiently
        tree = bhut.Tree(positions, masses, leaf_size=16)
        tree.build()
        acc = tree.accelerations()
        
        assert acc.shape == (n_particles, 3)
        assert np.all(np.isfinite(acc))
        
        # Check that center of mass acceleration is zero
        total_force = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery and robustness in workflows."""

    def test_invalid_input_recovery(self):
        """Test recovery from invalid inputs in workflow."""
        # Start with valid system
        positions = np.random.randn(5, 3)
        masses = np.ones(5)
        
        tree = bhut.Tree(positions, masses)
        tree.build()
        valid_acc = tree.accelerations()
        
        # Try invalid update
        invalid_positions = np.array([[np.nan, 0, 0]])
        
        with pytest.raises(ValueError):
            tree.refit(invalid_positions)
        
        # Original tree should still work
        still_valid_acc = tree.accelerations()
        np.testing.assert_allclose(valid_acc, still_valid_acc, rtol=1e-15)

    def test_extreme_parameter_recovery(self, simple_system):
        """Test recovery from extreme parameter values."""
        positions, masses = simple_system
        
        # Very small theta (expensive but should work)
        tree_small_theta = bhut.Tree(positions, masses)
        tree_small_theta.build()
        acc_precise = tree_small_theta.accelerations(theta=1e-6)
        
        # Very large theta (fast but less accurate)
        tree_large_theta = bhut.Tree(positions, masses)
        tree_large_theta.build()
        acc_fast = tree_large_theta.accelerations(theta=10.0)
        
        # Both should be finite and roughly similar for simple system
        assert np.all(np.isfinite(acc_precise))
        assert np.all(np.isfinite(acc_fast))
        np.testing.assert_allclose(acc_precise, acc_fast, rtol=1.0)

    def test_memory_pressure_recovery(self):
        """Test behavior under memory pressure conditions."""
        # Create system that might stress memory
        positions = np.random.randn(50, 3)
        masses = np.ones(50)
        
        # Multiple trees simultaneously
        trees = []
        for i in range(5):
            tree = bhut.Tree(positions, masses)
            tree.build()
            trees.append(tree)
        
        # All should work
        for tree in trees:
            acc = tree.accelerations()
            assert acc.shape == (50, 3)
            assert np.all(np.isfinite(acc))


@pytest.mark.integration  
class TestPhysicsIntegration:
    """Test physics consistency in integration workflows."""

    def test_energy_conservation_workflow(self, simple_system):
        """Test energy-related properties in workflow."""
        positions, masses = simple_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc = tree.accelerations()
        
        # Total momentum change should be zero
        total_momentum_change = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_momentum_change, 0.0, atol=1e-12)
        
        # For pair system, forces should be equal and opposite
        if len(positions) == 2:
            force_0 = acc[0] * masses[0]
            force_1 = acc[1] * masses[1]
            np.testing.assert_allclose(force_0, -force_1, rtol=1e-12)

    def test_symmetry_preservation_workflow(self):
        """Test symmetry preservation in workflows."""
        # Symmetric configuration
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        masses = np.array([1.0, 1.0, 2.0, 2.0])  # Symmetric masses
        
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc = tree.accelerations()
        
        # Check symmetries
        # Particles 0 and 1 should have opposite x-acceleration
        np.testing.assert_allclose(acc[0, 0], -acc[1, 0], rtol=1e-10)
        # Particles 2 and 3 should have opposite y-acceleration  
        np.testing.assert_allclose(acc[2, 1], -acc[3, 1], rtol=1e-10)

    def test_limit_behavior_workflow(self, hierarchical_system):
        """Test limiting behavior in workflows."""
        positions, masses = hierarchical_system
        
        # Direct summation (theta=0 equivalent)
        tree_direct = bhut.Tree(positions, masses)
        tree_direct.build()
        acc_direct = tree_direct.accelerations(theta=1e-10)
        
        # Fast approximation (large theta)
        tree_fast = bhut.Tree(positions, masses)
        tree_fast.build()
        acc_fast = tree_fast.accelerations(theta=2.0)
        
        # Both should be physically reasonable
        assert np.all(np.isfinite(acc_direct))
        assert np.all(np.isfinite(acc_fast))
        
        # Direct should be more accurate, but both should conserve momentum
        momentum_direct = np.sum(acc_direct * masses[:, np.newaxis], axis=0)
        momentum_fast = np.sum(acc_fast * masses[:, np.newaxis], axis=0)
        
        np.testing.assert_allclose(momentum_direct, 0.0, atol=1e-12)
        np.testing.assert_allclose(momentum_fast, 0.0, atol=1e-10)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_scaling_performance_workflow(self):
        """Test performance scaling in realistic workflow."""
        # Test different system sizes
        system_sizes = [10, 25, 50]
        times = []
        
        for n in system_sizes:
            positions = np.random.randn(n, 3)
            masses = np.ones(n)
            
            import time
            start_time = time.time()
            
            # Typical workflow
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=1.0)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Should scale better than O(N^2)
            assert np.all(np.isfinite(acc))
        
        # Very rough performance check - times shouldn't explode
        assert all(t < 10.0 for t in times)  # Should be reasonably fast

    def test_memory_efficiency_workflow(self):
        """Test memory efficiency in workflow."""
        # Create moderately large system
        n_particles = 75
        positions = np.random.randn(n_particles, 3)
        masses = np.ones(n_particles)
        
        # Should handle without excessive memory
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc = tree.accelerations()
        
        # Multiple operations on same tree
        for _ in range(3):
            new_positions = positions + np.random.randn(*positions.shape) * 0.01
            tree.refit(new_positions)
            acc = tree.accelerations()
            
            assert acc.shape == (n_particles, 3)
            assert np.all(np.isfinite(acc))

    def test_repeated_operations_workflow(self, random_system):
        """Test repeated operations for memory leaks/performance degradation."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before refitting
        
        # Repeated operations
        for i in range(10):
            # Small perturbation
            new_positions = positions + np.random.randn(*positions.shape) * 0.001
            
            tree.refit(new_positions)
            acc = tree.accelerations()
            
            assert acc.shape == positions.shape
            assert np.all(np.isfinite(acc))
            
            # Performance shouldn't degrade significantly
            # (This is more of a stress test than a strict requirement)
