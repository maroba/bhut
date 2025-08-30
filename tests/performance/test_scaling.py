"""
Performance tests for scaling behavior and computational efficiency.

Tests timing, memory usage, and scaling properties.
"""

import time
import numpy as np
import pytest

import bhut


@pytest.mark.performance
@pytest.mark.slow
class TestScalingPerformance:
    """Test performance scaling with system size."""

    def test_acceleration_scaling(self):
        """Test that acceleration computation scales better than O(N²)."""
        system_sizes = [10, 25, 50, 100]
        times = []
        
        for n in system_sizes:
            positions = np.random.randn(n, 3).astype(np.float64)
            masses = np.abs(np.random.randn(n)) + 0.1
            
            # Warm up
            bhut.accelerations(positions, masses)
            
            # Time the computation
            start_time = time.time()
            for _ in range(3):  # Multiple runs for better timing
                acc = bhut.accelerations(positions, masses)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 3
            times.append(avg_time)
            
            # Verify correctness
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))
        
        # Check scaling - should be better than O(N²)
        # For Barnes-Hut, expect roughly O(N log N)
        for i in range(1, len(system_sizes)):
            n_ratio = system_sizes[i] / system_sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Should scale better than quadratic
            quadratic_scaling = n_ratio**2
            expected_scaling = n_ratio * np.log(n_ratio)  # Rough O(N log N)
            
            # Allow some tolerance for timing variability
            assert time_ratio < quadratic_scaling * 1.5, f"Scaling worse than quadratic at N={system_sizes[i]}"

    def test_tree_construction_scaling(self):
        """Test tree construction performance scaling."""
        system_sizes = [20, 50, 100]
        times = []
        
        for n in system_sizes:
            positions = np.random.randn(n, 3)
            masses = np.ones(n)
            
            # Time tree construction
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()  # Must build before using
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify tree is functional
            acc = tree.accelerations()
            assert acc.shape == (n, 3)
        
        # Tree construction should scale reasonably
        # (exact scaling depends on implementation)
        for i in range(len(times)):
            assert times[i] < 10.0, f"Tree construction too slow for N={system_sizes[i]}"

    def test_theta_parameter_performance(self):
        """Test performance vs accuracy tradeoff with theta."""
        n = 50
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        theta_values = [0.1, 0.5, 1.0, 2.0]
        times = []
        accuracies = []
        
        # Reference computation (most accurate)
        tree_ref = bhut.Tree(positions, masses)
        tree_ref.build()
        acc_ref = tree_ref.accelerations(theta=0.01)
        
        for theta in theta_values:
            # Time the computation
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=theta)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Compute accuracy (relative error)
            error = np.linalg.norm(acc - acc_ref) / np.linalg.norm(acc_ref)
            accuracies.append(error)
        
        # Larger theta should generally be faster
        for i in range(1, len(theta_values)):
            if theta_values[i] > theta_values[i-1]:
                # Allow some tolerance for timing noise
                assert times[i] <= times[i-1] * 2, f"Performance not improving with larger theta"
        
        # Accuracy should decrease with larger theta
        for i in range(1, len(theta_values)):
            if theta_values[i] > theta_values[i-1]:
                assert accuracies[i] >= accuracies[i-1] * 0.5, f"Accuracy relationship unexpected"

    def test_repeated_computation_performance(self):
        """Test performance of repeated computations on same tree."""
        n = 50
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        
        # Time repeated acceleration computations
        n_repeats = 10
        start_time = time.time()
        
        for _ in range(n_repeats):
            acc = tree.accelerations()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_repeats
        
        # Repeated computations should be fast (tree already built)
        assert avg_time < 1.0, "Repeated acceleration computation too slow"
        assert np.all(np.isfinite(acc))


@pytest.mark.performance  
class TestMemoryPerformance:
    """Test memory usage and efficiency."""

    def test_memory_scaling(self):
        """Test memory usage scales reasonably with system size."""
        import gc
        import sys
        
        # This is a rough test - exact memory measurement is platform-dependent
        system_sizes = [25, 50, 100]
        
        for n in system_sizes:
            # Force garbage collection before test
            gc.collect()
            
            positions = np.random.randn(n, 3)
            masses = np.ones(n)
            
            # Create tree and compute accelerations
            tree = bhut.Tree(positions, masses)
            tree.build()  # Must build before using
            acc = tree.accelerations()
            
            # Should complete without memory errors
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))
            
            # Clean up
            del tree, acc
            gc.collect()

    def test_tree_reuse_efficiency(self):
        """Test efficiency of tree reuse vs reconstruction."""
        n = 30
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        # Time tree reconstruction
        reconstruct_times = []
        for _ in range(5):
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()  # Must build before using
            acc = tree.accelerations()
            end_time = time.time()
            reconstruct_times.append(end_time - start_time)
        
        # Time tree refit
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build initial tree
        refit_times = []
        
        for _ in range(5):
            new_positions = positions + np.random.randn(*positions.shape) * 0.01
            
            start_time = time.time()
            tree.refit(new_positions)
            acc = tree.accelerations()
            end_time = time.time()
            refit_times.append(end_time - start_time)
        
        # Refit should generally be faster than reconstruction
        avg_reconstruct = np.mean(reconstruct_times)
        avg_refit = np.mean(refit_times)
        
        # Allow some tolerance - refit might not always be faster for small systems
        assert avg_refit <= avg_reconstruct * 1.5, "Refit not showing expected efficiency gain"

    def test_large_system_handling(self):
        """Test handling of moderately large systems."""
        # Test with larger system (but not huge to avoid CI issues)
        n = 200
        
        try:
            positions = np.random.randn(n, 3).astype(np.float64)
            masses = np.ones(n, dtype=np.float64)
            
            # Should handle without excessive memory or time
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()  # Must build before using
            acc = tree.accelerations(theta=1.0)
            end_time = time.time()
            
            # Verify correctness
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))
            
            # Should complete in reasonable time
            assert end_time - start_time < 30.0, "Large system computation too slow"
            
            # Momentum conservation check - with theta=1.0, conservation is approximate
            total_momentum = np.sum(acc * masses[:, np.newaxis], axis=0)
            momentum_magnitude = np.linalg.norm(total_momentum)
            force_magnitude = np.linalg.norm(acc * masses[:, np.newaxis])
            
            # Momentum conservation should be reasonable relative to force scale
            relative_momentum_error = momentum_magnitude / (force_magnitude + 1e-10)
            assert relative_momentum_error < 0.1, f"Momentum conservation too poor: {relative_momentum_error}"
            
        except MemoryError:
            pytest.skip("Insufficient memory for large system test")


@pytest.mark.performance
class TestAlgorithmicEfficiency:
    """Test algorithmic efficiency and optimization."""

    def test_theta_efficiency_tradeoff(self):
        """Test efficiency vs accuracy tradeoff systematically."""
        n = 40
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        # Reference (direct summation equivalent)
        tree_ref = bhut.Tree(positions, masses)
        tree_ref.build()
        acc_ref = tree_ref.accelerations(theta=1e-8)
        
        # Test different theta values
        theta_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        results = []
        
        for theta in theta_values:
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=theta)
            end_time = time.time()
            
            # Calculate relative error
            error = np.linalg.norm(acc - acc_ref) / np.linalg.norm(acc_ref)
            
            results.append({
                'theta': theta,
                'time': end_time - start_time,
                'error': error,
                'acc': acc
            })
        
        # Verify that larger theta generally means faster computation
        # and higher error (though relationship might not be monotonic)
        for i in range(len(results)):
            assert results[i]['error'] < 1.0, f"Error too large for theta={results[i]['theta']}"
            assert results[i]['time'] < 10.0, f"Computation too slow for theta={results[i]['theta']}"

    def test_depth_efficiency(self):
        """Test leaf_size parameter efficiency."""
        n = 30
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        theta = 1.0  # Fixed theta
        leaf_sizes = [4, 8, 16, 32]  # Use valid leaf_size parameter
        
        for leaf_size in leaf_sizes:
            start_time = time.time()
            tree = bhut.Tree(positions, masses, leaf_size=leaf_size)
            tree.build()
            acc = tree.accelerations(theta=theta)
            end_time = time.time()
            
            # Should complete reasonably fast
            assert end_time - start_time < 5.0, f"Too slow for leaf_size={leaf_size}"
            
            # Should give valid results
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))

    def test_particle_distribution_efficiency(self):
        """Test efficiency with different particle distributions."""
        n = 50
        distributions = {
            'random': np.random.randn(n, 3),
            'clustered': np.concatenate([
                np.random.randn(n//2, 3) * 0.1,  # Tight cluster
                np.random.randn(n//2, 3) * 0.1 + [5, 0, 0]  # Another cluster
            ]),
            'uniform': np.random.uniform(-2, 2, (n, 3)),
        }
        
        masses = np.ones(n)
        
        for dist_name, positions in distributions.items():
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=1.0)
            end_time = time.time()
            
            # Should handle all distributions reasonably
            assert end_time - start_time < 5.0, f"Too slow for {dist_name} distribution"
            assert np.all(np.isfinite(acc))
            
            # Conservation check
            total_momentum = np.sum(acc * masses[:, np.newaxis], axis=0)
            np.testing.assert_allclose(total_momentum, 0.0, atol=1e-10)


@pytest.mark.performance
@pytest.mark.slow
class TestStressPerformance:
    """Stress tests for performance under difficult conditions."""

    def test_extreme_mass_ratio_performance(self):
        """Test performance with extreme mass ratios."""
        n = 30
        positions = np.random.randn(n, 3)
        
        # Create extreme mass ratios
        masses = np.ones(n)
        masses[0] = 1e12  # Very massive
        masses[1] = 1e-12  # Very light
        
        start_time = time.time()
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        acc = tree.accelerations()
        end_time = time.time()
        
        # Should still complete in reasonable time
        assert end_time - start_time < 10.0, "Too slow with extreme mass ratios"
        assert np.all(np.isfinite(acc))

    def test_close_encounter_performance(self):
        """Test performance with very close particle encounters."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-8, 0.0, 0.0],  # Very close
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] + [[np.random.randn(), np.random.randn(), np.random.randn()] for _ in range(15)])
        
        masses = np.ones(len(positions))
        
        start_time = time.time()
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        acc = tree.accelerations()
        end_time = time.time()
        
        # Should handle close encounters without excessive slowdown
        assert end_time - start_time < 5.0, "Too slow with close encounters"
        assert np.all(np.isfinite(acc))

    def test_repeated_refit_performance(self):
        """Test performance of many repeated refits."""
        n = 25
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build initial tree
        
        # Many small refits
        n_refits = 20
        start_time = time.time()
        
        current_positions = positions.copy()
        for i in range(n_refits):
            # Small perturbation
            perturbation = np.random.randn(*positions.shape) * 0.001
            current_positions += perturbation
            
            tree.refit(current_positions)
            acc = tree.accelerations()
            
            assert np.all(np.isfinite(acc))
        
        end_time = time.time()
        avg_refit_time = (end_time - start_time) / n_refits
        
        # Average refit should be fast
        assert avg_refit_time < 1.0, "Average refit time too slow"

    def test_parameter_sweep_performance(self):
        """Test performance across parameter sweeps."""
        n = 30
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        # Sweep theta and leaf_size instead of max_depth
        theta_values = [0.5, 1.0, 1.5]
        leaf_size_values = [8, 16, 32]  # Use valid leaf_size parameter
        
        total_time = 0
        n_tests = 0
        
        for theta in theta_values:
            for leaf_size in leaf_size_values:
                start_time = time.time()
                tree = bhut.Tree(positions, masses, leaf_size=leaf_size)
                tree.build()
                acc = tree.accelerations(theta=theta)
                end_time = time.time()
                
                total_time += end_time - start_time
                n_tests += 1
                
                assert np.all(np.isfinite(acc))
        
        avg_time = total_time / n_tests
        assert avg_time < 2.0, "Average parameter sweep computation too slow"


@pytest.mark.performance
class TestBenchmarkComparisons:
    """Benchmark tests for comparison purposes."""

    def test_direct_vs_tree_performance(self):
        """Compare direct summation vs tree algorithm performance."""
        # Small system where direct summation is feasible
        n = 15
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        # Time tree algorithm
        start_time = time.time()
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc_tree = tree.accelerations(theta=1.0)
        tree_time = time.time() - start_time
        
        # Manual direct summation for comparison
        start_time = time.time()
        acc_direct = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    acc_direct[i] += masses[j] * r_vec / (r_mag**3 + 1e-10)  # Small softening
        direct_time = time.time() - start_time
        
        # For small systems, tree might be slower due to overhead
        # But results should be similar
        np.testing.assert_allclose(acc_tree, acc_direct, rtol=0.1)
        
        # Both times should be reasonable
        assert tree_time < 5.0, "Tree algorithm too slow for small system"
        assert direct_time < 5.0, "Direct summation too slow for small system"

    def test_scaling_crossover_point(self):
        """Find approximate crossover point where tree becomes efficient."""
        sizes = [8, 12, 16, 20]
        tree_times = []
        
        for n in sizes:
            positions = np.random.randn(n, 3)
            masses = np.ones(n)
            
            # Tree algorithm timing
            start_time = time.time()
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=1.0)
            tree_time = time.time() - start_time
            
            tree_times.append(tree_time)
            
            # Verify correctness
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))
        
        # Tree algorithm should not show excessive scaling
        for i in range(1, len(sizes)):
            time_ratio = tree_times[i] / tree_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Should scale better than quadratic
            assert time_ratio < size_ratio**2, f"Excessive scaling at N={sizes[i]}"
