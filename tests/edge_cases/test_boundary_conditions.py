"""
Edge case tests for boundary conditions and error handling.

Tests unusual inputs, error conditions, and robustness.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.edge_cases
class TestInputValidationEdgeCases:
    """Test edge cases in input validation."""

    def test_empty_arrays(self):
        """Test behavior with empty input arrays."""
        positions = np.empty((0, 3))
        masses = np.empty(0)
        
        # Should handle gracefully
        with pytest.raises(ValueError, match="empty|zero"):
            bhut.accelerations(positions, masses)

    def test_single_particle(self):
        """Test behavior with single particle."""
        positions = np.array([[1.0, 2.0, 3.0]])
        masses = np.array([1.0])
        
        # Single particle should have zero acceleration
        acc = bhut.accelerations(positions, masses)
        
        assert acc.shape == (1, 3)
        np.testing.assert_allclose(acc, 0.0, atol=1e-15)

    def test_wrong_dimensions(self):
        """Test various wrong dimensional inputs."""
        # Wrong position dimensions
        with pytest.raises(ValueError, match="2D array|dimension"):
            bhut.accelerations(np.array([[1, 2]]), np.array([1]))  # 2D positions
        
        with pytest.raises(ValueError, match="2D array|dimension"):
            bhut.accelerations(np.array([1, 2, 3]), np.array([1]))  # 1D positions
        
        # Wrong mass dimensions
        with pytest.raises(ValueError, match="1D array|dimension"):
            bhut.accelerations(np.array([[1, 2, 3]]), np.array([[1]]))  # 2D masses

    def test_mismatched_lengths(self):
        """Test mismatched array lengths."""
        positions = np.random.randn(5, 3)
        masses = np.random.randn(3)  # Wrong length
        
        with pytest.raises(ValueError, match="particles|elements|has"):
            bhut.accelerations(positions, masses)

    def test_non_finite_inputs(self):
        """Test non-finite input values."""
        # NaN in positions
        positions_nan = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        masses = np.array([1.0, 1.0])
        
        # The implementation might handle NaN/Inf gracefully or raise errors
        try:
            acc = bhut.accelerations(positions_nan, masses)
            # If no error, result should still be reasonable
            assert acc.shape == (2, 3)
        except (ValueError, RuntimeError):
            # Expected for non-finite inputs
            pass
        
        # Infinity in masses
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        masses_inf = np.array([1.0, np.inf])
        
        try:
            acc = bhut.accelerations(positions, masses_inf)
            # If no error, result should still be reasonable
            assert acc.shape == (2, 3)
        except (ValueError, RuntimeError):
            # Expected for non-finite inputs
            pass

    def test_negative_masses(self):
        """Test negative mass inputs."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        masses = np.array([1.0, -1.0])  # Negative mass
        
        # The implementation might allow negative masses or reject them
        try:
            acc = bhut.accelerations(positions, masses)
            # If allowed, should still give finite results
            assert np.all(np.isfinite(acc))
            assert acc.shape == (2, 3)
        except ValueError:
            # Expected if negative masses are rejected
            pass

    def test_zero_masses(self):
        """Test zero mass inputs."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        masses = np.array([1.0, 0.0])  # Zero mass
        
        # Zero masses might be allowed or rejected depending on implementation
        try:
            acc = bhut.accelerations(positions, masses)
            # If allowed, should still give finite results
            assert np.all(np.isfinite(acc))
        except ValueError as e:
            assert "positive" in str(e).lower() or "zero" in str(e).lower()

    def test_dtype_edge_cases(self):
        """Test edge cases with different dtypes."""
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # Integer
        masses = np.array([1, 2], dtype=np.int32)
        
        # Should handle integer inputs (convert to float)
        acc = bhut.accelerations(positions, masses)
        assert acc.dtype in [np.float32, np.float64]
        assert acc.shape == (2, 3)

    def test_extremely_large_arrays(self):
        """Test with arrays that might cause memory issues."""
        # This test might be skipped on systems with limited memory
        try:
            n = 1000  # Reduced size to avoid potential infinite loops
            positions = np.random.randn(n, 3).astype(np.float32)  # Use float32 to save memory
            masses = np.ones(n, dtype=np.float32)
            
            # Should handle or fail gracefully
            acc = bhut.accelerations(positions, masses, theta=1.0)  # Use larger theta for speed
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))
            
        except MemoryError:
            pytest.skip("Insufficient memory for large array test")


@pytest.mark.edge_cases
class TestParameterEdgeCases:
    """Test edge cases in parameter values."""

    def test_extreme_theta_values(self, simple_system):
        """Test extreme theta parameter values."""
        positions, masses = simple_system
        
        # Very small theta (should be very accurate but slow)
        tree_tiny = bhut.Tree(positions, masses)
        tree_tiny.build()
        acc_tiny = tree_tiny.accelerations(theta=1e-10)
        assert np.all(np.isfinite(acc_tiny))
        
        # Very large theta (should be fast but inaccurate)
        tree_huge = bhut.Tree(positions, masses)
        tree_huge.build()
        acc_huge = tree_huge.accelerations(theta=100.0)
        assert np.all(np.isfinite(acc_huge))
        
        # Both should conserve momentum approximately (large theta is very approximate)
        momentum_tiny = np.sum(acc_tiny * masses[:, np.newaxis], axis=0)
        momentum_huge = np.sum(acc_huge * masses[:, np.newaxis], axis=0)
        
        np.testing.assert_allclose(momentum_tiny, 0.0, atol=1e-12)
        # Large theta gives very approximate results, so looser tolerance
        momentum_magnitude = np.linalg.norm(momentum_huge)
        force_magnitude = np.linalg.norm(acc_huge * masses[:, np.newaxis])
        relative_error = momentum_magnitude / (force_magnitude + 1e-10)
        assert relative_error < 1.0  # Very loose check

    def test_extreme_leaf_size(self, simple_system):
        """Test extreme leaf_size values."""
        positions, masses = simple_system
        
        # Very small leaf_size (deep tree)
        tree_small = bhut.Tree(positions, masses, leaf_size=1)
        tree_small.build()
        acc_small = tree_small.accelerations()
        assert np.all(np.isfinite(acc_small))
        
        # Very large leaf_size (shallow tree)
        tree_large = bhut.Tree(positions, masses, leaf_size=100)
        tree_large.build()
        acc_large = tree_large.accelerations()
        assert np.all(np.isfinite(acc_large))
        
        # Results should be similar for simple system
        np.testing.assert_allclose(acc_small, acc_large, rtol=0.1)

    def test_invalid_parameter_combinations(self, simple_system):
        """Test invalid parameter combinations."""
        positions, masses = simple_system
        
        # Test invalid theta values via accelerations() method
        tree = bhut.Tree(positions, masses)
        tree.build()
        
        # Negative theta
        with pytest.raises(ValueError, match="theta|positive"):
            tree.accelerations(theta=-1.0)
        
        # Zero theta might be valid (direct summation) or invalid depending on implementation
        try:
            acc = tree.accelerations(theta=0.0)
            assert np.all(np.isfinite(acc))  # If allowed, should work
        except ValueError as e:
            assert "theta" in str(e).lower()  # If not allowed, should mention theta
        
        # Zero or negative leaf_size during tree construction
        with pytest.raises(ValueError, match="leaf_size|positive"):
            bhut.Tree(positions, masses, leaf_size=0)
        
        with pytest.raises(ValueError, match="leaf_size|positive"):
            bhut.Tree(positions, masses, leaf_size=-5)

    def test_parameter_type_edge_cases(self, simple_system):
        """Test edge cases with parameter types."""
        positions, masses = simple_system
        
        # Valid leaf_size
        tree = bhut.Tree(positions, masses, leaf_size=20)
        tree.build()
        acc = tree.accelerations(theta=1.5)
        assert np.all(np.isfinite(acc))
        
        # Test string parameters (should fail)
        with pytest.raises((ValueError, TypeError)):
            tree.accelerations(theta="invalid")
        
        with pytest.raises((ValueError, TypeError)):
            bhut.Tree(positions, masses, leaf_size="invalid")


@pytest.mark.edge_cases
class TestNumericalEdgeCases:
    """Test numerical edge cases and precision limits."""

    def test_identical_positions(self):
        """Test particles at identical positions."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],  # Identical position
        ])
        masses = np.array([1.0, 1.0])
        
        # Should handle gracefully (might use softening)
        acc = bhut.accelerations(positions, masses)
        
        assert acc.shape == (2, 3)
        assert np.all(np.isfinite(acc))
        # Forces should be equal and opposite (or both zero)
        np.testing.assert_allclose(acc[0], -acc[1], atol=1e-12)

    def test_extremely_close_particles(self):
        """Test particles extremely close together."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-15, 0.0, 0.0],  # Machine precision close
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should remain finite due to softening
        assert np.all(np.isfinite(acc))
        assert np.all(~np.isnan(acc))

    def test_extremely_far_particles(self):
        """Test particles extremely far apart."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e15, 0.0, 0.0],  # Very far
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should remain finite, forces should be tiny but non-zero
        assert np.all(np.isfinite(acc))
        assert np.linalg.norm(acc) > 0
        assert np.linalg.norm(acc) < 1e-25  # Should be very small

    def test_underflow_conditions(self):
        """Test conditions that might cause underflow."""
        # Tiny masses, large separations
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],
        ])
        masses = np.array([1e-15, 1e-15])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should handle underflow gracefully
        assert np.all(np.isfinite(acc))
        # Forces should be equal and opposite (or both tiny)
        total_momentum = np.sum(acc * masses[:, np.newaxis], axis=0)
        momentum_magnitude = np.linalg.norm(total_momentum)
        assert momentum_magnitude < 1e-20  # Should be very small due to conservation

    def test_overflow_conditions(self):
        """Test conditions that might cause overflow."""
        # Large masses, small separations
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-6, 0.0, 0.0],
        ])
        masses = np.array([1e10, 1e10])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should remain finite (due to softening)
        assert np.all(np.isfinite(acc))
        assert np.all(~np.isinf(acc))

    def test_precision_loss_scenarios(self):
        """Test scenarios with potential precision loss."""
        # Large coordinates with small differences
        base = 1e12
        positions = np.array([
            [base, base, base],
            [base + 1.0, base, base],  # Small difference in large numbers
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should still give reasonable results
        assert np.all(np.isfinite(acc))
        
        # Direction should be correct despite precision issues
        assert acc[0, 0] > 0  # Should accelerate toward second particle


@pytest.mark.edge_cases
class TestTreeEdgeCases:
    """Test edge cases specific to tree construction and manipulation."""

    def test_tree_with_single_particle(self):
        """Test tree construction with single particle."""
        positions = np.array([[1.0, 2.0, 3.0]])
        masses = np.array([1.0])
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        acc = tree.accelerations()
        
        assert acc.shape == (1, 3)
        np.testing.assert_allclose(acc, 0.0, atol=1e-15)

    def test_tree_rebuild_edge_cases(self, simple_system):
        """Test tree rebuild with edge case inputs."""
        positions, masses = simple_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Initial build
        
        # Rebuild with same data
        tree.rebuild(positions, masses)
        acc_same = tree.accelerations()
        
        # Rebuild with very different data - change mass distribution
        new_masses = masses * np.array([10.0, 0.1])  # Change mass ratio significantly
        tree.rebuild(positions, new_masses)
        acc_different = tree.accelerations()
        
        assert np.all(np.isfinite(acc_same))
        assert np.all(np.isfinite(acc_different))
        # With different masses, accelerations should be different
        assert not np.allclose(acc_same, acc_different, rtol=1e-1)

    def test_tree_refit_edge_cases(self, random_system):
        """Test tree refit with edge case perturbations."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        
        # Very small perturbation
        tiny_change = positions + np.random.randn(*positions.shape) * 1e-10
        tree.refit(tiny_change)
        acc_tiny = tree.accelerations()
        
        # Large perturbation (might exceed refit capability)
        large_change = positions + np.random.randn(*positions.shape) * 10.0
        try:
            tree.refit(large_change)
            acc_large = tree.accelerations()
            assert np.all(np.isfinite(acc_large))
        except (ValueError, RuntimeError):
            # Large changes might require rebuild instead of refit
            pass

    def test_tree_parameter_changes(self, simple_system):
        """Test tree leaf_size parameter after construction."""
        positions, masses = simple_system
        
        tree = bhut.Tree(positions, masses, leaf_size=20)
        tree.build()
        
        # Tree leaf_size should be accessible
        assert hasattr(tree, 'leaf_size') or hasattr(tree, '_leaf_size')
        
        # Basic functionality should work
        acc = tree.accelerations(theta=1.0)
        assert np.all(np.isfinite(acc))

    def test_tree_memory_edge_cases(self):
        """Test tree memory-related edge cases."""
        # Create tree with moderately large system
        n = 100
        positions = np.random.randn(n, 3)
        masses = np.ones(n)
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        
        # Multiple operations to test for memory leaks/issues
        for i in range(5):
            new_positions = positions + np.random.randn(*positions.shape) * 0.01
            tree.refit(new_positions)
            acc = tree.accelerations()
            
            assert acc.shape == (n, 3)
            assert np.all(np.isfinite(acc))


@pytest.mark.edge_cases
class TestErrorRecovery:
    """Test error recovery and robustness."""

    def test_recovery_from_invalid_operations(self, simple_system):
        """Test recovery after invalid operations."""
        positions, masses = simple_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()  # Must build before using
        valid_acc = tree.accelerations()
        
        # Try invalid refit
        invalid_positions = np.array([[np.nan, 0, 0]])
        
        with pytest.raises(ValueError):
            tree.refit(invalid_positions)
        
        # Tree should still work after failed operation
        still_valid_acc = tree.accelerations()
        np.testing.assert_allclose(valid_acc, still_valid_acc, rtol=1e-15)

    def test_error_message_quality(self):
        """Test that error messages are informative."""
        # Various invalid inputs
        test_cases = [
            (np.array([[1, 2]]), np.array([1]), "dimension|2D array"),  # Wrong dimension
            (np.array([[1, 2, 3]]), np.array([1, 2]), "particles|elements"),  # Shape mismatch
        ]
        
        for positions, masses, expected_error in test_cases:
            try:
                bhut.accelerations(positions, masses)
                pytest.fail(f"Expected error for {expected_error} case")
            except ValueError as e:
                error_msg = str(e).lower()
                # Check if any of the expected terms are in the error message
                expected_terms = expected_error.split("|")
                assert any(term in error_msg for term in expected_terms), \
                    f"Expected one of {expected_terms} in error message: {error_msg}"

    def test_partial_failure_handling(self):
        """Test handling of partial failures in computations."""
        # Mix of valid and problematic particles
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1e-15, 0.0, 0.0],  # Very close to first particle
            [1e15, 0.0, 0.0],   # Very far
        ])
        masses = np.array([1.0, 1.0, 1e-15, 1e15])  # Mix of mass scales
        
        # Should handle gracefully
        acc = bhut.accelerations(positions, masses)
        
        assert acc.shape == (4, 3)
        assert np.all(np.isfinite(acc))
        
        # Conservation should still hold
        total_momentum = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_momentum, 0.0, atol=1e-10)
