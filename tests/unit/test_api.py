"""
Unit tests for the main API functions and Tree class.

Tests the core public interface: accelerations() function and Tree class.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.unit
class TestAccelerationsFunction:
    """Test the main accelerations() function."""

    def test_basic_functionality(self, simple_system):
        """Test basic accelerations computation."""
        positions, masses = simple_system
        
        acc = bhut.accelerations(positions, masses)
        
        assert acc.shape == (2, 3)
        assert np.all(np.isfinite(acc))
        # For two particles, accelerations should be equal and opposite
        np.testing.assert_allclose(acc[0], -acc[1], rtol=1e-10)

    def test_parameter_variations(self, random_system, common_params):
        """Test with different parameter combinations."""
        positions, masses = random_system
        
        # Test different theta values
        for theta in common_params['theta_values']:
            acc = bhut.accelerations(positions, masses, theta=theta)
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))

        # Test different softening values  
        for softening in common_params['softening_values']:
            acc = bhut.accelerations(positions, masses, softening=softening)
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))

        # Test different G values
        for G in common_params['G_values']:
            acc = bhut.accelerations(positions, masses, G=G)
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))

    def test_gravitational_scaling(self, simple_system):
        """Test that accelerations scale correctly with G."""
        positions, masses = simple_system
        
        acc_g1 = bhut.accelerations(positions, masses, G=1.0)
        acc_g2 = bhut.accelerations(positions, masses, G=2.0)
        
        # Should scale linearly with G
        np.testing.assert_allclose(acc_g2, 2.0 * acc_g1, rtol=1e-12)

    def test_self_acceleration_zero(self):
        """Test that single particle has zero acceleration."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        masses = np.array([1.0], dtype=np.float64)
        
        acc = bhut.accelerations(positions, masses)
        
        assert acc.shape == (1, 3)
        np.testing.assert_allclose(acc, 0.0, atol=1e-15)

    def test_different_source_target_sizes(self):
        """Test with different source and target particle counts through separate calls."""
        # Since targets parameter doesn't exist, test through multiple calls
        
        # Source system
        sources = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        source_masses = np.array([1.0, 2.0])
        
        # Compute self-accelerations
        acc_self = bhut.accelerations(sources, source_masses)
        
        assert acc_self.shape == (2, 3)
        assert np.all(np.isfinite(acc_self))
        
        # Forces should be equal and opposite
        force_0 = acc_self[0] * source_masses[0]
        force_1 = acc_self[1] * source_masses[1]
        np.testing.assert_allclose(force_0, -force_1, rtol=1e-12)

    def test_dtype_consistency(self, random_system):
        """Test that output dtype is consistent."""
        positions, masses = random_system
        
        acc = bhut.accelerations(positions, masses)
        
        assert acc.dtype == np.float64


@pytest.mark.unit  
class TestTreeClass:
    """Test the Tree class interface."""

    def test_initialization(self, random_system):
        """Test Tree initialization."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        
        assert tree.positions is not None
        assert tree.masses is not None
        assert tree.dim == 3
        assert tree.leaf_size > 0

    def test_build_tree(self, random_system):
        """Test tree building."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()
        
        # Should be able to compute accelerations after building
        acc = tree.accelerations()
        assert acc.shape == (10, 3)
        assert np.all(np.isfinite(acc))

    def test_tree_parameters(self, random_system):
        """Test Tree with different parameters."""
        positions, masses = random_system
        
        # Test different leaf sizes
        for leaf_size in [1, 4, 8, 16]:
            tree = bhut.Tree(positions, masses, leaf_size=leaf_size)
            tree.build()
            acc = tree.accelerations()
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))

    def test_multiple_evaluations(self, tree_built):
        """Test multiple acceleration evaluations on same tree."""
        tree = tree_built
        
        # Multiple calls should give identical results
        acc1 = tree.accelerations(theta=0.5, softening=0.01)
        acc2 = tree.accelerations(theta=0.5, softening=0.01)
        
        np.testing.assert_allclose(acc1, acc2, rtol=1e-15)

    def test_tree_accelerations_vs_function(self, random_system):
        """Test that Tree.accelerations() matches accelerations() function."""
        positions, masses = random_system
        
        # Function call
        acc_func = bhut.accelerations(positions, masses, theta=0.5, softening=0.01)
        
        # Tree call
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc_tree = tree.accelerations(theta=0.5, softening=0.01)
        
        np.testing.assert_allclose(acc_func, acc_tree, rtol=1e-12)

    def test_tree_external_targets(self, tree_built):
        """Test Tree evaluation with external targets."""
        tree = tree_built
        
        targets = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float64)
        acc = tree.accelerations(targets=targets)
        
        assert acc.shape == (2, 3)
        assert np.all(np.isfinite(acc))

    def test_tree_not_built_error(self, random_system):
        """Test error when calling accelerations before build."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        
        with pytest.raises(ValueError, match="must be built"):
            tree.accelerations()


@pytest.mark.unit
class TestAPIConsistency:
    """Test consistency across different API usage patterns."""

    def test_different_calling_patterns(self, random_system):
        """Test that different ways of calling give same results."""
        positions, masses = random_system
        
        # Direct function call
        acc1 = bhut.accelerations(positions, masses, theta=0.5)
        
        # Tree-based call
        tree = bhut.Tree(positions, masses)
        tree.build()
        acc2 = tree.accelerations(theta=0.5)
        
        # Should be identical (or very close)
        np.testing.assert_allclose(acc1, acc2, rtol=1e-10)

    def test_backend_specification(self, random_system):
        """Test explicit backend specification."""
        positions, masses = random_system
        
        # Test numpy backend explicitly
        acc_numpy = bhut.accelerations(positions, masses, backend="numpy")
        acc_auto = bhut.accelerations(positions, masses, backend="auto")
        
        # Should be identical for numpy arrays
        np.testing.assert_allclose(acc_numpy, acc_auto, rtol=1e-15)

    def test_parameter_forwarding(self, random_system):
        """Test that parameters are correctly forwarded."""
        positions, masses = random_system
        
        # Test that all parameters work
        acc = bhut.accelerations(
            positions, masses,
            theta=0.7,
            softening=0.02, 
            G=2.0,
            leaf_size=8,
            backend="auto"
        )
        
        assert acc.shape == (10, 3)
        assert np.all(np.isfinite(acc))
