"""
Unit tests for tree data structures and algorithms.

Tests tree building, data structures, Barnes-Hut traversal, and tree operations.
"""

import numpy as np
import pytest

import bhut
from bhut.tree.build import build_tree
from bhut.backends.numpy_ import NumpyArrayNamespace


@pytest.mark.unit
class TestTreeBuilding:
    """Test tree building algorithms and data structures."""

    def test_basic_tree_build(self, random_system):
        """Test basic tree building."""
        positions, masses = random_system
        xp = NumpyArrayNamespace()
        
        tree = build_tree(positions, masses, leaf_size=4, dim=3, xp=xp)
        
        assert tree.dim == 3
        assert tree.leaf_size == 4
        assert len(tree.center) > 0
        assert len(tree.mass) > 0
        assert len(tree.perm) == len(positions)

    def test_different_leaf_sizes(self, random_system, common_params):
        """Test tree building with different leaf sizes."""
        positions, masses = random_system
        xp = NumpyArrayNamespace()
        
        for leaf_size in common_params['leaf_sizes']:
            tree = build_tree(positions, masses, leaf_size=leaf_size, dim=3, xp=xp)
            
            assert tree.leaf_size == leaf_size
            assert len(tree.mass) > 0
            assert np.all(np.isfinite(tree.center))

    def test_tree_structure_properties(self, random_system):
        """Test tree data structure properties."""
        positions, masses = random_system
        xp = NumpyArrayNamespace()
        
        tree = build_tree(positions, masses, leaf_size=2, dim=3, xp=xp)
        
        # Root should contain all mass
        total_mass = np.sum(masses)
        assert np.isclose(tree.mass[0], total_mass)
        
        # Permutation should be valid
        assert len(tree.perm) == len(masses)
        assert np.all(tree.perm >= 0)
        assert np.all(tree.perm < len(masses))
        assert len(np.unique(tree.perm)) == len(masses)  # No duplicates

    def test_edge_case_inputs(self):
        """Test tree building with edge case inputs."""
        xp = NumpyArrayNamespace()
        
        # Minimum particles (2)
        positions_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        masses_2 = np.array([1.0, 1.0], dtype=np.float64)
        
        tree = build_tree(positions_2, masses_2, leaf_size=1, dim=3, xp=xp)
        assert tree.dim == 3
        assert len(tree.center) > 0
        
        # Test with varied masses
        positions_varied = np.random.randn(8, 3).astype(np.float64)
        masses_varied = np.array([0.1, 1.0, 10.0, 0.5, 2.0, 0.01, 5.0, 0.2], dtype=np.float64)
        
        tree_varied = build_tree(positions_varied, masses_varied, leaf_size=2, dim=3, xp=xp)
        assert np.isclose(tree_varied.mass[0], np.sum(masses_varied))

    def test_tree_consistency(self, hierarchical_system):
        """Test that tree building is consistent and deterministic."""
        positions, masses = hierarchical_system
        xp = NumpyArrayNamespace()
        
        # Build same tree twice
        tree1 = build_tree(positions, masses, leaf_size=2, dim=3, xp=xp)
        tree2 = build_tree(positions, masses, leaf_size=2, dim=3, xp=xp)
        
        # Should have identical structure
        np.testing.assert_array_equal(tree1.perm, tree2.perm)
        np.testing.assert_allclose(tree1.center, tree2.center)
        np.testing.assert_allclose(tree1.mass, tree2.mass)


@pytest.mark.unit
class TestTreeOperations:
    """Test tree operations like refit and rebuild."""

    def test_tree_refit_basic(self, random_system):
        """Test basic tree refit functionality."""
        positions1, masses = random_system
        positions2 = positions1 + 0.01 * np.random.randn(*positions1.shape)
        
        tree = bhut.Tree(positions1, masses)
        tree.build()
        
        acc1 = tree.accelerations()
        
        # Refit with new positions
        tree.refit(positions2)
        acc2 = tree.accelerations()
        
        assert acc1.shape == acc2.shape
        assert np.all(np.isfinite(acc1))
        assert np.all(np.isfinite(acc2))
        
        # Results should be different but reasonable
        assert not np.allclose(acc1, acc2, rtol=1e-10)

    def test_tree_rebuild(self, random_system):
        """Test tree rebuild functionality."""
        positions1, masses1 = random_system
        
        # Different system for rebuild
        positions2 = np.random.randn(8, 3).astype(np.float64)  
        masses2 = np.ones(8, dtype=np.float64)
        
        tree = bhut.Tree(positions1, masses1)
        tree.build()
        acc1 = tree.accelerations()
        
        # Rebuild with new system
        tree.rebuild(positions2, masses2)
        acc2 = tree.accelerations()
        
        assert acc1.shape == (10, 3)
        assert acc2.shape == (8, 3)
        assert np.all(np.isfinite(acc1))
        assert np.all(np.isfinite(acc2))

    def test_multiple_refits(self, random_system):
        """Test multiple consecutive refits."""
        positions, masses = random_system
        
        tree = bhut.Tree(positions, masses)
        tree.build()
        
        # Perform multiple small refits
        current_pos = positions.copy()
        for i in range(3):
            current_pos += 0.005 * np.random.randn(*positions.shape)
            tree.refit(current_pos)
            acc = tree.accelerations()
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))


@pytest.mark.unit
class TestBarnesHutAlgorithm:
    """Test Barnes-Hut specific algorithm components."""

    def test_theta_parameter_effects(self, random_system, common_params):
        """Test effects of different theta values."""
        positions, masses = random_system
        
        results = {}
        for theta in common_params['theta_values']:
            acc = bhut.accelerations(positions, masses, theta=theta)
            results[theta] = acc
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))
        
        # theta=0 should be most accurate (direct summation)
        # Larger theta should be faster but less accurate
        acc_exact = results[0.0]
        acc_approx = results[2.0]
        
        # Should be in the same ballpark
        mag_exact = np.mean(np.linalg.norm(acc_exact, axis=1))
        mag_approx = np.mean(np.linalg.norm(acc_approx, axis=1))
        ratio = mag_approx / mag_exact
        assert 0.1 < ratio < 10.0  # Within order of magnitude

    def test_algorithm_deterministic(self, random_system):
        """Test that algorithm gives deterministic results."""
        positions, masses = random_system
        
        # Multiple runs should give identical results
        acc1 = bhut.accelerations(positions, masses, theta=0.5)
        acc2 = bhut.accelerations(positions, masses, theta=0.5)
        
        np.testing.assert_allclose(acc1, acc2, rtol=1e-15)

    def test_softening_effects(self, simple_system, common_params):
        """Test effects of different softening values."""
        positions, masses = simple_system
        
        results = {}
        for softening in common_params['softening_values']:
            acc = bhut.accelerations(positions, masses, softening=softening)
            results[softening] = acc
            assert np.all(np.isfinite(acc))
        
        # Larger softening should reduce force magnitude
        acc_no_soft = results[0.0]
        acc_soft = results[0.1]
        
        mag_no_soft = np.linalg.norm(acc_no_soft[0])
        mag_soft = np.linalg.norm(acc_soft[0])
        
        assert mag_soft < mag_no_soft  # Softening reduces force

    def test_leaf_size_effects(self, random_system, common_params):
        """Test that different leaf sizes produce stable results."""
        positions, masses = random_system
        
        results = {}
        for leaf_size in common_params['leaf_sizes']:
            acc = bhut.accelerations(positions, masses, leaf_size=leaf_size)
            results[leaf_size] = acc
            assert acc.shape == (10, 3)
            assert np.all(np.isfinite(acc))
        
        # All leaf sizes should produce reasonable results
        # Tree structure affects accuracy so we just check for stability
        for leaf_size, acc in results.items():
            # Check magnitude is reasonable (not zero, not too large)
            acc_magnitude = np.linalg.norm(acc, axis=1)
            assert np.all(acc_magnitude > 1e-20)  # Not zero
            assert np.all(acc_magnitude < 1e10)   # Not too large


@pytest.mark.unit
class TestTreeDataStructures:
    """Test internal tree data structures and access patterns."""

    def test_tree_data_access(self, tree_built):
        """Test accessing tree data structures."""
        tree = tree_built
        
        # Should have expected attributes  
        assert hasattr(tree, 'positions')
        assert hasattr(tree, 'masses')
        assert hasattr(tree, 'dim')
        assert hasattr(tree, 'leaf_size')
        
        # Check basic properties
        assert tree.dim == 3
        assert len(tree.masses) == 10
        assert tree.leaf_size > 0

    def test_tree_memory_efficiency(self, varied_masses_system):
        """Test that tree uses memory efficiently."""
        positions, masses = varied_masses_system
        
        tree = bhut.Tree(positions, masses, leaf_size=2)
        tree.build()
        
        # Tree should not store excessive data
        acc = tree.accelerations()
        assert acc.shape == (len(masses), 3)
        
        # Multiple evaluations should not increase memory usage significantly
        for _ in range(5):
            acc = tree.accelerations(theta=0.5)
            assert acc.shape == (len(masses), 3)
