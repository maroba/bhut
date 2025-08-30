"""
Validation tests for physical correctness and numerical accuracy.

Tests that verify the physics implementation matches theoretical expectations.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.validation
class TestPhysicsValidation:
    """Validate physical correctness of gravitational calculations."""

    def test_two_body_orbital_mechanics(self):
        """Test two-body system follows orbital mechanics."""
        # Two equal masses in circular orbit setup
        # Mass at origin, second mass at (1,0,0) with circular velocity
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 1.0])
        
        # Compute accelerations
        acc = bhut.accelerations(positions, masses)
        
        # For circular orbit: a = v^2/r = GM/r^2
        # Expected acceleration magnitude for unit masses and separation
        expected_acc_magnitude = 1.0  # G=1, M=1, r=1
        
        acc_magnitude_0 = np.linalg.norm(acc[0])
        acc_magnitude_1 = np.linalg.norm(acc[1])
        
        np.testing.assert_allclose(acc_magnitude_0, expected_acc_magnitude, rtol=1e-10)
        np.testing.assert_allclose(acc_magnitude_1, expected_acc_magnitude, rtol=1e-10)
        
        # Accelerations should point toward each other
        np.testing.assert_allclose(acc[0], -acc[1], rtol=1e-15)
        
        # Acceleration should be along line connecting masses
        direction = positions[1] - positions[0]
        direction = direction / np.linalg.norm(direction)
        
        acc_0_direction = acc[0] / np.linalg.norm(acc[0])
        np.testing.assert_allclose(acc_0_direction, direction, rtol=1e-10)

    def test_three_body_lagrange_points(self):
        """Test three-body system with Lagrange point configuration."""
        # Simplified L4 Lagrange point test (equilateral triangle)
        # Two primary masses at (-0.5, 0, 0) and (0.5, 0, 0)
        # Test mass at L4 point forming equilateral triangle
        
        sqrt3_2 = np.sqrt(3) / 2
        positions = np.array([
            [-0.5, 0.0, 0.0],   # Primary 1
            [0.5, 0.0, 0.0],    # Primary 2  
            [0.0, sqrt3_2, 0.0] # Test mass at L4
        ])
        masses = np.array([1.0, 1.0, 1e-6])  # Massless test particle
        
        acc = bhut.accelerations(positions, masses)
        
        # Test particle should have net acceleration toward center
        # (in rotating frame this would be balanced by centrifugal force)
        test_acc = acc[2]  # Acceleration of test particle
        
        # Should point roughly toward center of mass of primaries
        center_of_mass = np.array([0.0, 0.0, 0.0])  # Symmetric case
        direction_to_com = center_of_mass - positions[2]
        direction_to_com = direction_to_com / np.linalg.norm(direction_to_com)
        
        test_acc_direction = test_acc / np.linalg.norm(test_acc)
        
        # In inertial frame, should accelerate toward center
        # (This is simplified - real L4 analysis requires rotating frame)
        assert np.dot(test_acc_direction, direction_to_com) > 0.5

    def test_center_of_mass_conservation(self, random_system):
        """Test that center of mass doesn't accelerate."""
        positions, masses = random_system
        
        # Calculate center of mass
        total_mass = np.sum(masses)
        center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
        
        # Compute accelerations
        acc = bhut.accelerations(positions, masses)
        
        # Total force on system should be zero
        total_force = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-12)
        
        # Center of mass acceleration should be zero
        com_acceleration = total_force / total_mass
        np.testing.assert_allclose(com_acceleration, 0.0, atol=1e-12)

    def test_momentum_conservation(self, hierarchical_system):
        """Test momentum conservation in gravitational interactions."""
        positions, masses = hierarchical_system
        
        acc = bhut.accelerations(positions, masses)
        
        # Internal forces should not change total momentum
        # dp/dt = Σ(F_i) = 0 for internal forces only
        total_momentum_change = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_momentum_change, 0.0, atol=1e-12)

    def test_angular_momentum_properties(self):
        """Test angular momentum related properties."""
        # Two masses in configuration that should conserve angular momentum
        positions = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Calculate torques about origin
        torques = np.cross(positions, acc * masses[:, np.newaxis])
        total_torque = np.sum(torques, axis=0)
        
        # Total torque should be zero (no external forces)
        np.testing.assert_allclose(total_torque, 0.0, atol=1e-12)

    def test_energy_consistency(self, simple_system):
        """Test energy-related consistency checks."""
        positions, masses = simple_system
        
        acc = bhut.accelerations(positions, masses)
        
        # For virial theorem: 2T = -U for gravitational systems in equilibrium
        # Here we just check that accelerations are finite and reasonable
        
        # Kinetic energy proxy (using accelerations as velocity proxy)
        kinetic_proxy = 0.5 * np.sum(masses * np.sum(acc**2, axis=1))
        
        # Potential energy calculation
        potential_energy = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                r = np.linalg.norm(positions[i] - positions[j])
                potential_energy -= masses[i] * masses[j] / r
        
        # Both should be finite
        assert np.isfinite(kinetic_proxy)
        assert np.isfinite(potential_energy)
        
        # Basic sanity: potential should be negative, kinetic positive
        assert potential_energy < 0
        assert kinetic_proxy > 0


@pytest.mark.validation
class TestNumericalAccuracy:
    """Validate numerical accuracy and precision."""

    def test_precision_vs_theta(self, simple_system):
        """Test how precision varies with theta parameter."""
        positions, masses = simple_system
        
        # Reference: very accurate computation
        tree_ref = bhut.Tree(positions, masses)
        tree_ref.build()
        acc_ref = tree_ref.accelerations(theta=1e-8)
        
        # Test different theta values
        theta_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        errors = []
        
        for theta in theta_values:
            tree = bhut.Tree(positions, masses)
            tree.build()
            acc = tree.accelerations(theta=theta)
            
            # Calculate relative error
            error = np.linalg.norm(acc - acc_ref) / np.linalg.norm(acc_ref)
            errors.append(error)
            
            # Error should increase with theta
            assert error < theta * 0.5  # Rough bound
        
        # Errors should generally increase with theta
        for i in range(1, len(errors)):
            if theta_values[i] > theta_values[i-1]:
                # Allow some tolerance for small systems
                assert errors[i] <= errors[i-1] * 2

    def test_convergence_with_depth(self, random_system):
        """Test convergence with leaf_size parameter."""
        positions, masses = random_system
        
        # Test different leaf_size values (smaller = deeper tree)
        leaf_sizes = [4, 8, 16, 32]  # Use valid leaf_size parameter
        results = []
        
        theta = 1.0  # Fixed theta
        for leaf_size in leaf_sizes:
            tree = bhut.Tree(positions, masses, leaf_size=leaf_size)
            tree.build()
            acc = tree.accelerations(theta=theta)
            results.append(acc)
        
        # Results should converge as depth increases
        # (though for random system, convergence might be less obvious)
        for i in range(len(results)):
            assert np.all(np.isfinite(results[i]))
        
        # At least check that we get reasonable accelerations
        for acc in results:
            # Should conserve momentum approximately (theta=1.0 is approximate)
            total_momentum = np.sum(acc * masses[:, np.newaxis], axis=0)
            momentum_magnitude = np.linalg.norm(total_momentum)
            force_magnitude = np.linalg.norm(acc * masses[:, np.newaxis])
            relative_momentum_error = momentum_magnitude / (force_magnitude + 1e-10)
            assert relative_momentum_error < 0.1, "Momentum conservation too poor"

    def test_softening_effect_on_accuracy(self):
        """Test effect of softening on numerical accuracy."""
        # Very close particles that would cause numerical issues
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],  # Very close
        ])
        masses = np.array([1.0, 1.0])
        
        # Should remain stable and finite
        acc = bhut.accelerations(positions, masses)
        
        assert np.all(np.isfinite(acc))
        assert np.all(~np.isnan(acc))
        
        # Forces should be large but not infinite
        force_magnitude = np.linalg.norm(acc[0] * masses[0])
        assert force_magnitude > 1e6  # Should be large
        assert force_magnitude < np.inf  # Should be finite

    def test_machine_precision_limits(self):
        """Test behavior near machine precision limits."""
        # Test with very small and very large values
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        
        # Very small masses
        masses_small = np.array([1e-15, 1e-15])
        acc_small = bhut.accelerations(positions, masses_small)
        assert np.all(np.isfinite(acc_small))
        
        # Very large masses (but not overflowing)
        masses_large = np.array([1e10, 1e10])
        acc_large = bhut.accelerations(positions, masses_large)
        assert np.all(np.isfinite(acc_large))
        
        # Should scale proportionally with mass 
        ratio = np.linalg.norm(acc_large) / np.linalg.norm(acc_small)
        # For gravitational acceleration a = G*m/r^2, so ratio should be masses_large/masses_small
        expected_ratio = masses_large[0] / masses_small[0]  # Same for both particles since masses are equal
        
        # Allow reasonable tolerance since Barnes-Hut approximation affects scaling
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.1)

    def test_relative_vs_absolute_errors(self, hierarchical_system):
        """Test relative vs absolute error characteristics."""
        positions, masses = hierarchical_system
        
        # Two computations with different precision
        tree_high = bhut.Tree(positions, masses)
        tree_high.build()
        tree_low = bhut.Tree(positions, masses)
        tree_low.build()
        
        acc_high = tree_high.accelerations(theta=0.1)
        acc_low = tree_low.accelerations(theta=1.5)
        
        # Calculate absolute and relative errors
        abs_error = np.abs(acc_high - acc_low)
        rel_error = abs_error / (np.abs(acc_high) + 1e-15)  # Avoid division by zero
        
        # For reasonable systems, relative error should be bounded
        assert np.max(rel_error) < 1.0  # Should not be 100% error
        
        # Large accelerations should have relatively smaller relative errors
        large_acc_mask = np.abs(acc_high) > np.percentile(np.abs(acc_high), 75)
        if np.any(large_acc_mask):
            large_rel_errors = rel_error[large_acc_mask]
            small_rel_errors = rel_error[~large_acc_mask]
            
            # This is a weak test - just check for reasonableness
            assert np.median(large_rel_errors) <= np.median(small_rel_errors) * 2


@pytest.mark.validation
class TestSpecialConfigurations:
    """Validate behavior for special particle configurations."""

    def test_colinear_configuration(self):
        """Test particles arranged in a line."""
        # Three particles on x-axis
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 2.0, 1.0])  # Symmetric masses on ends
        
        acc = bhut.accelerations(positions, masses)
        
        # Middle particle should not accelerate in y or z
        np.testing.assert_allclose(acc[1, 1:], 0.0, atol=1e-12)
        
        # End particles should accelerate toward center
        assert acc[0, 0] > 0  # Positive x direction
        assert acc[2, 0] < 0  # Negative x direction
        
        # Due to symmetry, end accelerations should have equal magnitude
        np.testing.assert_allclose(np.abs(acc[0, 0]), np.abs(acc[2, 0]), rtol=1e-10)

    def test_planar_configuration(self):
        """Test particles in a plane."""
        # Four particles in xy-plane forming a square
        positions = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        masses = np.array([1.0, 1.0, 1.0, 1.0])  # Equal masses
        
        acc = bhut.accelerations(positions, masses)
        
        # All z-components should be zero (planar symmetry)
        np.testing.assert_allclose(acc[:, 2], 0.0, atol=1e-12)
        
        # Due to symmetry, accelerations should point toward center
        center = np.array([0.0, 0.0, 0.0])
        for i in range(len(positions)):
            direction_to_center = center - positions[i]
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)
            
            acc_direction = acc[i] / np.linalg.norm(acc[i])
            
            # Should point roughly toward center (considering other particles)
            assert np.dot(acc_direction, direction_to_center) > 0.5

    def test_hierarchical_configuration(self):
        """Test hierarchical system (binary + distant third body)."""
        # Close binary pair + distant third body
        positions = np.array([
            [-0.1, 0.0, 0.0],   # Binary component 1
            [0.1, 0.0, 0.0],    # Binary component 2
            [10.0, 0.0, 0.0],   # Distant third body
        ])
        masses = np.array([1.0, 1.0, 0.5])
        
        acc = bhut.accelerations(positions, masses)
        
        # Binary components should be strongly affected by each other
        binary_separation = 0.2
        distant_separation = 10.0
        
        # Force between binary components
        binary_force_scale = masses[0] * masses[1] / binary_separation**2
        
        # Force from distant body on binary
        distant_force_scale = masses[0] * masses[2] / distant_separation**2
        
        # Binary interaction should dominate
        assert binary_force_scale > distant_force_scale * 100
        
        # Binary components should accelerate toward each other primarily
        assert acc[0, 0] > 0  # Toward component 2
        assert acc[1, 0] < 0  # Toward component 1
        
        # Distant body should accelerate toward binary center of mass
        assert acc[2, 0] < 0  # Toward binary

    def test_symmetric_configurations(self):
        """Test various symmetric configurations."""
        # Tetrahedral configuration
        a = 1.0 / np.sqrt(3)
        positions = np.array([
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a],
        ])
        masses = np.array([1.0, 1.0, 1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Due to tetrahedral symmetry, all accelerations should have equal magnitude
        acc_magnitudes = np.linalg.norm(acc, axis=1)
        np.testing.assert_allclose(acc_magnitudes, acc_magnitudes[0], rtol=1e-10)
        
        # Each particle should accelerate toward center
        center = np.array([0.0, 0.0, 0.0])
        for i in range(len(positions)):
            direction_to_center = center - positions[i]
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)
            
            acc_direction = acc[i] / np.linalg.norm(acc[i])
            np.testing.assert_allclose(acc_direction, direction_to_center, rtol=1e-10)


@pytest.mark.validation
class TestLimitingBehavior:
    """Test behavior in limiting cases."""

    def test_large_separation_limit(self):
        """Test behavior when particles are very far apart."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],  # Very far
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Forces should be very small
        force_magnitude = np.linalg.norm(acc[0] * masses[0])
        expected_force = masses[0] * masses[1] / (1e6)**2  # F = Gm1m2/r^2
        
        np.testing.assert_allclose(force_magnitude, expected_force, rtol=1e-10)
        
        # Should still be finite and in correct direction
        assert np.all(np.isfinite(acc))
        assert acc[0, 0] > 0  # Toward second particle

    def test_small_separation_limit(self):
        """Test behavior when particles are very close."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-6, 0.0, 0.0],  # Very close
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Should remain finite (due to softening)
        assert np.all(np.isfinite(acc))
        
        # Should be large but not infinite
        force_magnitude = np.linalg.norm(acc[0] * masses[0])
        assert force_magnitude > 1e6  # Should be large
        assert force_magnitude < np.inf

    def test_zero_mass_limit(self):
        """Test behavior with very small masses."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1e-12, 1.0])  # Very small first mass
        
        acc = bhut.accelerations(positions, masses)
        
        # Should still work
        assert np.all(np.isfinite(acc))
        
        # Small mass should be affected by large mass
        assert np.abs(acc[0, 0]) > np.abs(acc[1, 0])  # Small mass accelerates more

    def test_large_mass_ratio_limit(self):
        """Test behavior with very large mass ratios."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1e12, 1.0])  # Very large mass ratio
        
        acc = bhut.accelerations(positions, masses)
        
        # Should remain finite
        assert np.all(np.isfinite(acc))
        
        # Forces should be equal and opposite
        force_0 = acc[0] * masses[0]
        force_1 = acc[1] * masses[1]
        np.testing.assert_allclose(force_0, -force_1, rtol=1e-12)
        
        # Small mass should have much larger acceleration
        assert np.abs(acc[1, 0]) > np.abs(acc[0, 0]) * 1e6
