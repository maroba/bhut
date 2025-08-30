"""
Unit tests for physics kernels and gravitational force calculations.

Tests the mathematical accuracy and consistency of force calculations through the public API.
"""

import numpy as np
import pytest

import bhut


@pytest.mark.unit
class TestPhysicsConsistency:
    """Test physics consistency through the public API."""

    def test_two_body_force_symmetry(self):
        """Test Newton's third law through accelerations."""
        # Two masses: should have equal and opposite forces
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 2.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Forces should be equal and opposite
        force_0 = acc[0] * masses[0]
        force_1 = acc[1] * masses[1]
        
        np.testing.assert_allclose(force_0, -force_1, rtol=1e-12)

    def test_inverse_square_law(self):
        """Test inverse square law scaling."""
        masses = np.array([1.0, 1.0])
        
        # Test at different distances
        distances = [1.0, 2.0, 4.0]
        force_magnitudes = []
        
        for d in distances:
            positions = np.array([
                [0.0, 0.0, 0.0],
                [d, 0.0, 0.0],
            ])
            
            acc = bhut.accelerations(positions, masses)
            force_magnitude = np.linalg.norm(acc[0] * masses[0])
            force_magnitudes.append(force_magnitude)
        
        # Force should scale as 1/r^2
        for i in range(1, len(distances)):
            ratio = force_magnitudes[0] / force_magnitudes[i]
            expected_ratio = (distances[i] / distances[0])**2
            np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-8)

    def test_center_of_mass_properties(self, random_system):
        """Test center of mass related properties."""
        positions, masses = random_system
        
        accelerations = bhut.accelerations(positions, masses)
        
        # Total momentum change should be zero (Newton's third law)
        total_force = np.sum(accelerations * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-12)

    def test_mass_scaling(self):
        """Test that accelerations scale correctly with mass."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        
        # Reference with unit masses
        masses_ref = np.array([1.0, 1.0])
        acc_ref = bhut.accelerations(positions, masses_ref)
        
        # Test mass scaling
        scale_factor = 3.0
        masses_scaled = masses_ref * scale_factor
        acc_scaled = bhut.accelerations(positions, masses_scaled)
        
        # Accelerations should scale with source mass
        expected_acc = acc_ref * scale_factor
        np.testing.assert_allclose(acc_scaled, expected_acc, rtol=1e-12)


@pytest.mark.unit
class TestPhysicsValidation:
    """Validate physics through API."""

    def test_conservation_laws(self, simple_system):
        """Test conservation laws."""
        positions, masses = simple_system
        
        acc = bhut.accelerations(positions, masses)
        
        # Conservation of momentum
        total_momentum_change = np.sum(acc * masses[:, np.newaxis], axis=0)
        np.testing.assert_allclose(total_momentum_change, 0.0, atol=1e-12)

    def test_symmetry_properties(self):
        """Test symmetry properties."""
        # Symmetric configuration
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 1.0])  # Equal masses
        
        accelerations = bhut.accelerations(positions, masses)
        
        # Due to symmetry, accelerations should have equal magnitude, opposite direction
        acc_magnitude_0 = np.linalg.norm(accelerations[0])
        acc_magnitude_1 = np.linalg.norm(accelerations[1])
        
        np.testing.assert_allclose(acc_magnitude_0, acc_magnitude_1, rtol=1e-15)
        np.testing.assert_allclose(accelerations[0], -accelerations[1], rtol=1e-15)

    def test_direction_correctness(self):
        """Test that forces point in correct directions."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        masses = np.array([1.0, 1.0])
        
        acc = bhut.accelerations(positions, masses)
        
        # Particle 0 should accelerate toward particle 1 (+x direction)
        assert acc[0, 0] > 0
        assert abs(acc[0, 1]) < 1e-12  # No y component
        assert abs(acc[0, 2]) < 1e-12  # No z component
        
        # Particle 1 should accelerate toward particle 0 (-x direction)
        assert acc[1, 0] < 0
        assert abs(acc[1, 1]) < 1e-12  # No y component
        assert abs(acc[1, 2]) < 1e-12  # No z component
