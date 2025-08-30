"""
Shared test fixtures and configuration for the bhut test suite.
"""

import numpy as np
import pytest

import bhut


@pytest.fixture
def simple_system():
    """Simple 2-particle system for basic testing."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64)
    masses = np.array([1.0, 1.0], dtype=np.float64)
    return positions, masses


@pytest.fixture
def random_system():
    """Random N-particle system with reproducible seed."""
    np.random.seed(42)
    positions = np.random.randn(10, 3).astype(np.float64)
    masses = np.ones(10, dtype=np.float64)
    return positions, masses


@pytest.fixture
def hierarchical_system():
    """Hierarchical system: tight binary + distant particle."""
    positions = np.array([
        [-0.1, 0.0, 0.0],  # Binary component 1
        [0.1, 0.0, 0.0],   # Binary component 2  
        [10.0, 0.0, 0.0],  # Distant particle
    ], dtype=np.float64)
    masses = np.array([1.0, 1.0, 0.1], dtype=np.float64)
    return positions, masses


@pytest.fixture
def planar_system():
    """System confined to xy-plane."""
    np.random.seed(123)
    positions = np.random.randn(8, 3).astype(np.float64)
    positions[:, 2] = 0.0  # Set z=0
    masses = np.ones(8, dtype=np.float64)
    return positions, masses


@pytest.fixture
def clustered_system():
    """Tightly clustered particles for testing extreme cases."""
    base_pos = np.array([5.0, 5.0, 5.0])
    positions = base_pos + 1e-8 * np.random.randn(6, 3)
    masses = np.ones(6, dtype=np.float64)
    return positions, masses


@pytest.fixture
def varied_masses_system():
    """System with large mass variations."""
    np.random.seed(456)
    positions = np.random.randn(8, 3).astype(np.float64)
    masses = np.array([0.1, 1.0, 10.0, 0.5, 2.0, 0.01, 5.0, 0.2], dtype=np.float64)
    return positions, masses


@pytest.fixture
def tree_built(random_system):
    """Pre-built tree for testing tree operations."""
    positions, masses = random_system
    tree = bhut.Tree(positions, masses)
    tree.build()
    return tree


@pytest.fixture
def common_params():
    """Common parameter sets for testing."""
    return {
        'theta_values': [0.0, 0.1, 0.5, 1.0, 2.0],
        'softening_values': [0.0, 1e-6, 0.01, 0.1],
        'leaf_sizes': [1, 4, 8, 16],
        'G_values': [1.0, 6.67430e-11, 2.0]
    }


# Test markers for easy subset running
pytest.register_marker = lambda x: x  # For IDEs that don't recognize markers

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration and workflow tests") 
    config.addinivalue_line("markers", "validation: Physics and numerical validation tests")
    config.addinivalue_line("markers", "edge_cases: Edge cases and error condition tests")
    config.addinivalue_line("markers", "performance: Performance and optimization tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "requires_dask: Tests that require dask to be installed")
