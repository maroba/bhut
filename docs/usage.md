# Usage Guide

## Key Parameters

### theta (θ)

The **theta** parameter controls the accuracy-speed tradeoff:

- `theta = 0.0`: Exact calculation (equivalent to brute force)
- `theta = 0.5`: Good balance of speed and accuracy (recommended default)
- `theta = 1.0`: Faster but less accurate
- `theta > 1.0`: Very fast but potentially poor accuracy

```python
# High accuracy, slower
forces = bhut.force(positions, masses, theta=0.1)

# Balanced (recommended)
forces = bhut.force(positions, masses, theta=0.5)

# Fast approximation
forces = bhut.force(positions, masses, theta=1.0)
```

### leaf_size

Controls when to stop subdividing tree nodes:

```python
# Smaller leaves = deeper tree, more accurate but slower
tree = bhut.Tree(positions, masses, leaf_size=5)

# Larger leaves = shallower tree, faster but less accurate
tree = bhut.Tree(positions, masses, leaf_size=20)
```

### softening

Softening parameter to avoid singularities at short distances:

```python
# Add softening for numerical stability
forces = bhut.force(positions, masses, softening=0.01)
```

## 2D vs 3D Usage

### 2D Problems

```python
# 2D positions (N x 2 array)
positions_2d = np.random.rand(1000, 2)
forces_2d = bhut.force(positions_2d, masses, theta=0.5)
```

### 3D Problems

```python
# 3D positions (N x 3 array)
positions_3d = np.random.rand(1000, 3)
forces_3d = bhut.force(positions_3d, masses, theta=0.5)
```

## Source vs Target Particles

You can compute forces on different target positions than source positions:

```python
# Source particles (create the tree)
source_pos = np.random.rand(10000, 2)
source_masses = np.ones(10000)

# Target particles (where we want forces)
target_pos = np.random.rand(100, 2)

# Compute forces on targets due to sources
tree = bhut.Tree(source_pos, source_masses)
forces = tree.force(target_pos)
```

## Backends

Choose different computational backends:

```python
# NumPy backend (default)
forces = bhut.force(positions, masses, backend='numpy')

# Dask backend for larger datasets
forces = bhut.force(positions, masses, backend='dask')
```

## Advanced Examples

### Time Evolution

```python
import bhut
import numpy as np

# Initial conditions
positions = np.random.rand(1000, 2)
velocities = np.zeros_like(positions)
masses = np.ones(1000)
dt = 0.01

# Time stepping loop
for step in range(100):
    # Compute forces
    forces = bhut.force(positions, masses, theta=0.5)
    
    # Update velocities and positions (leapfrog)
    velocities += forces / masses[:, np.newaxis] * dt
    positions += velocities * dt
```

### Custom Force Laws

```python
# Different force kernels
forces = bhut.force(positions, masses, kernel='gravitational')
forces = bhut.force(positions, masses, kernel='coulomb')
```