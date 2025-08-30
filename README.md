# bhut

<div align="center">
  <img src="docs/assets/logo.png" alt="bhut logo" width="200"/>
</div>

A high-performance Barnes-Hut N-body accelerator that is array-agnostic, supporting both NumPy and Dask arrays for distributed computation.

[![CI](https://github.com/maroba/bhut/workflows/CI/badge.svg)](https://github.com/maroba/bhut/actions)
[![PyPI](https://img.shields.io/pypi/v/bhut.svg)](https://pypi.org/project/bhut/)
[![Python](https://img.shields.io/pypi/pyversions/bhut.svg)](https://pypi.org/project/bhut/)

## Documentation

Full documentation is available at:

**https://maroba.github.io/bhut/**

## Installation

```bash
pip install bhut
```

For acceleration features and optional dependencies:
```bash
pip install bhut[accel]
pip install 'dask[array]'  # For distributed computation
pip install numba         # For JIT acceleration (recommended)
```

## Quick Start

### Functional API

```python
import numpy as np
import bhut

# Generate random N-body system
N = 10000
positions = np.random.randn(N, 3)
masses = np.ones(N)

# Compute gravitational accelerations using Barnes-Hut algorithm
accelerations = bhut.accelerations(
    positions, masses, 
    theta=0.5,        # Opening angle criterion
    softening=0.01,   # Gravitational softening
    G=1.0            # Gravitational constant
)

print(f"Acceleration shape: {accelerations.shape}")  # (10000, 3)
```

### Object-Oriented API

```python
import numpy as np
from bhut import Tree

# Create and build tree
positions = np.random.randn(1000, 3)
masses = np.ones(1000)

tree = Tree(positions, masses, leaf_size=32)
tree.build()

# Query accelerations for same particles
self_accel = tree.accelerations(theta=0.5, softening=0.01)

# Query accelerations at different target positions
targets = np.random.randn(500, 3)
target_accel = tree.accelerations(targets, theta=0.5, softening=0.01)

# Efficient tree updates for time evolution
new_positions = positions + 0.1 * np.random.randn(1000, 3)

# Refit: fast update when particles move but topology is similar
tree.refit(new_positions)
accel_refit = tree.accelerations(theta=0.5, softening=0.01)

# Rebuild: complete reconstruction when topology changes significantly
tree.rebuild(new_positions, new_masses=masses * 1.1)
accel_rebuild = tree.accelerations(theta=0.5, softening=0.01)
```

### Distributed Computing with Dask

```python
import numpy as np
import dask.array as da
import bhut

# Create large dataset distributed across chunks
N = 1_000_000
positions_np = np.random.randn(N, 3).astype(np.float64)
masses_np = np.ones(N, dtype=np.float64)

# Convert to Dask arrays with chunking
positions_da = da.from_array(positions_np, chunks=(100_000, 3))
masses_da = da.from_array(masses_np, chunks=(100_000,))

# Compute accelerations in parallel (auto-detects Dask backend)
accelerations_da = bhut.accelerations(positions_da, masses_da, theta=0.5)

# Result preserves chunking structure for efficient downstream processing
print(f"Input chunks: {positions_da.chunks}")
print(f"Output chunks: {accelerations_da.chunks}")

# Compute final result when needed
result = accelerations_da.compute()
```

### Numba JIT Acceleration

For maximum performance with NumPy arrays, install Numba for automatic JIT compilation:

```python
import numpy as np
import bhut

# Numba automatically detected and used if available
positions = np.random.randn(10000, 3)
masses = np.ones(10000)

# First call includes compilation overhead (~1s)
accel1 = bhut.accelerations(positions, masses, theta=0.5)

# Subsequent calls use cached compiled code (~27x faster)
accel2 = bhut.accelerations(positions, masses, theta=0.5)
```

**Performance comparison (10,000 particles):**
- **Without Numba**: ~4.5 seconds (pure Python)
- **With Numba**: ~0.16 seconds after compilation (~27x speedup)
- **Memory**: No additional memory overhead
- **Compatibility**: Falls back gracefully when Numba unavailable

**When Numba is used:**
- Automatically detected when `import numba` succeeds
- Only accelerates compute-intensive leaf node interactions
- Pure Python fallback always available
- Works with NumPy arrays (Dask arrays use pure Python)

```python
# Check if Numba acceleration is active
from bhut.traverse.bh import HAVE_NUMBA
print(f"Numba acceleration: {'✓ Available' if HAVE_NUMBA else '✗ Not available'}")
```

## Features

- ** High Performance**: O(N log N) tree construction, O(M log N) force evaluation
- ** Numba Acceleration**: Optional JIT compilation for ~27x speedup in particle interactions
- ** Array-Agnostic**: Seamless support for NumPy and Dask arrays
- ** Distributed**: Built-in Dask integration for large-scale computation
- ** Accurate**: Configurable Barnes-Hut approximation with error control
- ** Efficient Updates**: Tree refit/rebuild for time-stepping simulations
- ** Deterministic**: Stable Morton ordering ensures reproducible results
- ** Multi-dimensional**: Support for 2D and 3D spatial problems
- ** Thoroughly Tested**: Comprehensive test suite with 153+ tests covering unit, integration, performance, validation, and edge cases

## API Reference

### Core Functions

#### `bhut.accelerations(positions, masses, **kwargs)`

Compute gravitational accelerations using the Barnes-Hut algorithm.

**Parameters:**
- `positions` *(array_like)*: Particle positions, shape `(N, dim)`
- `masses` *(array_like)*: Particle masses, shape `(N,)`
- `theta` *(float, default=0.5)*: Opening angle criterion
- `softening` *(float, default=0.0)*: Gravitational softening length
- `G` *(float, default=1.0)*: Gravitational constant
- `dim` *(int, default=3)*: Spatial dimensions (2 or 3)
- `backend` *(str, default="auto")*: Array backend ("numpy", "dask", "auto")
- `leaf_size` *(int, default=32)*: Maximum particles per leaf node

**Returns:**
- `accelerations` *(array_like)*: Gravitational accelerations, shape `(N, dim)`

### Tree Class

#### `Tree(positions, masses, **kwargs)`

Object-oriented interface for Barnes-Hut tree operations.

**Constructor Parameters:**
- `positions` *(array_like)*: Particle positions, shape `(N, dim)`
- `masses` *(array_like)*: Particle masses, shape `(N,)`
- `leaf_size` *(int, default=32)*: Maximum particles per leaf node
- `backend` *(str, default="auto")*: Array backend ("numpy", "dask", "auto")
- `dim` *(int, default=3)*: Spatial dimensions (2 or 3)

**Methods:**
- `build()`: Construct the tree structure (required before first use)
- `accelerations(targets=None, theta=0.5, **kwargs)`: Evaluate accelerations
- `refit(new_positions, new_masses=None)`: Update tree with new positions
- `rebuild(new_positions, new_masses=None)`: Reconstruct tree completely

**Important:** Always call `tree.build()` after creating a Tree and before calling other methods. The `theta` parameter is passed to the `accelerations()` method, not the constructor.

## Parameter Tuning Guide

### Opening Angle (`theta`)

Controls the accuracy-performance tradeoff:

```python
# High accuracy, slow computation
accel_accurate = bhut.accelerations(pos, masses, theta=0.1)

# Balanced (recommended for most applications)
accel_balanced = bhut.accelerations(pos, masses, theta=0.5)

# Fast approximation, lower accuracy
accel_fast = bhut.accelerations(pos, masses, theta=1.0)

# Direct summation (exact but O(N²))
accel_exact = bhut.accelerations(pos, masses, theta=0.0)
```

**Guidelines:**
- `theta = 0.0`: Exact O(N²) calculation (small systems only)
- `theta = 0.1-0.3`: High accuracy for precision-critical applications
- `theta = 0.5`: Good balance for most scientific simulations
- `theta = 0.7-1.0`: Fast approximation for large-scale surveys
- `theta > 1.0`: Very approximate, mainly for prototyping

### Leaf Size (`leaf_size`)

Controls tree granularity and performance:

```python
# Fine-grained tree (more memory, potentially faster for large queries)
tree_fine = Tree(pos, masses, leaf_size=16)
tree_fine.build()

# Balanced (recommended default)
tree_balanced = Tree(pos, masses, leaf_size=32)
tree_balanced.build()

# Coarse-grained tree (less memory, faster construction)
tree_coarse = Tree(pos, masses, leaf_size=64)
tree_coarse.build()
```

**Guidelines:**
- `leaf_size = 8-16`: Best for systems with many small query sets
- `leaf_size = 32`: Recommended default for most applications
- `leaf_size = 64-128`: Better for large query sets or memory-constrained systems
- `leaf_size > 128`: May degrade performance due to increased direct summation

### Performance Optimization

#### Tree Refit vs Rebuild

For time-stepping simulations:

```python
tree = Tree(positions, masses)
tree.build()

for timestep in range(num_steps):
    # Compute forces
    accel = tree.accelerations(theta=0.5, softening=0.01)
    
    # Update positions
    positions += velocity * dt + 0.5 * accel * dt**2
    velocity += accel * dt
    
    # Decide whether to refit or rebuild
    if should_refit_vs_rebuild(tree, positions):
        tree.refit(positions)  # Fast: O(N), preserves tree structure
    else:
        tree.rebuild(positions)  # Slower: O(N log N), rebuilds from scratch
```

#### Chunking Strategy for Dask

```python
# Good chunking: ~100K-1M particles per chunk
positions_da = da.from_array(positions, chunks=(500_000, 3))

# Avoid: too small chunks (overhead dominates)
positions_bad = da.from_array(positions, chunks=(1_000, 3))

# Avoid: too large chunks (memory issues, poor parallelization)
positions_bad = da.from_array(positions, chunks=(10_000_000, 3))
```

## Performance Characteristics

| System Size | Construction Time | Memory Usage | Recommended θ | Numba Speedup |
|-------------|------------------|--------------|---------------|---------------|
| N < 10³     | < 1ms            | < 1MB        | 0.0 (exact)   | ~2x           |
| N ~ 10⁴     | ~ 10ms           | ~ 10MB       | 0.3           | ~10x          |
| N ~ 10⁵     | ~ 100ms          | ~ 100MB      | 0.5           | ~27x          |
| N ~ 10⁶     | ~ 1s             | ~ 1GB        | 0.5           | ~27x          |
| N > 10⁶     | Use Dask         | Distributed  | 0.7           | N/A*          |

*Dask arrays use pure Python (Numba optimizations planned for future releases)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use bhut in your research, please cite:

```bibtex
@software{bhut,
  title={bhut: A Barnes-Hut N-body Accelerator},
  author={Your Name},
  url={https://github.com/your-org/bhut},
  year={2025}
}
```
