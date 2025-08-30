# bhut

**bhut** is a fast, flexible Barnes-Hut algorithm implementation for N-body force calculations in Python. It provides O(N log N) tree construction and O(M log N) force evaluation, making it ideal for large-scale simulations in astrophysics, molecular dynamics, and other domains requiring efficient long-range force computations.

## Key Features

- **Fast**: Optimized tree construction and traversal algorithms
- **Flexible**: Multiple backends (NumPy, Dask) for different scales
- **Accurate**: Configurable approximation parameters
- **Pythonic**: Clean API supporting both functional and object-oriented usage

## Quick Install

```bash
pip install bhut
```

## Quick Start

```python
import bhut
import numpy as np

# Generate some particles
positions = np.random.rand(1000, 2)
masses = np.ones(1000)

# Compute forces
forces = bhut.force(positions, masses, theta=0.5)
```

## Documentation Sections

- **[Getting Started](getting-started.md)** - Installation and first examples
- **[Usage Guide](usage.md)** - Parameters, 2D/3D usage, advanced features
- **[API Reference](api/)** - Complete function and class documentation
- **[Theory](theory/barnes-hut.md)** - Mathematical background and algorithms
- **[Development](dev/contributing.md)** - Contributing and development setup