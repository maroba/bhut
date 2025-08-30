# Getting Started

## Installation

Install bhut from PyPI using pip:

```bash
pip install bhut
```

For development or the latest features:

```bash
pip install git+https://github.com/username/bhut.git
```

## Basic Usage

### Functional Interface

The simplest way to use bhut is through the functional interface:

```python
import bhut
import numpy as np

# Create sample data
n_particles = 1000
positions = np.random.rand(n_particles, 2)  # 2D positions
masses = np.ones(n_particles)

# Compute forces using Barnes-Hut algorithm
forces = bhut.force(positions, masses, theta=0.5)

# Compute potential energy
potential = bhut.potential(positions, masses, theta=0.5)
```

### Object-Oriented Interface

For more control and repeated calculations:

```python
import bhut
import numpy as np

# Create and configure the tree
tree = bhut.Tree(
    positions=positions,
    masses=masses,
    theta=0.5,
    leaf_size=10
)

# Compute forces on the same particles
forces = tree.force()

# Or compute forces on different target positions
target_positions = np.random.rand(500, 2)
forces_on_targets = tree.force(target_positions)

# Access tree properties
print(f"Tree depth: {tree.depth}")
print(f"Number of nodes: {tree.n_nodes}")
```

## Next Steps

- Learn about [parameters and configuration](usage.md)
- Explore the [complete API reference](api/)
- Understand the [theory behind Barnes-Hut](theory/barnes-hut.md)