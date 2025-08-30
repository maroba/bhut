"""
Barnes-Hut tree traversal algorithm for gravitational force calculation.

This module implements the core Barnes-Hut algorithm using an iterative
(stack-based) tree traversal with configurable opening criteria.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ..tree.node import TreeData
from .kernels import acc_monopole_node, validate_kernel_inputs

# Optional Numba acceleration
try:
    from numba import jit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    
    # Create a no-op decorator for when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def barnes_hut_accelerations_targets(
    tree: TreeData,
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]],
    target_positions: NDArray[np.floating[Any]],
    softening: float,
    theta: float = 0.5,
    G: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """
    Compute gravitational accelerations at target positions using Barnes-Hut algorithm.
    
    This function evaluates accelerations at arbitrary target positions due to 
    source particles that were used to build the tree.
    
    Parameters
    ----------
    tree : TreeData
        Constructed Barnes-Hut tree
    tree_positions : ndarray, shape (N, D)
        Positions of particles used to build the tree (source particles)
    tree_masses : ndarray, shape (N,)
        Masses of particles used to build the tree
    target_positions : ndarray, shape (M, D)
        Target positions where accelerations are evaluated  
    softening : float
        Gravitational softening length
    theta : float, default 0.5
        Opening angle criterion for node approximation
    G : float, default 1.0
        Gravitational constant
        
    Returns
    -------
    accelerations : ndarray, shape (M, D)
        Gravitational accelerations at each target position
    """
    validate_kernel_inputs(tree_positions, tree_masses, softening)
    
    n_targets, n_dims = target_positions.shape
    accelerations = np.zeros_like(target_positions)
    
    # Traverse tree for each target position
    for i in range(n_targets):
        target_pos = target_positions[i]
        acceleration = _barnes_hut_single_target_external(
            tree, tree_positions, tree_masses, target_pos, softening, theta, G
        )
        accelerations[i] = acceleration
    
    return accelerations


def _barnes_hut_single_target_external(
    tree: TreeData,
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]], 
    target_pos: NDArray[np.floating[Any]],
    softening: float,
    theta: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Compute acceleration on external target using Barnes-Hut traversal.
    
    This version is for computing acceleration at positions that are not 
    part of the original particle set used to build the tree.
    """
    n_dims = len(target_pos)
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    # Stack for tree traversal: each element is a node index
    stack = [0]  # Start with root node
    
    while stack:
        node_idx = stack.pop()
        
        # Skip if invalid node
        if node_idx < 0 or node_idx >= len(tree.mass):
            continue
        
        # Skip empty nodes
        if tree.mass[node_idx] == 0:
            continue
        
        # Check if this is a leaf node
        is_leaf = tree.first_child[node_idx] == -1
        
        if is_leaf:
            # Leaf node: compute direct particle-particle interactions
            acceleration += _leaf_acceleration_external(
                tree, node_idx, target_pos, tree_positions, tree_masses, softening, G
            )
        else:
            # Internal node: check opening criterion
            should_open = _should_open_node(
                tree, node_idx, target_pos, theta
            )
            
            if should_open:
                # Open node: add children to stack
                child_idx = tree.first_child[node_idx]
                for i in range(tree.child_count[node_idx]):
                    if child_idx != -1:
                        stack.append(child_idx)
                        child_idx += 1  # Children are stored contiguously
            else:
                # Use node as monopole approximation
                node_com = tree.com[node_idx]
                node_mass = tree.mass[node_idx]
                
                node_acc = acc_monopole_node(
                    target_pos, node_com, node_mass, softening, G
                )
                acceleration += node_acc
    
    return acceleration


def _leaf_acceleration_external(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Compute acceleration from particles in a leaf node for external target."""
    # Use Numba if available and working with NumPy arrays
    if (HAVE_NUMBA and 
        isinstance(target_pos, np.ndarray) and 
        isinstance(tree_positions, np.ndarray) and 
        isinstance(tree_masses, np.ndarray)):
        return _leaf_acceleration_external_numba(
            tree, leaf_idx, target_pos, tree_positions, tree_masses, softening, G
        )
    else:
        return _leaf_acceleration_external_pure_python(
            tree, leaf_idx, target_pos, tree_positions, tree_masses, softening, G
        )


# Numba-optimized leaf acceleration functions
@jit(nopython=True, cache=True)
def _numba_leaf_particle_loop_external(
    target_pos: NDArray[np.floating[Any]],
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]],
    perm: NDArray[np.integer[Any]],
    start: int,
    particle_count: int,
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Numba-optimized inner loop for leaf particle acceleration computation (external targets).
    
    This function computes the direct particle-particle interactions within
    a leaf node for external target positions.
    """
    n_dims = target_pos.shape[0]
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    softening_sq = softening * softening
    
    for i in range(start, start + particle_count):
        # Get original particle index via permutation
        p_idx = perm[i]
        
        # Get particle data from tree source arrays
        particle_pos = tree_positions[p_idx]
        particle_mass = tree_masses[p_idx]
        
        # Compute displacement vector
        dx = particle_pos - target_pos
        
        # Compute distance squared with softening
        r_sq = 0.0
        for d in range(n_dims):
            r_sq += dx[d] * dx[d]
        r_sq += softening_sq
        
        # Compute acceleration magnitude
        r = np.sqrt(r_sq)
        r_inv = 1.0 / r
        r_inv_cubed = r_inv * r_inv * r_inv
        acc_mag = G * particle_mass * r_inv_cubed
        
        # Add to acceleration vector
        for d in range(n_dims):
            acceleration[d] += acc_mag * dx[d]
    
    return acceleration


@jit(nopython=True, cache=True)
def _numba_leaf_particle_loop(
    target_pos: NDArray[np.floating[Any]],
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    perm: NDArray[np.integer[Any]],
    start: int,
    particle_count: int,
    target_idx: int,
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Numba-optimized inner loop for leaf particle acceleration computation.
    
    This function computes the direct particle-particle interactions within
    a leaf node, excluding self-interaction.
    """
    n_dims = target_pos.shape[0]
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    softening_sq = softening * softening
    
    for i in range(start, start + particle_count):
        # Get original particle index via permutation
        p_idx = perm[i]
        
        # Skip self-interaction
        if p_idx == target_idx:
            continue
        
        # Get particle data
        particle_pos = positions[p_idx]
        particle_mass = masses[p_idx]
        
        # Compute displacement vector
        dx = particle_pos - target_pos
        
        # Compute distance squared with softening
        r_sq = 0.0
        for d in range(n_dims):
            r_sq += dx[d] * dx[d]
        r_sq += softening_sq
        
        # Compute acceleration magnitude
        r = np.sqrt(r_sq)
        r_inv = 1.0 / r
        r_inv_cubed = r_inv * r_inv * r_inv
        acc_mag = G * particle_mass * r_inv_cubed
        
        # Add to acceleration vector
        for d in range(n_dims):
            acceleration[d] += acc_mag * dx[d]
    
    return acceleration


def _leaf_acceleration_external_numba(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Compute acceleration from particles in a leaf node for external target (Numba-optimized)."""
    # Get particle range for this leaf
    start = tree.start[leaf_idx]
    particle_count = tree.count[leaf_idx]
    
    if start == -1 or particle_count <= 0:
        n_dims = len(target_pos)
        return np.zeros(n_dims, dtype=np.float64)
    
    # Use Numba-optimized inner loop
    try:
        return _numba_leaf_particle_loop_external(
            target_pos, tree_positions, tree_masses, tree.perm,
            start, particle_count, softening, G
        )
    except Exception:
        # Fallback to pure Python if Numba compilation fails
        return _leaf_acceleration_external_pure_python(
            tree, leaf_idx, target_pos, tree_positions, tree_masses, softening, G
        )


def _leaf_acceleration_numba(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    target_idx: int,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Compute acceleration from particles in a leaf node (Numba-optimized)."""
    # Get particle range for this leaf
    start = tree.start[leaf_idx]
    particle_count = tree.count[leaf_idx]
    
    if start == -1 or particle_count <= 0:
        n_dims = len(target_pos)
        return np.zeros(n_dims, dtype=np.float64)
    
    # Use Numba-optimized inner loop
    try:
        return _numba_leaf_particle_loop(
            target_pos, positions, masses, tree.perm,
            start, particle_count, target_idx, softening, G
        )
    except Exception:
        # Fallback to pure Python if Numba compilation fails
        return _leaf_acceleration_pure_python(
            tree, leaf_idx, target_pos, target_idx, positions, masses, softening, G
        )


def _leaf_acceleration_external_pure_python(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    tree_positions: NDArray[np.floating[Any]],
    tree_masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Pure Python implementation for external leaf acceleration."""
    n_dims = len(target_pos)
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    # Get particle range for this leaf
    start = tree.start[leaf_idx]
    particle_count = tree.count[leaf_idx]
    
    if start == -1 or particle_count <= 0:
        return acceleration
    
    # Compute direct particle interactions
    for i in range(start, start + particle_count):
        # Get original particle index via permutation
        p_idx = tree.perm[i]
        
        # Get particle data from tree source arrays
        particle_pos = tree_positions[p_idx]
        particle_mass = tree_masses[p_idx]
        
        # Compute acceleration from this particle
        particle_acc = acc_monopole_node(
            target_pos, particle_pos, particle_mass, softening, G
        )
        acceleration += particle_acc
    
    return acceleration


def _leaf_acceleration_pure_python(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    target_idx: int,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Pure Python implementation for leaf acceleration."""
    n_dims = len(target_pos)
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    # Get particle range for this leaf
    start = tree.start[leaf_idx]
    particle_count = tree.count[leaf_idx]
    end = start + particle_count
    
    if start == -1 or particle_count <= 0:
        return acceleration
    
    # Compute direct particle interactions
    for i in range(start, end):
        # Get original particle index via permutation
        p_idx = tree.perm[i]
        
        # Skip self-interaction
        if p_idx == target_idx:
            continue
        
        # Get particle data
        particle_pos = positions[p_idx]
        particle_mass = masses[p_idx]
        
        # Compute acceleration from this particle
        particle_acc = acc_monopole_node(
            target_pos, particle_pos, particle_mass, softening, G
        )
        acceleration += particle_acc
    
    return acceleration


def barnes_hut_accelerations(
    tree: TreeData,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: float,
    theta: float = 0.5,
    G: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """
    Compute gravitational accelerations using Barnes-Hut algorithm.
    
    Parameters
    ----------
    tree : TreeData
        Constructed Barnes-Hut tree
    positions : ndarray, shape (N, D)
        Particle positions
    masses : ndarray, shape (N,)
        Particle masses
    softening : float
        Gravitational softening length
    theta : float, default 0.5
        Opening angle criterion for node approximation.
        Smaller values give higher accuracy but slower computation.
    G : float, default 1.0
        Gravitational constant
        
    Returns
    -------
    accelerations : ndarray, shape (N, D)
        Gravitational accelerations for each particle
        
    Notes
    -----
    The Barnes-Hut algorithm uses a tree data structure to group distant
    particles and approximate their gravitational effects using multipole
    expansions (monopole in this implementation).
    
    The opening criterion is: distance > size / theta
    Where size is the node width and distance is from target to node COM.
    """
    validate_kernel_inputs(positions, masses, softening)
    
    n_particles, n_dims = positions.shape
    accelerations = np.zeros_like(positions)
    
    # Traverse tree for each particle
    for i in range(n_particles):
        target_pos = positions[i]
        acceleration = _barnes_hut_single_target(
            tree, target_pos, i, positions, masses, softening, theta, G
        )
        accelerations[i] = acceleration
    
    return accelerations


def _barnes_hut_single_target(
    tree: TreeData,
    target_pos: NDArray[np.floating[Any]],
    target_idx: int,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: float,
    theta: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Compute acceleration on single target using Barnes-Hut traversal.
    
    Uses an iterative (stack-based) approach to avoid recursion limits.
    """
    n_dims = len(target_pos)
    acceleration = np.zeros(n_dims, dtype=np.float64)
    
    # Stack for tree traversal: each element is a node index
    stack = [0]  # Start with root node
    
    while stack:
        node_idx = stack.pop()
        
        # Skip if invalid node
        if node_idx < 0 or node_idx >= len(tree.mass):
            continue
        
        # Skip empty nodes
        if tree.mass[node_idx] == 0:
            continue
        
        # Check if this is a leaf node
        is_leaf = tree.first_child[node_idx] == -1
        
        if is_leaf:
            # Leaf node: compute direct particle-particle interactions
            acceleration += _leaf_acceleration(
                tree, node_idx, target_pos, target_idx, positions, masses, softening, G
            )
        else:
            # Internal node: check opening criterion
            should_open = _should_open_node(
                tree, node_idx, target_pos, theta
            )
            
            if should_open:
                # Open node: add children to stack
                child_idx = tree.first_child[node_idx]
                for i in range(tree.child_count[node_idx]):
                    if child_idx != -1:
                        stack.append(child_idx)
                        child_idx += 1  # Children are stored contiguously
            else:
                # Use node as monopole approximation
                node_com = tree.com[node_idx]
                node_mass = tree.mass[node_idx]
                
                node_acc = acc_monopole_node(
                    target_pos, node_com, node_mass, softening, G
                )
                acceleration += node_acc
    
    return acceleration


def _leaf_acceleration(
    tree: TreeData,
    leaf_idx: int,
    target_pos: NDArray[np.floating[Any]],
    target_idx: int,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """Compute acceleration from particles in a leaf node."""
    # Use Numba if available and working with NumPy arrays
    if (HAVE_NUMBA and 
        isinstance(target_pos, np.ndarray) and 
        isinstance(positions, np.ndarray) and 
        isinstance(masses, np.ndarray)):
        return _leaf_acceleration_numba(
            tree, leaf_idx, target_pos, target_idx, positions, masses, softening, G
        )
    else:
        return _leaf_acceleration_pure_python(
            tree, leaf_idx, target_pos, target_idx, positions, masses, softening, G
        )


def _should_open_node(
    tree: TreeData,
    node_idx: int,
    target_pos: NDArray[np.floating[Any]],
    theta: float,
) -> bool:
    """
    Determine if a node should be opened based on opening criterion.
    
    Uses the standard Barnes-Hut criterion: s/d < theta
    Where s is the node size and d is the distance to the node COM.
    """
    # Get node properties
    node_com = tree.com[node_idx]
    node_size = 2 * tree.half_size[node_idx]  # Full size of bounding box
    
    # Compute distance from target to node COM
    distance = np.linalg.norm(target_pos - node_com)
    
    # Avoid division by zero
    if distance == 0:
        return True  # Always open if target is at node COM
    
    # Opening criterion: open if s/d >= theta
    ratio = node_size / distance
    return ratio >= theta


def compute_tree_statistics(
    tree: TreeData,
    positions: NDArray[np.floating[Any]],
    theta: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute statistics about tree traversal for performance analysis.
    
    Parameters
    ----------
    tree : TreeData
        Barnes-Hut tree
    positions : ndarray, shape (N, D)
        Particle positions
    theta : float
        Opening angle parameter
        
    Returns
    -------
    stats : dict
        Dictionary with traversal statistics:
        - 'avg_nodes_visited': Average nodes visited per particle
        - 'avg_particles_computed': Average direct particle interactions
        - 'avg_monopoles_used': Average monopole approximations used
        - 'total_operations': Total force evaluations
    """
    n_particles = len(positions)
    total_nodes = 0
    total_particles = 0
    total_monopoles = 0
    
    for i in range(n_particles):
        target_pos = positions[i]
        stats = _count_operations_single_target(tree, target_pos, i, theta)
        total_nodes += stats['nodes_visited']
        total_particles += stats['particles_computed']
        total_monopoles += stats['monopoles_used']
    
    return {
        'avg_nodes_visited': total_nodes / n_particles,
        'avg_particles_computed': total_particles / n_particles,
        'avg_monopoles_used': total_monopoles / n_particles,
        'total_operations': total_particles + total_monopoles,
    }


def _count_operations_single_target(
    tree: TreeData,
    target_pos: NDArray[np.floating[Any]],
    target_idx: int,
    theta: float,
) -> Dict[str, int]:
    """Count operations for single target (for performance analysis)."""
    nodes_visited = 0
    particles_computed = 0
    monopoles_used = 0
    
    stack = [0]  # Start with root
    
    while stack:
        node_idx = stack.pop()
        nodes_visited += 1
        
        # Skip invalid/empty nodes
        if (node_idx < 0 or node_idx >= len(tree.mass) or 
            tree.mass[node_idx] == 0):
            continue
        
        is_leaf = tree.first_child[node_idx] == -1
        
        if is_leaf:
            # Count direct particle interactions
            start = tree.start[node_idx]
            particle_count = tree.count[node_idx]
            if start != -1 and particle_count > 0:
                # Subtract 1 for self-interaction that gets skipped
                leaf_particles = max(0, particle_count - 1)
                particles_computed += leaf_particles
        else:
            should_open = _should_open_node(tree, node_idx, target_pos, theta)
            
            if should_open:
                # Add children to stack
                child_idx = tree.first_child[node_idx]
                for i in range(tree.child_count[node_idx]):
                    if child_idx != -1:
                        stack.append(child_idx)
                        child_idx += 1
            else:
                # Use monopole approximation
                monopoles_used += 1
    
    return {
        'nodes_visited': nodes_visited,
        'particles_computed': particles_computed,
        'monopoles_used': monopoles_used,
    }
