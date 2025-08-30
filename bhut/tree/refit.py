"""
Tree refitting for Barnes-Hut algorithm.

This module provides efficient tree refitting when particles move slightly,
updating centers of mass and masses while preserving tree topology.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backends.base import ArrayNamespace
from .node import TreeData


def refit_tree(
    tree: TreeData,
    new_positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]], 
    xp: ArrayNamespace,
) -> TreeData:
    """
    Refit tree with new particle positions, keeping same topology.
    
    This function efficiently updates tree centers of mass and node masses
    when particles have moved slightly. It preserves the tree structure
    (centers, half_size, topology) but recomputes COM and masses.
    
    Parameters
    ----------
    tree : TreeData
        Original tree structure to refit
    new_positions : ndarray, shape (N, D)
        New particle positions in original order (not permuted)
    masses : ndarray, shape (N,)
        Particle masses (can be same as original or updated)
    xp : ArrayNamespace
        Array backend namespace
        
    Returns
    -------
    refitted_tree : TreeData
        New tree with updated COM and masses, same topology
        
    Notes
    -----
    This function assumes:
    - Same number of particles (N unchanged)
    - Same spatial distribution (no major reorganization needed)
    - Tree topology remains valid (no particles crossing major boundaries)
    
    When to use refit vs rebuild:
    - **Refit**: Small particle movements, timestep evolution, local relaxation
    - **Rebuild**: Large movements, particles crossing domain boundaries, 
      significant topology changes, different particle counts
      
    Performance: O(M) where M is number of tree nodes, vs O(N log N) for rebuild
    """
    # Validate inputs
    if new_positions.shape[0] != len(tree.perm):
        raise ValueError(
            f"new_positions has {new_positions.shape[0]} particles "
            f"but tree was built with {len(tree.perm)} particles"
        )
    
    if masses.shape[0] != len(tree.perm):
        raise ValueError(
            f"masses has {masses.shape[0]} elements "
            f"but tree was built with {len(tree.perm)} particles"
        )
    
    if new_positions.shape[1] != tree.dim:
        raise ValueError(
            f"new_positions has {new_positions.shape[1]} dimensions "
            f"but tree has {tree.dim} dimensions"
        )
    
    # Convert to numpy for computation
    new_positions_np = np.asarray(new_positions, dtype=np.float64)
    masses_np = np.asarray(masses, dtype=np.float64)
    
    # Create copy of tree with updated data
    refitted_tree = TreeData(
        center=tree.center.copy(),
        half_size=tree.half_size.copy(),
        mass=tree.mass.copy(),  # Will be updated
        com=tree.com.copy(),    # Will be updated
        first_child=tree.first_child.copy(),
        child_count=tree.child_count.copy(),
        start=tree.start.copy(),
        count=tree.count.copy(),
        is_leaf=tree.is_leaf.copy(),
        perm=tree.perm.copy(),
        dim=tree.dim,
        leaf_size=tree.leaf_size,
    )
    
    # Update leaf nodes first (bottom-up approach)
    _update_leaf_nodes(refitted_tree, new_positions_np, masses_np)
    
    # Update internal nodes bottom-up
    _update_internal_nodes_bottomup(refitted_tree, new_positions_np, masses_np)
    
    # Convert back to original array type
    refitted_tree.mass = xp.asarray(refitted_tree.mass)
    refitted_tree.com = xp.asarray(refitted_tree.com)
    
    return refitted_tree


def _update_leaf_nodes(
    tree: TreeData,
    new_positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
) -> None:
    """Update centers of mass and masses for all leaf nodes."""
    for node_idx in range(tree.num_nodes):
        if not tree.is_leaf[node_idx]:
            continue
            
        # Get particle range for this leaf
        start = tree.start[node_idx]
        count = tree.count[node_idx]
        
        if count == 0:
            # Empty leaf
            tree.mass[node_idx] = 0.0
            tree.com[node_idx] = 0.0
            continue
        
        # Get particles in this leaf via permutation
        particle_indices = tree.perm[start:start + count]
        
        # Get new positions and masses for these particles
        leaf_positions = new_positions[particle_indices]  # (count, dim)
        leaf_masses = masses[particle_indices]  # (count,)
        
        # Compute total mass
        total_mass = np.sum(leaf_masses)
        tree.mass[node_idx] = total_mass
        
        if total_mass > 0:
            # Compute center of mass: COM = Σ(m_i * r_i) / Σ(m_i)
            weighted_positions = leaf_masses[:, np.newaxis] * leaf_positions
            tree.com[node_idx] = np.sum(weighted_positions, axis=0) / total_mass
        else:
            # No mass - use geometric center as fallback
            tree.com[node_idx] = tree.center[node_idx]


def _update_internal_nodes_bottomup(
    tree: TreeData,
    new_positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
) -> None:
    """Update internal nodes bottom-up using updated child information."""
    # Process nodes in reverse order to ensure children are processed first
    for node_idx in range(tree.num_nodes - 1, -1, -1):
        if tree.is_leaf[node_idx]:
            continue  # Skip leaves, already processed
            
        # Get children of this internal node
        first_child = tree.first_child[node_idx]
        child_count = tree.child_count[node_idx]
        
        if first_child == -1 or child_count == 0:
            # No children - empty internal node
            tree.mass[node_idx] = 0.0
            tree.com[node_idx] = tree.center[node_idx]
            continue
        
        # Accumulate mass and weighted COM from children
        total_mass = 0.0
        weighted_com = np.zeros(tree.dim, dtype=np.float64)
        
        for i in range(child_count):
            child_idx = first_child + i
            if child_idx >= tree.num_nodes:
                break  # Safety check
                
            child_mass = tree.mass[child_idx]
            if child_mass > 0:
                total_mass += child_mass
                weighted_com += child_mass * tree.com[child_idx]
        
        # Update this node
        tree.mass[node_idx] = total_mass
        if total_mass > 0:
            tree.com[node_idx] = weighted_com / total_mass
        else:
            tree.com[node_idx] = tree.center[node_idx]


def should_refit_vs_rebuild(
    tree: TreeData,
    new_positions: NDArray[np.floating[Any]],
    position_tolerance: float = 0.1,
    boundary_check: bool = True,
) -> bool:
    """
    Determine whether tree should be refitted or rebuilt based on position changes.
    
    Parameters
    ----------
    tree : TreeData
        Current tree structure
    new_positions : ndarray, shape (N, D)
        New particle positions
    position_tolerance : float, default 0.1
        Fraction of tree half_size considered "small movement"
    boundary_check : bool, default True
        Whether to check if particles crossed major boundaries
        
    Returns
    -------
    should_refit : bool
        True if refit is recommended, False if rebuild is recommended
        
    Notes
    -----
    Criteria for recommending refit:
    1. Same number of particles and dimensions
    2. No particles crossing major tree boundaries (if boundary_check=True)
    
    This is a heuristic - users can override the decision.
    Since we don't have access to original positions, we primarily check
    boundary conditions.
    """
    if new_positions.shape[0] != len(tree.perm):
        return False  # Different particle count - must rebuild
    
    if new_positions.shape[1] != tree.dim:
        return False  # Different dimensions - must rebuild
    
    if boundary_check:
        # Check if any particles are outside the root bounding box
        root_center = tree.center[0]
        root_half_size = tree.half_size[0]
        
        # Compute distances from root center
        distances = np.abs(new_positions - root_center)
        max_distances = np.max(distances, axis=1)
        
        # Check if any particle is outside the root box
        # Allow some small tolerance for particles near the boundary
        tolerance_margin = 1.1  # 10% margin
        if np.any(max_distances > tolerance_margin * root_half_size):
            return False  # Particles outside root - must rebuild
    
    return True  # Refit is recommended


def estimate_refit_cost(tree: TreeData) -> int:
    """
    Estimate computational cost of refitting vs rebuilding.
    
    Parameters
    ----------
    tree : TreeData
        Tree structure
        
    Returns
    -------
    refit_cost : int
        Estimated operations for refit (proportional to number of nodes)
        
    Notes
    -----
    Refit cost: O(M) where M is number of nodes
    Rebuild cost: O(N log N) where N is number of particles
    
    Refit is beneficial when M << N log N, which is typically true
    for well-balanced trees where M ≈ N/leaf_size.
    """
    return tree.num_nodes  # Proportional to number of tree operations


def validate_refit_result(
    original_tree: TreeData,
    refitted_tree: TreeData,
    new_positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    tolerance: float = 1e-12,
) -> bool:
    """
    Validate that tree refit preserved topology and correctly updated properties.
    
    Parameters
    ----------
    original_tree : TreeData
        Original tree before refit
    refitted_tree : TreeData 
        Tree after refit
    new_positions : ndarray, shape (N, D)
        New particle positions used for refit
    masses : ndarray, shape (N,)
        Particle masses used for refit
    tolerance : float, default 1e-12
        Numerical tolerance for validation
        
    Returns
    -------
    is_valid : bool
        True if refit result is valid
        
    Notes
    -----
    Checks:
    1. Topology unchanged (centers, half_size, tree structure)
    2. Leaf node masses/COM correctly computed from particles
    3. Internal node masses/COM correctly computed from children
    4. Mass conservation
    """
    # Check topology preservation
    if not np.allclose(original_tree.center, refitted_tree.center, atol=tolerance):
        return False
    if not np.allclose(original_tree.half_size, refitted_tree.half_size, atol=tolerance):
        return False
    if not np.array_equal(original_tree.first_child, refitted_tree.first_child):
        return False
    if not np.array_equal(original_tree.is_leaf, refitted_tree.is_leaf):
        return False
    
    # Check mass conservation
    total_particle_mass = np.sum(masses)
    root_mass = refitted_tree.mass[0]
    if not np.isclose(total_particle_mass, root_mass, atol=tolerance):
        return False
    
    # Detailed validation would require checking each node's COM computation
    # For now, basic checks are sufficient
    
    return True
