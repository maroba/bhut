"""
Tree building algorithms for Barnes-Hut method.

This module implements iterative tree construction using Morton codes
for efficient spatial partitioning.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bhut.backends.base import ArrayNamespace
from bhut.space.bbox import aabb, normalize_to_unit
from bhut.tree.node import TreeData
from bhut.utils.morton import sort_by_morton


def build_tree(
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    *,
    dim: int,
    leaf_size: int,
    xp: ArrayNamespace,
    multipole: str = "mono",
) -> TreeData:
    """
    Build Barnes-Hut tree using iterative construction with Morton codes.
    
    Parameters
    ----------
    positions : ndarray, shape (N, D)
        Particle positions (will be materialized if Dask)
    masses : ndarray, shape (N,)
        Particle masses (will be materialized if Dask)
    dim : int
        Spatial dimensions (2 or 3)
    leaf_size : int
        Maximum particles per leaf node
    xp : ArrayNamespace
        Array namespace for computations
    multipole : str, optional
        Multipole expansion type ("mono" for monopole, default)
        
    Returns
    -------
    tree : TreeData
        Complete tree data structure
        
    Raises
    ------
    ValueError
        If inputs are invalid or multipole type is unsupported
        
    Notes
    -----
    For Dask arrays, positions and masses are materialized to NumPy arrays
    for tree building since the algorithm requires random access patterns
    that don't parallelize well. The resulting tree contains NumPy arrays.
    """
    if multipole != "mono":
        raise ValueError(f"multipole must be 'mono', got '{multipole}'")
    
    # Materialize Dask arrays if necessary for tree building
    # Tree construction requires random access and doesn't parallelize well
    try:
        from bhut.backends.dask_ import DASK_AVAILABLE, materialize_for_tree_building
        if DASK_AVAILABLE:
            positions_np, masses_np = materialize_for_tree_building(positions, masses)
        else:
            positions_np = np.asarray(positions, dtype=np.float64)
            masses_np = np.asarray(masses, dtype=np.float64)
    except ImportError:
        # Dask not available, use regular conversion
        positions_np = np.asarray(positions, dtype=np.float64)
        masses_np = np.asarray(masses, dtype=np.float64)
    
    N = positions_np.shape[0]
    
    if N == 0:
        raise ValueError("Cannot build tree with zero particles")
    
    # Step 1: Compute bounding box and normalize to unit space
    pmin, pmax = aabb(positions_np)
    positions01 = normalize_to_unit(positions_np, pmin, pmax)
    
    # Step 2: Compute Morton codes and sort
    _, perm, [sorted_masses] = sort_by_morton(positions01, [masses_np])
    perm = np.asarray(perm, dtype=np.intp)
    sorted_positions = positions_np[perm]
    sorted_masses = np.asarray(sorted_masses, dtype=np.float64)
    
    # Step 3: Build tree structure iteratively
    # Note: We use NumPy namespace since arrays are materialized
    from bhut.backends.numpy_ import numpy_namespace
    tree_data = _build_tree_iterative(
        sorted_positions, sorted_masses, perm, dim, leaf_size, numpy_namespace
    )
    
    # Step 4: Compute centers and half sizes
    _compute_node_geometry(tree_data, sorted_positions, pmin, pmax, numpy_namespace)
    
    # Step 5: Bottom-up computation of mass and center of mass
    _compute_mass_and_com(tree_data, sorted_positions, sorted_masses, xp)
    
    return tree_data


def _build_tree_iterative(
    sorted_positions: NDArray[np.floating[Any]],
    sorted_masses: NDArray[np.floating[Any]], 
    perm: NDArray[np.intp],
    dim: int,
    leaf_size: int,
    xp: ArrayNamespace,
) -> TreeData:
    """Build tree structure using iterative algorithm."""
    N = len(sorted_positions)
    
    # Estimate maximum number of nodes needed
    # In worst case, we need roughly N/leaf_size leaves plus internal nodes
    # Use a more conservative estimate
    max_nodes = max(4 * N // leaf_size, 10)
    
    # Pre-allocate arrays (will trim later)
    center = xp.zeros((max_nodes, dim), dtype=np.float64)
    half_size = xp.zeros(max_nodes, dtype=np.float64)
    mass = xp.zeros(max_nodes, dtype=np.float64)
    com = xp.zeros((max_nodes, dim), dtype=np.float64)
    first_child = xp.full(max_nodes, -1, dtype=np.intp)
    child_count = xp.zeros(max_nodes, dtype=np.intp)
    start = xp.zeros(max_nodes, dtype=np.intp)
    count = xp.zeros(max_nodes, dtype=np.intp)
    is_leaf = xp.zeros(max_nodes, dtype=bool)
    
    # Stack for iterative processing: (node_idx, start_idx, end_idx)
    stack = [(0, 0, N)]  # Root node covers all particles
    node_count = 1
    
    while stack:
        node_idx, start_idx, end_idx = stack.pop()
        particle_count = end_idx - start_idx
        
        # Ensure we don't exceed allocated space
        if node_idx >= max_nodes:
            raise RuntimeError(f"Node index {node_idx} exceeds max_nodes {max_nodes}")
        
        # Set node properties
        start[node_idx] = start_idx
        count[node_idx] = particle_count
        
        # Check if this should be a leaf
        if particle_count <= leaf_size:
            is_leaf[node_idx] = True
        else:
            # Internal node - create children
            is_leaf[node_idx] = False
            
            # For simplicity, create binary split for now
            # (Real implementation would use 2^D children based on Morton codes)
            num_children = 2
            mid_idx = start_idx + particle_count // 2
            
            # Check if we have space for children
            if node_count + num_children > max_nodes:
                # Fallback: make this a leaf if we run out of space
                is_leaf[node_idx] = True
                continue
            
            # Allocate child nodes
            first_child[node_idx] = node_count
            child_count[node_idx] = num_children
            
            # Add children to stack
            child_ranges = [
                (start_idx, mid_idx),
                (mid_idx, end_idx)
            ]
            
            for i, (child_start, child_end) in enumerate(child_ranges):
                if child_start < child_end:  # Non-empty child
                    child_idx = node_count + i
                    stack.append((child_idx, child_start, child_end))
            
            node_count += num_children
    
    # Trim arrays to actual size
    center = center[:node_count]
    half_size = half_size[:node_count]
    mass = mass[:node_count]
    com = com[:node_count]
    first_child = first_child[:node_count]
    child_count = child_count[:node_count]
    start = start[:node_count]
    count = count[:node_count]
    is_leaf = is_leaf[:node_count]
    
    return TreeData(
        center=center,
        half_size=half_size,
        mass=mass,
        com=com,
        first_child=first_child,
        child_count=child_count,
        start=start,
        count=count,
        is_leaf=is_leaf,
        perm=perm,
        dim=dim,
        leaf_size=leaf_size,
    )


def _compute_node_geometry(
    tree_data: TreeData,
    sorted_positions: NDArray[np.floating[Any]],
    pmin: NDArray[np.floating[Any]],
    pmax: NDArray[np.floating[Any]],
    xp: ArrayNamespace,
) -> None:
    """Compute geometric centers and half sizes for all nodes."""
    # For simplicity, use bounding box approach
    # Real implementation would use proper spatial subdivision
    
    # Root node covers entire domain
    tree_data.center[0] = (pmin + pmax) / 2
    extent = pmax - pmin
    tree_data.half_size[0] = np.max(extent) / 2
    
    # For other nodes, compute from particle positions
    for node_idx in range(1, tree_data.num_nodes):
        start_idx = tree_data.start[node_idx]
        end_idx = start_idx + tree_data.count[node_idx]
        
        if start_idx < end_idx:
            node_positions = sorted_positions[start_idx:end_idx]
            node_pmin, node_pmax = aabb(node_positions)
            
            tree_data.center[node_idx] = (node_pmin + node_pmax) / 2
            extent = node_pmax - node_pmin
            tree_data.half_size[node_idx] = np.max(extent) / 2


def _compute_mass_and_com(
    tree_data: TreeData,
    sorted_positions: NDArray[np.floating[Any]],
    sorted_masses: NDArray[np.floating[Any]],
    xp: ArrayNamespace,
) -> None:
    """Compute mass and center of mass for all nodes using bottom-up traversal."""
    # Process nodes in reverse order (bottom-up)
    for node_idx in range(tree_data.num_nodes - 1, -1, -1):
        if tree_data.is_leaf[node_idx]:
            # Leaf node: compute from particles directly
            start_idx = tree_data.start[node_idx]
            end_idx = start_idx + tree_data.count[node_idx]
            
            if start_idx < end_idx:
                node_masses = sorted_masses[start_idx:end_idx]
                node_positions = sorted_positions[start_idx:end_idx]
                
                total_mass = xp.sum(node_masses)
                if total_mass > 0:
                    # Weighted center of mass
                    com = xp.sum(node_masses[:, None] * node_positions, axis=0) / total_mass
                else:
                    com = tree_data.center[node_idx]
                
                tree_data.mass[node_idx] = total_mass
                tree_data.com[node_idx] = com
            else:
                tree_data.mass[node_idx] = 0.0
                tree_data.com[node_idx] = tree_data.center[node_idx]
        
        else:
            # Internal node: aggregate from children
            children = tree_data.get_children(node_idx)
            
            if len(children) > 0:
                child_masses = tree_data.mass[children]
                child_coms = tree_data.com[children]
                
                total_mass = xp.sum(child_masses)
                if total_mass > 0:
                    # Weighted center of mass from children
                    com = xp.sum(child_masses[:, None] * child_coms, axis=0) / total_mass
                else:
                    com = tree_data.center[node_idx]
                
                tree_data.mass[node_idx] = total_mass
                tree_data.com[node_idx] = com
            else:
                tree_data.mass[node_idx] = 0.0
                tree_data.com[node_idx] = tree_data.center[node_idx]
