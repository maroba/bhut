"""
Tree node data structures for Barnes-Hut algorithm.

This module defines the TreeData dataclass that stores all tree information
in Structure-of-Arrays (SoA) format for efficient computation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TreeData:
    """
    Structure-of-Arrays representation of Barnes-Hut tree.
    
    This dataclass stores all tree node information in flat arrays,
    which is more cache-friendly and efficient than pointer-based trees.
    
    Parameters
    ----------
    center : ndarray, shape (M, D)
        Center coordinates of each tree node
    half_size : ndarray, shape (M,)
        Half the side length of each node's bounding box
    mass : ndarray, shape (M,)
        Total mass contained in each node
    com : ndarray, shape (M, D)
        Center of mass coordinates for each node
    first_child : ndarray, shape (M,)
        Index of first child node (-1 if leaf)
    child_count : ndarray, shape (M,)
        Number of child nodes (0 for leaves, 2^D for internal nodes)
    start : ndarray, shape (M,)
        Starting index in permuted particle arrays for this node
    count : ndarray, shape (M,)
        Number of particles contained in this node (including children)
    is_leaf : ndarray, shape (M,)
        Boolean array indicating if node is a leaf
    perm : ndarray, shape (N,)
        Permutation array mapping original particle indices to sorted order
    dim : int
        Spatial dimensions (2 or 3)
    leaf_size : int
        Maximum particles per leaf node
        
    Notes
    -----
    - M is the total number of tree nodes
    - N is the number of particles
    - Root node is always at index 0
    - For internal nodes, children are stored contiguously starting at first_child
    - For 2D: 4 children (quadtree), for 3D: 8 children (octree)
    """
    
    # Node properties (M nodes)
    center: NDArray[np.floating[Any]]       # (M, D) - geometric center
    half_size: NDArray[np.floating[Any]]    # (M,) - half bounding box size
    mass: NDArray[np.floating[Any]]         # (M,) - total mass in node
    com: NDArray[np.floating[Any]]          # (M, D) - center of mass
    
    # Tree structure (M nodes)
    first_child: NDArray[np.intp]           # (M,) - index of first child (-1 if leaf)
    child_count: NDArray[np.intp]           # (M,) - number of children
    start: NDArray[np.intp]                 # (M,) - start index in particle arrays
    count: NDArray[np.intp]                 # (M,) - particle count in subtree
    is_leaf: NDArray[np.bool_]              # (M,) - leaf node flag
    
    # Particle permutation (N particles)
    perm: NDArray[np.intp]                  # (N,) - permutation to sorted order
    
    # Metadata
    dim: int                                # spatial dimensions
    leaf_size: int                          # max particles per leaf
    
    @property
    def num_nodes(self) -> int:
        """Total number of nodes in the tree."""
        return len(self.center)
    
    @property
    def num_particles(self) -> int:
        """Total number of particles."""
        return len(self.perm)
    
    def validate(self) -> None:
        """Validate tree data structure consistency."""
        M = self.num_nodes
        N = self.num_particles
        D = self.dim
        
        # Check dimensions
        if D not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {D}")
        
        # Check array shapes
        arrays_M = [
            ("center", self.center, (M, D)),
            ("half_size", self.half_size, (M,)),
            ("mass", self.mass, (M,)),
            ("com", self.com, (M, D)),
            ("first_child", self.first_child, (M,)),
            ("child_count", self.child_count, (M,)),
            ("start", self.start, (M,)),
            ("count", self.count, (M,)),
            ("is_leaf", self.is_leaf, (M,)),
        ]
        
        for name, arr, expected_shape in arrays_M:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")
        
        if self.perm.shape != (N,):
            raise ValueError(f"perm has shape {self.perm.shape}, expected ({N},)")
        
        # Check tree structure
        if M > 0:
            # Root node should be at index 0
            if self.count[0] != N:
                raise ValueError(f"Root node count {self.count[0]} != num_particles {N}")
            
            # Check permutation is valid
            if not np.array_equal(np.sort(self.perm), np.arange(N)):
                raise ValueError("perm is not a valid permutation")
            
            # Check child indices are valid
            valid_child_mask = self.first_child >= 0
            if np.any(self.first_child[valid_child_mask] >= M):
                raise ValueError("first_child indices exceed number of nodes")
            
            # Check leaf_size constraint
            leaf_mask = self.is_leaf
            if np.any(self.count[leaf_mask] > self.leaf_size):
                raise ValueError("Some leaf nodes exceed leaf_size")
    
    def get_children(self, node_idx: int) -> NDArray[np.intp]:
        """
        Get indices of child nodes for given node.
        
        Parameters
        ----------
        node_idx : int
            Index of parent node
            
        Returns
        -------
        children : ndarray
            Array of child node indices (empty if leaf)
        """
        if self.is_leaf[node_idx]:
            return np.array([], dtype=np.intp)
        
        first = self.first_child[node_idx]
        count = self.child_count[node_idx]
        return np.arange(first, first + count, dtype=np.intp)
    
    def get_particles(self, node_idx: int) -> NDArray[np.intp]:
        """
        Get original particle indices for particles in this node.
        
        Parameters
        ----------
        node_idx : int
            Node index
            
        Returns
        -------
        particle_indices : ndarray
            Original particle indices (before permutation)
        """
        start = self.start[node_idx]
        count = self.count[node_idx]
        
        # Get permuted indices for this node
        perm_indices = np.arange(start, start + count, dtype=np.intp)
        
        # Map back to original indices
        return self.perm[perm_indices]
