"""
Morton code utilities for spatial sorting.

This module provides vectorized Morton code computation for efficient
spatial sorting of particles in 2D and 3D spaces.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _expand_bits_2d(x: NDArray[Any]) -> NDArray[Any]:
    """Expand bits for 2D Morton encoding (interleave with zeros)."""
    # Convert to integers in range [0, 2^bits-1]
    x = x.astype(np.uint64)
    
    # Expand bits by interleaving with zeros
    # Original: abcdefgh...
    # Result:   a0b0c0d0e0f0g0h0...
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    
    return x


def _expand_bits_3d(x: NDArray[Any]) -> NDArray[Any]:
    """Expand bits for 3D Morton encoding (interleave with two zeros)."""
    # Convert to integers in range [0, 2^bits-1]
    x = x.astype(np.uint64)
    
    # Expand bits by interleaving with two zeros
    # Original: abcdefgh...
    # Result:   a00b00c00d00e00f00g00h00...
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    
    return x


def morton_codes(points01: NDArray[np.floating[Any]], bits: int = 21) -> NDArray[np.uint64]:
    """
    Compute Morton codes for points in unit hypercube [0,1)^D.
    
    Morton codes provide a space-filling curve that preserves spatial locality,
    making them ideal for tree construction and spatial sorting.
    
    Parameters
    ----------
    points01 : ndarray, shape (N, D)
        Points in unit hypercube [0, 1)^D where D is 2 or 3
    bits : int, optional
        Number of bits per dimension (default: 21, giving 42-bit or 63-bit codes)
        
    Returns
    -------
    codes : ndarray, shape (N,), dtype uint64
        Morton codes for each point
        
    Raises
    ------
    ValueError
        If points are not in [0, 1) or dimension is not 2 or 3
    """
    points01 = np.asarray(points01, dtype=np.float64)
    
    if len(points01.shape) != 2:
        raise ValueError(f"points01 must be 2D array, got shape {points01.shape}")
    
    N, D = points01.shape
    
    if D not in (2, 3):
        raise ValueError(f"points01 must have 2 or 3 dimensions, got {D}")
    
    # Check bounds
    if np.any(points01 < 0) or np.any(points01 >= 1):
        raise ValueError("All points must be in range [0, 1)")
    
    # Convert to integer coordinates
    max_coord = (1 << bits) - 1
    coords = (points01 * (1 << bits)).astype(np.uint64)
    coords = np.minimum(coords, max_coord)  # Clamp to avoid overflow
    
    if D == 2:
        # 2D Morton codes: interleave x and y coordinates
        x = _expand_bits_2d(coords[:, 0])
        y = _expand_bits_2d(coords[:, 1])
        return x | (y << 1)
    
    else:  # D == 3
        # 3D Morton codes: interleave x, y, and z coordinates
        x = _expand_bits_3d(coords[:, 0])
        y = _expand_bits_3d(coords[:, 1])
        z = _expand_bits_3d(coords[:, 2])
        return x | (y << 1) | (z << 2)


def sort_by_morton(
    points01: NDArray[np.floating[Any]], 
    data_arrays: list[NDArray[Any]] | None = None,
    bits: int = 21
) -> tuple[NDArray[np.uint64], NDArray[np.intp], list[NDArray[Any]]]:
    """
    Sort points by Morton code and permute associated data arrays.
    
    Parameters
    ----------
    points01 : ndarray, shape (N, D)
        Points in unit hypercube [0, 1)^D
    data_arrays : list of ndarray, optional
        Additional arrays to permute along with points (each shape (N, ...))
    bits : int, optional
        Number of bits per dimension for Morton codes
        
    Returns
    -------
    codes : ndarray, shape (N,)
        Morton codes in sorted order
    perm : ndarray, shape (N,)
        Permutation indices (original indices in sorted order)
    sorted_data : list of ndarray
        Data arrays permuted to match sorted order
    """
    # Compute Morton codes
    codes = morton_codes(points01, bits)
    
    # Sort by Morton code (stable sort to handle ties deterministically)
    perm = np.argsort(codes, kind='stable')
    sorted_codes = codes[perm]
    
    # Permute data arrays if provided
    sorted_data = []
    if data_arrays is not None:
        for arr in data_arrays:
            arr = np.asarray(arr)
            if arr.shape[0] != len(points01):
                raise ValueError(f"Data array has {arr.shape[0]} elements but need {len(points01)}")
            sorted_data.append(arr[perm])
    
    return sorted_codes, perm, sorted_data
