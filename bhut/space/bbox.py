"""
Bounding box utilities for spatial operations.

This module provides functions for computing bounding boxes and
normalizing point coordinates to unit space.
"""

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def aabb(points: NDArray[np.floating[Any]]) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Compute axis-aligned bounding box (AABB) for a set of points.
    
    Parameters
    ----------
    points : ndarray, shape (N, D)
        Input points
        
    Returns
    -------
    pmin : ndarray, shape (D,)
        Minimum coordinates along each axis
    pmax : ndarray, shape (D,)
        Maximum coordinates along each axis
        
    Raises
    ------
    ValueError
        If points array is empty or not 2D
    """
    points = np.asarray(points, dtype=np.float64)
    
    if len(points.shape) != 2:
        raise ValueError(f"points must be 2D array, got shape {points.shape}")
    
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")
    
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    
    return pmin, pmax


def normalize_to_unit(
    points: NDArray[np.floating[Any]], 
    pmin: NDArray[np.floating[Any]], 
    pmax: NDArray[np.floating[Any]], 
    eps: float = 1e-12
) -> NDArray[np.floating[Any]]:
    """
    Normalize points to unit hypercube [0, 1)^D.
    
    Parameters
    ----------
    points : ndarray, shape (N, D)
        Input points to normalize
    pmin : ndarray, shape (D,)
        Minimum coordinates (from aabb)
    pmax : ndarray, shape (D,)
        Maximum coordinates (from aabb)
    eps : float, optional
        Small epsilon to prevent points from reaching 1.0 (default: 1e-12)
        
    Returns
    -------
    points01 : ndarray, shape (N, D)
        Points normalized to [0, 1)^D
        
    Notes
    -----
    For dimensions where pmin == pmax (no extent), points are mapped to 0.5.
    The eps parameter ensures no point reaches exactly 1.0, which is required
    for Morton code computation.
    """
    points = np.asarray(points, dtype=np.float64)
    pmin = np.asarray(pmin, dtype=np.float64)
    pmax = np.asarray(pmax, dtype=np.float64)
    
    # Compute extent along each dimension
    extent = pmax - pmin
    
    # Handle degenerate dimensions (where all points have same coordinate)
    # Map to center of unit interval (0.5)
    extent = np.where(extent <= 0, 1.0, extent)
    center_offset = np.where(pmax - pmin <= 0, 0.5, 0.0)
    
    # Normalize to [0, 1] then subtract epsilon to ensure < 1
    points01 = (points - pmin) / extent + center_offset
    points01 = np.minimum(points01, 1.0 - eps)
    
    return points01
