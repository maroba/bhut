"""
Gravitational force kernels for Barnes-Hut algorithm.

This module provides efficient implementations of gravitational force
calculations with various softening schemes.
"""

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray


def acc_monopole(
    dx: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: Union[float, NDArray[np.floating[Any]]],
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Compute gravitational acceleration using monopole approximation with Plummer softening.
    
    This function computes the acceleration on a target due to source particles using
    the standard gravitational force law with Plummer softening to avoid singularities.
    
    Parameters
    ----------
    dx : ndarray, shape (N, D) or (D,)
        Displacement vectors from target to sources: dx = source_pos - target_pos
    masses : ndarray, shape (N,) or scalar
        Source masses
    softening : float or ndarray, shape (N,) or scalar
        Plummer softening length(s)
    G : float
        Gravitational constant
        
    Returns
    -------
    acceleration : ndarray, shape (D,) or (N, D)
        Gravitational acceleration vector(s)
        
    Notes
    -----
    The acceleration is computed as:
    a = G * m * dx / (r² + ε²)^(3/2)
    
    Where:
    - r = |dx| is the distance
    - ε is the softening length
    - The softening prevents singularities when r → 0
    """
    dx = np.asarray(dx, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    softening = np.asarray(softening, dtype=np.float64)
    
    # Handle both single vector and array of vectors
    if dx.ndim == 1:
        # Single displacement vector
        r2 = np.sum(dx**2)
        r2_soft = r2 + softening**2
        
        if r2_soft == 0:
            return np.zeros_like(dx)
        
        # Force magnitude per unit mass: G * m / (r² + ε²)^(3/2)
        force_mag = G * masses / (r2_soft ** 1.5)
        
        # Force vector
        return force_mag * dx
    
    else:
        # Array of displacement vectors (N, D)
        r2 = np.sum(dx**2, axis=1)  # (N,)
        
        # Handle softening (scalar or array)
        if np.isscalar(softening):
            r2_soft = r2 + softening**2
        else:
            r2_soft = r2 + softening**2
        
        # Avoid division by zero
        nonzero = r2_soft > 0
        
        # Initialize result
        acceleration = np.zeros_like(dx)
        
        if np.any(nonzero):
            # Force magnitude per unit mass for non-zero distances
            force_mag = np.zeros_like(r2)
            force_mag[nonzero] = G * masses[nonzero] / (r2_soft[nonzero] ** 1.5)
            
            # Force vectors
            acceleration[nonzero] = force_mag[nonzero, np.newaxis] * dx[nonzero]
        
        return acceleration


def acc_monopole_single_target(
    target_pos: NDArray[np.floating[Any]],
    source_positions: NDArray[np.floating[Any]],
    source_masses: NDArray[np.floating[Any]],
    softening: Union[float, NDArray[np.floating[Any]]],
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Compute total acceleration on a single target from multiple sources.
    
    Parameters
    ----------
    target_pos : ndarray, shape (D,)
        Target position
    source_positions : ndarray, shape (N, D)
        Source particle positions
    source_masses : ndarray, shape (N,)
        Source particle masses
    softening : float or ndarray, shape (N,)
        Softening length(s)
    G : float
        Gravitational constant
        
    Returns
    -------
    acceleration : ndarray, shape (D,)
        Total gravitational acceleration on target
    """
    # Displacement vectors from target to sources
    dx = source_positions - target_pos[np.newaxis, :]
    
    # Compute individual accelerations
    individual_acc = acc_monopole(dx, source_masses, softening, G)
    
    # Sum over all sources
    return np.sum(individual_acc, axis=0)


def acc_monopole_node(
    target_pos: NDArray[np.floating[Any]],
    node_com: NDArray[np.floating[Any]],
    node_mass: float,
    softening: float,
    G: float,
) -> NDArray[np.floating[Any]]:
    """
    Compute acceleration on target from a single tree node (monopole approximation).
    
    Parameters
    ----------
    target_pos : ndarray, shape (D,)
        Target position
    node_com : ndarray, shape (D,)
        Node center of mass
    node_mass : float
        Total mass in node
    softening : float
        Softening length
    G : float
        Gravitational constant
        
    Returns
    -------
    acceleration : ndarray, shape (D,)
        Gravitational acceleration from node
    """
    # Displacement from target to node COM
    dx = node_com - target_pos
    
    # Use single-vector monopole kernel
    return acc_monopole(dx, node_mass, softening, G)


def validate_kernel_inputs(
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    softening: Union[float, NDArray[np.floating[Any]]],
) -> None:
    """Validate inputs for acceleration kernels."""
    if len(positions.shape) != 2:
        raise ValueError(f"positions must be 2D array, got shape {positions.shape}")
    
    if len(masses.shape) != 1:
        raise ValueError(f"masses must be 1D array, got shape {masses.shape}")
    
    if positions.shape[0] != masses.shape[0]:
        raise ValueError(
            f"positions has {positions.shape[0]} particles "
            f"but masses has {masses.shape[0]} elements"
        )
    
    if not np.isscalar(softening):
        softening = np.asarray(softening)
        if len(softening.shape) != 1:
            raise ValueError(f"softening must be scalar or 1D array, got shape {softening.shape}")
        if softening.shape[0] != positions.shape[0]:
            raise ValueError(
                f"softening array has {softening.shape[0]} elements "
                f"but need {positions.shape[0]}"
            )
