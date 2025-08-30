"""
Public API for bhut Barnes-Hut N-body accelerator.

This module provides both functional and object-oriented interfaces
for N-body force calculations.
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bhut.backends.base import ArrayNamespace, get_namespace, validate_compatible_shapes, validate_shape
from bhut.tree.build import build_tree as _build_tree_impl
from bhut.tree.node import TreeData
from bhut.tree.refit import refit_tree, should_refit_vs_rebuild


def _is_dask_array(array: ArrayLike) -> bool:
    """Check if array is a Dask array."""
    try:
        from bhut.backends.dask_ import detect_dask_array
        return detect_dask_array(array)
    except ImportError:
        return False


def _accelerations_dask(
    positions: ArrayLike,
    masses: ArrayLike,
    *,
    theta: float,
    softening: Union[float, ArrayLike],
    G: float,
    dim: int,
    leaf_size: int,
    xp: ArrayNamespace,
) -> ArrayLike:
    """
    Compute accelerations for Dask arrays using map_blocks.
    
    Strategy:
    1. Materialize positions/masses for tree building
    2. Build tree once using materialized data
    3. Use map_blocks to compute accelerations for each chunk of targets
    4. Return Dask array with same chunking as input positions
    """
    from bhut.backends.dask_ import materialize_for_tree_building
    
    # Materialize source data for tree building
    positions_np, masses_np = materialize_for_tree_building(positions, masses)
    
    # Build tree using NumPy arrays
    tree = build_tree(positions_np, masses_np, leaf_size=leaf_size, backend="numpy", dim=dim)
    
    # Define chunk function for map_blocks
    def _compute_chunk_accelerations(targets_chunk, *, tree_data, source_pos, source_masses, 
                                   theta_val, softening_val, G_val):
        """Compute accelerations for a chunk of target positions."""
        from .traverse.bh import barnes_hut_accelerations_targets
        
        # Compute accelerations for this chunk
        return barnes_hut_accelerations_targets(
            tree_data, source_pos, source_masses, targets_chunk, 
            softening_val, theta_val, G_val
        )
    
    # Use map_blocks to compute accelerations preserving chunking
    return xp.map_blocks_accelerations(
        _compute_chunk_accelerations,
        positions,  # targets (Dask array)
        tree,       # tree data
        positions_np,  # source positions (NumPy)
        masses_np,     # source masses (NumPy)
        theta_val=theta,
        softening_val=softening,
        G_val=G
    )

# Type alias for array-like objects
Array = NDArray[np.floating[Any]]


def _validate_dimensions(dim: int) -> None:
    """Validate that dim is 2 or 3."""
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def _validate_accelerations_inputs(
    positions: ArrayLike,
    masses: ArrayLike,
    *,
    dim: int,
    softening: Union[float, ArrayLike],
    theta: float,
    G: float,
    leaf_size: int,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], Union[float, NDArray[np.floating[Any]]]]:
    """
    Validate and convert inputs for accelerations calculation.
    
    Returns
    -------
    positions : ndarray
        Validated positions array, shape (N, dim), dtype float64
    masses : ndarray 
        Validated masses array, shape (N,), dtype float64
    softening : float or ndarray
        Validated softening parameter
    """
    # Validate dimensions parameter
    _validate_dimensions(dim)
    
    # Convert to float64 arrays
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    
    # Validate shapes
    if len(positions.shape) != 2:
        raise ValueError(f"positions must be 2D array, got shape {positions.shape}")
    if len(masses.shape) != 1:
        raise ValueError(f"masses must be 1D array, got shape {masses.shape}")
    
    N, actual_dim = positions.shape
    if actual_dim != dim:
        raise ValueError(f"positions has {actual_dim} dimensions but dim={dim}")
    
    if masses.shape[0] != N:
        raise ValueError(f"positions has {N} particles but masses has {masses.shape[0]} elements")
    
    # Validate softening
    if np.isscalar(softening):
        softening = float(softening)
        if softening < 0:
            raise ValueError(f"softening must be non-negative, got {softening}")
    else:
        softening = np.asarray(softening, dtype=np.float64)
        if len(softening.shape) != 1:
            raise ValueError(f"softening array must be 1D, got shape {softening.shape}")
        if softening.shape[0] != N:
            raise ValueError(f"softening array has {softening.shape[0]} elements but need {N}")
        if np.any(softening < 0):
            raise ValueError("all softening values must be non-negative")
    
    # Validate other parameters
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    if leaf_size <= 0:
        raise ValueError(f"leaf_size must be positive, got {leaf_size}")
    
    return positions, masses, softening


def build_tree(
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]], 
    *,
    leaf_size: int,
    backend: str,
    dim: int,
) -> TreeData:
    """
    Build Barnes-Hut tree structure.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions, shape (N, dim)
    masses : ndarray
        Particle masses, shape (N,)
    leaf_size : int
        Maximum particles per leaf node
    backend : str
        Array backend to use
    dim : int
        Spatial dimensions
        
    Returns
    -------
    tree : TreeData
        Tree data structure
    """
    # Get array namespace
    xp = get_namespace(positions, backend)
    
    # Call actual tree building implementation
    return _build_tree_impl(
        positions, masses, 
        dim=dim, leaf_size=leaf_size, xp=xp, multipole="mono"
    )


    def evaluate_accelerations(
        self, 
        softening: Union[float, NDArray[np.floating[Any]]], 
        theta: float = 0.5,
        G: float = 1.0
    ) -> NDArray[np.floating[Any]]:
        """
        Evaluate gravitational accelerations using the Barnes-Hut algorithm.
        
        Parameters
        ----------
        softening : float or array_like
            Gravitational softening length(s). If scalar, same softening is used
            for all particles. If array, must have shape (N,) for per-particle
            softening.
        theta : float, default 0.5
            Opening angle criterion for Barnes-Hut approximation. Smaller values
            give higher accuracy but slower computation. Typical range: 0.1-1.0.
        G : float, default 1.0
            Gravitational constant
            
        Returns
        -------
        accelerations : array_like, shape (N, D)
            Gravitational accelerations for each particle
            
        Raises
        ------
        RuntimeError
            If tree has not been built (call `build()` first)
        ValueError
            If input parameters have invalid shapes or values
        """
        if self._tree is None:
            raise RuntimeError("Tree not built. Call build() first.")
        
        from .traverse.bh import barnes_hut_accelerations
        
        # Validate inputs
        if not isinstance(softening, (int, float)):
            raise ValueError("Only scalar softening is currently supported")
        
        if softening <= 0:
            raise ValueError("Softening must be positive")
        
        if theta <= 0:
            raise ValueError("Opening angle theta must be positive")
        
        # Convert to numpy arrays for computation
        import numpy as np
        positions_np = np.asarray(self._positions)
        masses_np = np.asarray(self._masses)
        
        # Compute accelerations using Barnes-Hut algorithm
        accelerations_np = barnes_hut_accelerations(
            self._tree, positions_np, masses_np, softening, theta, G
        )
        
        # Convert back to original array type
        return self._xp.asarray(accelerations_np)
def accelerations(
    positions: ArrayLike,
    masses: ArrayLike,
    *,
    theta: float = 0.5,
    softening: Union[float, ArrayLike] = 0.0,
    G: float = 1.0,
    dim: int = 3,
    backend: str = "auto",
    leaf_size: int = 32,
    criterion: str = "bh",
    multipole: str = "mono",
) -> NDArray[np.floating[Any]]:
    """
    Compute gravitational accelerations using the Barnes-Hut algorithm.

    Parameters
    ----------
    positions : array_like
        Particle positions, shape (N, dim)
    masses : array_like
        Particle masses, shape (N,)
    theta : float, optional
        Opening angle criterion. 0.0 gives direct sum, larger values are more approximate.
        Default: 0.5
    softening : float or array_like, optional
        Plummer softening length to avoid singularities. Can be scalar or array of shape (N,).
        Default: 0.0
    G : float, optional
        Gravitational constant. Default: 1.0
    dim : int, optional
        Spatial dimensions (2 or 3). Default: 3
    backend : str, optional
        Array backend ("numpy", "dask", or "auto"). Default: "auto"
    leaf_size : int, optional
        Maximum particles per leaf node. Default: 32
    criterion : str, optional
        Tree opening criterion ("bh" for Barnes-Hut). Default: "bh"
    multipole : str, optional
        Multipole expansion order ("mono" for monopole). Default: "mono"

    Returns
    -------
    accelerations : ndarray
        Gravitational accelerations, shape (N, dim)

    Raises
    ------
    ValueError
        If input shapes are incompatible or parameters are invalid
    """
    # Validate criterion and multipole parameters
    if criterion != "bh":
        raise ValueError(f"criterion must be 'bh', got '{criterion}'")
    if multipole != "mono":
        raise ValueError(f"multipole must be 'mono', got '{multipole}'")
    
    # Get array namespace for backend
    xp = get_namespace(positions, backend)
    
    # Check if we're using Dask backend for special handling
    is_dask = backend == "dask" or (backend == "auto" and _is_dask_array(positions))
    
    if is_dask:
        # For Dask arrays, do basic validation without converting to NumPy
        # Validate dimensions parameter
        _validate_dimensions(dim)
        
        # Validate other parameters
        if theta < 0:
            raise ValueError(f"theta must be non-negative, got {theta}")
        if isinstance(softening, (int, float)) and softening < 0:
            raise ValueError(f"softening must be non-negative, got {softening}")
        if leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {leaf_size}")
        
        # Basic shape validation for Dask arrays
        if len(positions.shape) != 2:
            raise ValueError(f"positions must be 2D array, got shape {positions.shape}")
        if len(masses.shape) != 1:
            raise ValueError(f"masses must be 1D array, got shape {masses.shape}")
        
        N, actual_dim = positions.shape
        if actual_dim != dim:
            raise ValueError(f"positions has {actual_dim} dimensions but dim={dim}")
        
        if masses.shape[0] != N:
            raise ValueError(f"positions has {N} particles but masses has {masses.shape[0]} elements")
        
        # Skip numpy conversion and use specialized computation path
        return _accelerations_dask(
            positions, masses, theta=theta, softening=softening, G=G,
            dim=dim, leaf_size=leaf_size, xp=xp
        )
    else:
        # Standard computation path for NumPy arrays
        # Validate and convert inputs
        positions, masses, softening = _validate_accelerations_inputs(
            positions, masses, 
            dim=dim, softening=softening, theta=theta, G=G, leaf_size=leaf_size
        )
        
        # Build tree structure 
        tree = build_tree(positions, masses, leaf_size=leaf_size, backend=backend, dim=dim)
        
        # Evaluate accelerations
        acc = evaluate_accelerations(
            tree, positions, masses, theta=theta, softening=softening, G=G
        )
        
        return acc


def evaluate_accelerations(
    tree: TreeData,
    positions: NDArray[np.floating[Any]],
    masses: NDArray[np.floating[Any]],
    theta: float = 0.5,
    softening: Union[float, NDArray[np.floating[Any]]] = 0.01,
    G: float = 1.0
) -> NDArray[np.floating[Any]]:
    """
    Evaluate gravitational accelerations using a pre-built Barnes-Hut tree.
    
    Parameters
    ----------
    tree : TreeData
        Pre-built Barnes-Hut tree data structure
    positions : array_like, shape (N, D)
        Particle positions (should match tree construction)  
    masses : array_like, shape (N,)
        Particle masses (should match tree construction)
    theta : float, default 0.5
        Opening angle criterion for Barnes-Hut approximation
    softening : float or array_like, default 0.01
        Gravitational softening length(s)
    G : float, default 1.0
        Gravitational constant
        
    Returns
    -------
    accelerations : array_like, shape (N, D)
        Gravitational accelerations for each particle
    """
    from .traverse.bh import barnes_hut_accelerations
    
    # Validate inputs
    softening_array = np.asarray(softening, dtype=np.float64)
    if softening_array.ndim > 1:
        raise ValueError("Softening must be scalar or 1D array")
    if softening_array.ndim == 1 and len(softening_array) != len(positions):
        raise ValueError("Softening array length must match number of particles")
    if np.any(softening_array < 0):
        raise ValueError("Softening must be non-negative")
    
    # For now, convert array softening to scalar if all values are the same
    if softening_array.ndim == 1:
        if not np.allclose(softening_array, softening_array[0]):
            raise ValueError("Variable softening per particle not yet implemented")
        softening_scalar = float(softening_array[0])
    else:
        softening_scalar = float(softening_array)
    
    if theta < 0:
        raise ValueError("Opening angle theta must be non-negative")
    
    # Convert to numpy arrays for computation
    positions_np = np.asarray(positions)
    masses_np = np.asarray(masses)
    
    # Compute accelerations using Barnes-Hut algorithm
    accelerations_np = barnes_hut_accelerations(
        tree, positions_np, masses_np, softening_scalar, theta, G
    )
    
    return accelerations_np


class Tree:
    """
    Barnes-Hut tree for efficient N-body force calculations.

    This class provides an object-oriented interface to the Barnes-Hut algorithm,
    allowing for tree construction, refitting, and force evaluation.

    Parameters
    ----------
    positions : array_like
        Initial particle positions, shape (N, dim)
    masses : array_like
        Initial particle masses, shape (N,)
    leaf_size : int, optional
        Maximum particles per leaf node. Default: 32
    backend : str, optional
        Array backend ("numpy", "dask", or "auto"). Default: "auto"
    dim : int, optional
        Spatial dimensions (2 or 3). Default: 3
        
    Attributes
    ----------
    positions : ndarray
        Particle positions, shape (N, dim)
    masses : ndarray  
        Particle masses, shape (N,)
    dim : int
        Spatial dimensions
    leaf_size : int
        Maximum particles per leaf node
    backend : str
        Array backend being used
    _tree : TreeData | None
        Internal tree data structure (None until built)
    _is_built : bool
        Whether tree has been built
    """

    def __init__(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        *,
        leaf_size: int = 32,
        backend: str = "auto",
        dim: int = 3,
    ) -> None:
        # Validate and store parameters
        _validate_dimensions(dim)
        
        # Convert and validate inputs
        self.positions, self.masses, _ = _validate_accelerations_inputs(
            positions, masses,
            dim=dim, softening=0.0, theta=0.5, G=1.0, leaf_size=leaf_size
        )
        
        self.dim = dim
        self.leaf_size = leaf_size
        self.backend = backend
        self._tree: TreeData | None = None
        self._is_built = False
        
        # Get namespace for this backend
        self._xp = get_namespace(self.positions, backend)

    def build(self) -> "Tree":
        """
        Build the Barnes-Hut tree structure.

        Returns
        -------
        self : Tree
            Returns self for method chaining
        """
        self._tree = build_tree(
            self.positions, self.masses, 
            leaf_size=self.leaf_size, backend=self.backend, dim=self.dim
        )
        self._is_built = True
        return self

    def refit(self, new_positions: ArrayLike, new_masses: Optional[ArrayLike] = None) -> "Tree":
        """
        Refit the tree with new positions, keeping the same topology.

        This is faster than rebuilding when particles move slightly.
        Uses efficient O(M) algorithm where M is number of tree nodes.

        Parameters
        ----------
        new_positions : array_like
            New particle positions, shape (N, dim)
        new_masses : array_like, optional
            New particle masses, shape (N,). If None, keeps existing masses.

        Returns
        -------
        self : Tree
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If arrays have incompatible shapes or tree is not built
        """
        if not self._is_built or self._tree is None:
            raise ValueError("Tree must be built before refitting. Call build() first.")
        
        # Use existing masses if not provided
        if new_masses is None:
            new_masses = self.masses
        
        # Validate inputs
        new_positions, new_masses, _ = _validate_accelerations_inputs(
            new_positions, new_masses,
            dim=self.dim, softening=0.0, theta=0.5, G=1.0, leaf_size=self.leaf_size
        )
        
        # Check if refit is recommended vs rebuild
        if not should_refit_vs_rebuild(self._tree, new_positions):
            # Fall back to rebuild for major changes
            return self.rebuild(new_positions, new_masses)
        
        # Perform efficient refit
        self._tree = refit_tree(self._tree, new_positions, new_masses, self._xp)
        
        # Update stored arrays
        self.positions = new_positions
        self.masses = new_masses
        
        return self

    def rebuild(
        self, new_positions: ArrayLike, new_masses: Optional[ArrayLike] = None
    ) -> "Tree":
        """
        Rebuild the tree with new positions and optionally new masses.

        Parameters
        ----------
        new_positions : array_like
            New particle positions, shape (N, dim)
        new_masses : array_like, optional
            New particle masses, shape (N,). If None, keeps existing masses.

        Returns
        -------
        self : Tree
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If new arrays have incompatible shapes
        """
        # Use existing masses if not provided
        if new_masses is None:
            new_masses = self.masses
        
        # Validate new inputs
        new_positions, new_masses, _ = _validate_accelerations_inputs(
            new_positions, new_masses,
            dim=self.dim, softening=0.0, theta=0.5, G=1.0, leaf_size=self.leaf_size
        )
        
        # Update stored arrays
        self.positions = new_positions
        self.masses = new_masses
        
        # Mark as needing rebuild
        self._is_built = False
        self._tree = None
        
        # Rebuild tree
        return self.build()

    def accelerations(
        self,
        targets: Optional[ArrayLike] = None,
        *,
        theta: float = 0.5,
        softening: Union[float, ArrayLike] = 0.0,
        G: float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gravitational accelerations for target positions.

        Parameters
        ----------
        targets : array_like, optional
            Target positions, shape (M, dim). If None, evaluates accelerations
            for the particles used to build the tree (self-evaluation).
        theta : float, optional
            Opening angle criterion. Default: 0.5
        softening : float or array_like, optional
            Plummer softening length. Default: 0.0
        G : float, optional
            Gravitational constant. Default: 1.0

        Returns
        -------
        accelerations : ndarray
            Gravitational accelerations, shape (M, dim)
            
        Raises
        ------
        ValueError
            If tree is not built or targets have wrong shape
        """
        if not self._is_built:
            raise ValueError("Tree must be built before evaluating accelerations")
        
        # Handle self-evaluation case
        if targets is None:
            return self.evaluate_accelerations(softening, theta, G)
        
        # Check if targets is a Dask array and handle accordingly
        if _is_dask_array(targets):
            # Use Dask implementation for targets
            return self._accelerations_dask_targets(
                targets, theta=theta, softening=softening, G=G
            )
        
        # NumPy path - validate targets
        targets = np.asarray(targets, dtype=np.float64)
        if len(targets.shape) != 2:
            raise ValueError(f"targets must be 2D array, got shape {targets.shape}")
        if targets.shape[1] != self.dim:
            raise ValueError(
                f"targets has {targets.shape[1]} dimensions but tree has {self.dim}"
            )
        
        # Validate softening for target array size
        if not np.isscalar(softening):
            softening = np.asarray(softening, dtype=np.float64)
            M = targets.shape[0]
            if softening.shape != (M,):
                raise ValueError(
                    f"softening array shape {softening.shape} does not match "
                    f"targets shape ({M},)"
                )
        
        # Validate other parameters
        if theta < 0:
            raise ValueError(f"theta must be non-negative, got {theta}")
        
        # Evaluate accelerations using tree  
        # For targets, we need to compute the acceleration at each target point
        # Use the specialized function for external targets
        from .traverse.bh import barnes_hut_accelerations_targets
        
        # Compute accelerations at target positions
        acc = barnes_hut_accelerations_targets(
            self._tree, self.positions, self.masses, targets, softening, theta, G
        )
        
        return acc

    def _accelerations_dask_targets(
        self,
        targets: ArrayLike,
        *,
        theta: float = 0.5,
        softening: Union[float, ArrayLike] = 0.0,
        G: float = 1.0,
    ) -> ArrayLike:
        """Compute accelerations for Dask target arrays."""
        # Import Dask array operations
        try:
            import dask.array as da
        except ImportError as e:
            raise RuntimeError("Dask not available") from e
        
        # Get the Dask namespace
        from .backends.dask_ import get_dask_namespace
        xp = get_dask_namespace()
        
        # Define chunk function for map_blocks
        def _compute_targets_chunk_accelerations(targets_chunk, *, tree_data, source_pos, source_masses, 
                                               theta_val, softening_val, G_val):
            """Compute accelerations for a chunk of target positions."""
            from .traverse.bh import barnes_hut_accelerations_targets
            
            # Compute accelerations for this chunk
            return barnes_hut_accelerations_targets(
                tree_data, source_pos, source_masses, targets_chunk, 
                softening_val, theta_val, G_val
            )
        
        # Use map_blocks to compute accelerations preserving chunking
        return xp.map_blocks_accelerations(
            _compute_targets_chunk_accelerations,
            targets,  # targets (Dask array)
            self._tree,       # tree data
            self.positions,  # source positions (NumPy)
            self.masses,     # source masses (NumPy)
            theta_val=theta,
            softening_val=softening,
            G_val=G
        )

    def evaluate_accelerations(
        self, 
        softening: Union[float, NDArray[np.floating[Any]]], 
        theta: float = 0.5,
        G: float = 1.0
    ) -> NDArray[np.floating[Any]]:
        """
        Evaluate gravitational accelerations for all particles in the tree.
        
        Parameters
        ----------
        softening : float or array_like
            Gravitational softening length(s). If scalar, same softening is used
            for all particles. If array, must have shape (N,) for per-particle
            softening.
        theta : float, default 0.5
            Opening angle criterion for Barnes-Hut approximation. Smaller values
            give higher accuracy but slower computation. Typical range: 0.1-1.0.
        G : float, default 1.0
            Gravitational constant
            
        Returns
        -------
        accelerations : array_like, shape (N, D)
            Gravitational accelerations for each particle
            
        Raises
        ------
        RuntimeError
            If tree has not been built (call `build()` first)
        ValueError
            If input parameters have invalid shapes or values
        """
        if self._tree is None:
            raise RuntimeError("Tree not built. Call build() first.")
        
        # Validate inputs
        if not isinstance(softening, (int, float)):
            raise ValueError("Only scalar softening is currently supported")
        
        if softening < 0:
            raise ValueError("Softening must be non-negative")
        
        if theta < 0:
            raise ValueError("Opening angle theta must be non-negative")
        
        # Use the standalone function to compute accelerations
        accelerations_np = evaluate_accelerations(
            self._tree, self.positions, self.masses, theta, softening, G
        )
        
        # Convert back to original array type  
        xp = get_namespace(self.positions, self.backend)
        return xp.asarray(accelerations_np)
