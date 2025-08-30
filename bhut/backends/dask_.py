"""
Dask backend for distributed array operations.

This module provides support for Dask arrays in the bhut package,
enabling distributed computation of gravitational accelerations.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import dask.array as da
    from dask.array import Array as DaskArray
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    da = None
    DaskArray = None

from .base import ArrayNamespace


class DaskArrayNamespace:
    """
    Concrete ArrayNamespace implementation using Dask.
    
    This class wraps Dask functions to provide the array operations
    needed by the Barnes-Hut algorithm. It uses smart materialization
    strategies to balance performance and memory efficiency.
    
    Key design decisions:
    1. Tree building: Materialize positions/masses to NumPy for speed and correctness
    2. Accelerations: Keep targets as Dask arrays, use map_blocks for evaluation
    3. Preserve chunking: Return results with same chunking as inputs
    """

    def __init__(self):
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not available. Install with: pip install dask[array]"
            )

    def asarray(self, obj: ArrayLike, dtype: Any = None) -> Union[NDArray[Any], DaskArray]:
        """Convert input to Dask array."""
        if isinstance(obj, DaskArray):
            if dtype is not None and obj.dtype != dtype:
                return obj.astype(dtype)
            return obj
        return da.asarray(obj, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None, chunks: Any = None) -> DaskArray:
        """Create Dask array filled with zeros."""
        if chunks is None:
            chunks = "auto"
        return da.zeros(shape, dtype=dtype, chunks=chunks)

    def empty(self, shape: tuple[int, ...], dtype: Any = None, chunks: Any = None) -> DaskArray:
        """Create uninitialized Dask array."""
        if chunks is None:
            chunks = "auto"
        return da.empty(shape, dtype=dtype, chunks=chunks)

    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None, chunks: Any = None) -> DaskArray:
        """Create Dask array filled with constant value."""
        if chunks is None:
            chunks = "auto"
        return da.full(shape, fill_value, dtype=dtype, chunks=chunks)

    def arange(self, start: int, stop: int = None, step: int = 1, dtype: Any = None, chunks: Any = None) -> DaskArray:
        """Create Dask array with evenly spaced values."""
        if stop is None:
            stop = start
            start = 0
        if chunks is None:
            chunks = "auto"
        return da.arange(start, stop, step, dtype=dtype, chunks=chunks)

    def argsort(self, a: Union[NDArray[Any], DaskArray], axis: int = -1, kind: str = "stable") -> DaskArray:
        """Return indices that would sort Dask array."""
        if isinstance(a, DaskArray):
            return da.argsort(a, axis=axis)  # Dask doesn't support kind parameter
        return da.from_array(np.argsort(a, axis=axis, kind=kind))

    def sort(self, a: Union[NDArray[Any], DaskArray], axis: int = -1, kind: str = "stable") -> DaskArray:
        """Return sorted copy of Dask array."""
        if isinstance(a, DaskArray):
            return da.sort(a, axis=axis)  # Dask doesn't support kind parameter
        return da.from_array(np.sort(a, axis=axis, kind=kind))

    def take(self, a: Union[NDArray[Any], DaskArray], indices: Union[NDArray[Any], DaskArray], axis: int = None) -> DaskArray:
        """Take elements along axis."""
        if isinstance(a, DaskArray):
            return da.take(a, indices, axis=axis)
        return da.from_array(np.take(a, indices, axis=axis))

    def sum(self, a: Union[NDArray[Any], DaskArray], axis: int = None, keepdims: bool = False) -> DaskArray:
        """Sum array elements."""
        if isinstance(a, DaskArray):
            return da.sum(a, axis=axis, keepdims=keepdims)
        return da.from_array(np.sum(a, axis=axis, keepdims=keepdims))

    def sqrt(self, x: Union[NDArray[Any], DaskArray]) -> DaskArray:
        """Element-wise square root."""
        if isinstance(x, DaskArray):
            return da.sqrt(x)
        return da.from_array(np.sqrt(x))

    def maximum(self, x1: Union[NDArray[Any], DaskArray], x2: Union[NDArray[Any], DaskArray]) -> DaskArray:
        """Element-wise maximum."""
        return da.maximum(x1, x2)

    def minimum(self, x1: Union[NDArray[Any], DaskArray], x2: Union[NDArray[Any], DaskArray]) -> DaskArray:
        """Element-wise minimum."""
        return da.minimum(x1, x2)

    def where(
        self, condition: Union[NDArray[Any], DaskArray], x: Union[NDArray[Any], DaskArray], y: Union[NDArray[Any], DaskArray]
    ) -> DaskArray:
        """Return elements chosen from x or y depending on condition."""
        return da.where(condition, x, y)

    def stack(self, arrays: list[Union[NDArray[Any], DaskArray]], axis: int = 0) -> DaskArray:
        """Stack arrays along new axis."""
        return da.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Union[NDArray[Any], DaskArray]], axis: int = 0) -> DaskArray:
        """Concatenate arrays along existing axis."""
        return da.concatenate(arrays, axis=axis)

    def expand_dims(self, a: Union[NDArray[Any], DaskArray], axis: int) -> DaskArray:
        """Expand array dimensions."""
        if isinstance(a, DaskArray):
            return da.expand_dims(a, axis)
        return da.from_array(np.expand_dims(a, axis))

    def squeeze(self, a: Union[NDArray[Any], DaskArray], axis: int = None) -> DaskArray:
        """Remove single-dimensional entries."""
        if isinstance(a, DaskArray):
            return da.squeeze(a, axis=axis)
        return da.from_array(np.squeeze(a, axis=axis))

    def reshape(self, a: Union[NDArray[Any], DaskArray], shape: tuple[int, ...]) -> DaskArray:
        """Reshape array."""
        if isinstance(a, DaskArray):
            return da.reshape(a, shape)
        return da.from_array(np.reshape(a, shape))

    def broadcast_to(self, array: Union[NDArray[Any], DaskArray], shape: tuple[int, ...]) -> DaskArray:
        """Broadcast array to shape."""
        if isinstance(array, DaskArray):
            return da.broadcast_to(array, shape)
        return da.from_array(np.broadcast_to(array, shape))

    # Dask-specific methods for tree operations
    
    def materialize(self, array: Union[NDArray[Any], DaskArray]) -> NDArray[Any]:
        """
        Materialize Dask array to NumPy for tree building operations.
        
        This is used when we need the full array in memory for operations
        like tree construction that don't parallelize well.
        """
        if isinstance(array, DaskArray):
            return array.compute()
        return np.asarray(array)
    
    def is_dask_array(self, array: Any) -> bool:
        """Check if array is a Dask array."""
        return isinstance(array, DaskArray)
    
    def map_blocks_accelerations(
        self,
        func: callable,
        targets: DaskArray,
        tree_data: Any,
        positions: NDArray,
        masses: NDArray,
        *args,
        **kwargs
    ) -> DaskArray:
        """
        Apply acceleration computation to target chunks using map_blocks.
        
        This preserves the chunking structure of the target array while
        evaluating accelerations against the materialized tree data.
        
        Parameters
        ----------
        func : callable
            Function to compute accelerations for a chunk of targets
        targets : DaskArray
            Target positions as Dask array
        tree_data : TreeData
            Materialized tree structure
        positions : ndarray
            Source positions (materialized)
        masses : ndarray
            Source masses (materialized)
        *args, **kwargs
            Additional arguments for the acceleration function
        
        Returns
        -------
        DaskArray
            Accelerations with same chunking as targets
        """
        # Create keyword arguments for the function
        func_kwargs = {
            'tree_data': tree_data,
            'source_pos': positions,
            'source_masses': masses,
            **kwargs
        }
        
        # Pack additional positional args into kwargs with numbered keys  
        for i, arg in enumerate(args):
            func_kwargs[f'arg_{i}'] = arg
        
        return da.map_blocks(
            func,
            targets,
            dtype=targets.dtype,
            chunks=targets.chunks,
            drop_axis=[],  # Keep same dimensionality (N, 3) -> (N, 3)
            **func_kwargs
        )


def _ensure_dask_available():
    """Ensure Dask is available, raise helpful error if not."""
    if not DASK_AVAILABLE:
        raise ImportError(
            "Dask backend requires dask[array]. Install with:\n"
            "  pip install dask[array]\n"
            "or\n"
            "  conda install dask"
        )


# Create singleton instance for use throughout the library
# Only create if Dask is available
if DASK_AVAILABLE:
    dask_namespace = DaskArrayNamespace()
else:
    dask_namespace = None


def get_dask_namespace() -> DaskArrayNamespace:
    """Get the Dask namespace, ensuring Dask is available."""
    _ensure_dask_available()
    if dask_namespace is None:
        return DaskArrayNamespace()
    return dask_namespace


def detect_dask_array(array: Any) -> bool:
    """
    Detect if an array is a Dask array without importing Dask.
    
    This is useful for backend detection in get_namespace.
    """
    if not DASK_AVAILABLE:
        return False
    return isinstance(array, DaskArray)


def materialize_for_tree_building(
    positions: Union[NDArray, DaskArray],
    masses: Union[NDArray, DaskArray]
) -> Tuple[NDArray, NDArray]:
    """
    Materialize Dask arrays to NumPy arrays for tree building.
    
    Tree construction requires random access patterns that are inefficient
    with Dask's chunked structure. This function materializes the arrays
    for efficient tree building.
    
    Parameters
    ----------
    positions : ndarray or DaskArray
        Position array that may need materialization
    masses : ndarray or DaskArray
        Mass array that may need materialization
        
    Returns
    -------
    positions_np : ndarray
        Materialized positions as NumPy array
    masses_np : ndarray
        Materialized masses as NumPy array
    """
    # Materialize positions if it's a Dask array
    if hasattr(positions, 'compute'):
        positions_np = positions.compute()
    else:
        positions_np = positions
        
    # Materialize masses if it's a Dask array
    if hasattr(masses, 'compute'):
        masses_np = masses.compute()
    else:
        masses_np = masses
        
    return positions_np, masses_np
