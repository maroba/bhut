"""
NumPy backend implementation for array-agnostic operations.

This module provides a concrete ArrayNamespace implementation using NumPy,
allowing the Barnes-Hut algorithm to work with NumPy arrays.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bhut.backends.base import ArrayNamespace


class NumpyArrayNamespace:
    """
    Concrete ArrayNamespace implementation using NumPy.
    
    This class wraps NumPy functions to provide the array operations
    needed by the Barnes-Hut algorithm in a consistent interface.
    """

    def asarray(self, obj: ArrayLike, dtype: Any = None) -> NDArray[Any]:
        """Convert input to NumPy array."""
        return np.asarray(obj, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> NDArray[Any]:
        """Create NumPy array filled with zeros."""
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...], dtype: Any = None) -> NDArray[Any]:
        """Create uninitialized NumPy array."""
        return np.empty(shape, dtype=dtype)

    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None) -> NDArray[Any]:
        """Create NumPy array filled with constant value."""
        return np.full(shape, fill_value, dtype=dtype)

    def arange(self, start: int, stop: int = None, step: int = 1, dtype: Any = None) -> NDArray[Any]:
        """Create NumPy array with evenly spaced values."""
        if stop is None:
            stop = start
            start = 0
        return np.arange(start, stop, step, dtype=dtype)

    def argsort(self, a: NDArray[Any], axis: int = -1, kind: str = "stable") -> NDArray[Any]:
        """Return indices that would sort NumPy array."""
        return np.argsort(a, axis=axis, kind=kind)

    def sort(self, a: NDArray[Any], axis: int = -1, kind: str = "stable") -> NDArray[Any]:
        """Return sorted copy of NumPy array."""
        return np.sort(a, axis=axis, kind=kind)

    def take(self, a: NDArray[Any], indices: NDArray[Any], axis: int = None) -> NDArray[Any]:
        """Take elements from NumPy array along axis."""
        return np.take(a, indices, axis=axis)

    def sum(self, a: NDArray[Any], axis: int = None, keepdims: bool = False) -> NDArray[Any]:
        """Sum NumPy array elements."""
        return np.sum(a, axis=axis, keepdims=keepdims)

    def sqrt(self, x: NDArray[Any]) -> NDArray[Any]:
        """Element-wise square root."""
        return np.sqrt(x)

    def maximum(self, x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        """Element-wise maximum."""
        return np.maximum(x1, x2)

    def minimum(self, x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        """Element-wise minimum."""
        return np.minimum(x1, x2)

    def where(
        self, condition: NDArray[Any], x: NDArray[Any], y: NDArray[Any]
    ) -> NDArray[Any]:
        """Return elements chosen from x or y depending on condition."""
        return np.where(condition, x, y)

    def stack(self, arrays: list[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        """Stack arrays along new axis."""
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        """Concatenate arrays along existing axis."""
        return np.concatenate(arrays, axis=axis)

    def expand_dims(self, a: NDArray[Any], axis: int) -> NDArray[Any]:
        """Expand array dimensions."""
        return np.expand_dims(a, axis)

    def squeeze(self, a: NDArray[Any], axis: int = None) -> NDArray[Any]:
        """Remove single-dimensional entries."""
        return np.squeeze(a, axis)

    def reshape(self, a: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
        """Reshape array."""
        return np.reshape(a, shape)

    def broadcast_to(self, array: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
        """Broadcast array to shape."""
        return np.broadcast_to(array, shape)


# Create singleton instance for use throughout the library
numpy_namespace = NumpyArrayNamespace()
