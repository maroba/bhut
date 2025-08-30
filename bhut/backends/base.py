"""
Base array namespace protocol and utilities for array-agnostic operations.

This module defines the ArrayNamespace protocol that abstracts over different
array libraries (NumPy, Dask, etc.) to enable array-agnostic code.
"""

from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class ArrayNamespace(Protocol):
    """
    Protocol defining array operations needed for Barnes-Hut algorithm.

    This abstraction allows the same code to work with NumPy, Dask, and other
    array libraries by providing a consistent interface.
    """

    def asarray(self, obj: ArrayLike, dtype: Any = None) -> NDArray[Any]:
        """Convert input to array."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> NDArray[Any]:
        """Create array filled with zeros."""
        ...

    def empty(self, shape: tuple[int, ...], dtype: Any = None) -> NDArray[Any]:
        """Create uninitialized array."""
        ...

    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None) -> NDArray[Any]:
        """Create array filled with constant value."""
        ...

    def arange(self, start: int, stop: int = None, step: int = 1, dtype: Any = None) -> NDArray[Any]:
        """Create array with evenly spaced values."""
        ...

    def argsort(self, a: NDArray[Any], axis: int = -1, kind: str = "stable") -> NDArray[Any]:
        """Return indices that would sort array."""
        ...

    def sort(self, a: NDArray[Any], axis: int = -1, kind: str = "stable") -> NDArray[Any]:
        """Return sorted copy of array."""
        ...

    def take(self, a: NDArray[Any], indices: NDArray[Any], axis: int = None) -> NDArray[Any]:
        """Take elements along axis."""
        ...

    def sum(self, a: NDArray[Any], axis: int = None, keepdims: bool = False) -> NDArray[Any]:
        """Sum array elements."""
        ...

    def sqrt(self, x: NDArray[Any]) -> NDArray[Any]:
        """Element-wise square root."""
        ...

    def maximum(self, x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        """Element-wise maximum."""
        ...

    def minimum(self, x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        """Element-wise minimum."""
        ...

    def where(
        self, condition: NDArray[Any], x: NDArray[Any], y: NDArray[Any]
    ) -> NDArray[Any]:
        """Return elements chosen from x or y depending on condition."""
        ...

    def stack(self, arrays: list[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        """Stack arrays along new axis."""
        ...

    def concatenate(self, arrays: list[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        """Concatenate arrays along existing axis."""
        ...

    def expand_dims(self, a: NDArray[Any], axis: int) -> NDArray[Any]:
        """Expand array dimensions."""
        ...

    def squeeze(self, a: NDArray[Any], axis: int = None) -> NDArray[Any]:
        """Remove single-dimensional entries."""
        ...

    def reshape(self, a: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
        """Reshape array."""
        ...

    def broadcast_to(self, array: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
        """Broadcast array to shape."""
        ...


def get_namespace(array: ArrayLike, backend: str = "auto") -> ArrayNamespace:
    """
    Get appropriate array namespace for given array and backend preference.

    Parameters
    ----------
    array : array_like
        Input array to determine backend from
    backend : str, optional
        Backend preference ("numpy", "dask", or "auto"). Default: "auto"

    Returns
    -------
    namespace : ArrayNamespace
        Array namespace object with required operations

    Raises
    ------
    ValueError
        If backend is not supported or array type is incompatible
    """
    import numpy as np
    
    # Convert to array to inspect type
    arr = np.asarray(array)
    
    if backend == "auto":
        # Auto-detect based on array type
        if hasattr(array, "__array_namespace__"):
            # Array API standard namespace
            array_module_name = array.__array_namespace__().__name__
            if "numpy" in array_module_name:
                backend = "numpy"
            elif "dask" in array_module_name:
                backend = "dask"
            else:
                # Default to numpy for unknown array API namespaces
                backend = "numpy"
        elif hasattr(array, "__module__"):
            # Check module name for common array libraries
            module_name = getattr(array, "__module__", "")
            if "dask" in module_name:
                backend = "dask"
            else:
                backend = "numpy"
        else:
            # Check for Dask array type
            try:
                from bhut.backends.dask_ import detect_dask_array
                if detect_dask_array(array):
                    backend = "dask"
                else:
                    backend = "numpy"
            except ImportError:
                # Dask not available, use numpy
                backend = "numpy"
    
    if backend == "numpy":
        from bhut.backends.numpy_ import numpy_namespace
        return numpy_namespace
    elif backend == "dask":
        from bhut.backends.dask_ import get_dask_namespace
        return get_dask_namespace()
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported: 'numpy', 'dask', 'auto'")
    
    # This line should never be reached, but satisfies mypy
    raise ValueError(f"Unable to determine backend for array type: {type(array)}")


def validate_shape(array: ArrayLike, expected_shape: tuple[int, ...], name: str) -> None:
    """
    Validate that array has expected shape.

    Parameters
    ----------
    array : array_like
        Array to validate
    expected_shape : tuple of int
        Expected shape, with -1 meaning any size for that dimension
    name : str
        Name of the array for error messages

    Raises
    ------
    ValueError
        If shape is incompatible
    """
    arr = np.asarray(array)
    if len(arr.shape) != len(expected_shape):
        raise ValueError(
            f"{name} must have {len(expected_shape)} dimensions, "
            f"got {len(arr.shape)}"
        )

    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape, strict=False)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} dimension {i} must have size {expected}, got {actual}"
            )


def validate_compatible_shapes(
    array1: ArrayLike, array2: ArrayLike, name1: str, name2: str
) -> None:
    """
    Validate that two arrays have compatible shapes for N-body calculations.

    Parameters
    ----------
    array1, array2 : array_like
        Arrays to validate
    name1, name2 : str
        Names for error messages

    Raises
    ------
    ValueError
        If shapes are incompatible
    """
    arr1 = np.asarray(array1)
    arr2 = np.asarray(array2)

    if len(arr1.shape) == 2 and len(arr2.shape) == 1:
        # positions (N, dim) and masses (N,)
        if arr1.shape[0] != arr2.shape[0]:
            raise ValueError(
                f"{name1} has {arr1.shape[0]} particles but "
                f"{name2} has {arr2.shape[0]} elements"
            )
    elif len(arr1.shape) == 2 and len(arr2.shape) == 2:
        # two position arrays
        if arr1.shape[1] != arr2.shape[1]:
            raise ValueError(
                f"{name1} has {arr1.shape[1]} dimensions but "
                f"{name2} has {arr2.shape[1]} dimensions"
            )
    else:
        raise ValueError(f"Incompatible array shapes: {name1}={arr1.shape}, {name2}={arr2.shape}")
