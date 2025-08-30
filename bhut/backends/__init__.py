"""Backend system for array-agnostic operations."""

from bhut.backends.base import ArrayNamespace, get_namespace
from bhut.backends.numpy_ import numpy_namespace

__all__ = ["ArrayNamespace", "get_namespace", "numpy_namespace"]
