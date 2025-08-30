"""
bhut: Barnes-Hut N-body accelerator that is array-agnostic.

This package provides efficient N-body force calculations using the Barnes-Hut
algorithm, with support for both NumPy and Dask arrays.
"""

from bhut.api import Tree, accelerations

__version__ = "0.1.0"
__all__ = ["Tree", "accelerations"]
