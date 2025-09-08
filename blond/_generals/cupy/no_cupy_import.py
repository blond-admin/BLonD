"""
This file should not have any dependency to Cupy
to allow cupy agnostic code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

numpy_asarray = np.asarray


def is_cupy_array(arr: NumpyArray | CupyArray | None) -> bool:
    """
    Checks if the array is a Cupy array

    Parameters
    ----------
    arr
        A Numpy or Cupy array

    Returns
    -------
    True, if it's an GPU array

    """
    if hasattr(arr, "device"):
        return not (arr.device == "cpu")  # type: ignore
    else:
        return False


class _AsarrayOverrideManager:
    def __init__(self) -> None:
        """Override functionality for 'np.asarray' with caching"""
        self.cache: dict[int, np.ndarray] = {}

    def asarray_override(self, a, dtype=None, order=None, *args, **kwargs):
        import cupy as cp

        if isinstance(a, cp.ndarray):
            key = a.data.ptr
            if key in self.cache.keys():
                a = self.cache[
                    key
                ]  # DON'T copy data from GPU, because it was done already
            else:
                a = a.get()  # copy data from GPU
                self.cache[key] = a
        return numpy_asarray(a, dtype=dtype, order=order, *args, **kwargs)


class AllowPlotting:
    def __init__(self) -> None:
        """Allows implicitly casting of cupy arrays to numpy arrays .

        Notes
        -----
        This is only intended for plotting of arrays.

        """
        if not backend.is_gpu:
            return
        # initialize cache, make override function available
        self.asarray_override_manager = _AsarrayOverrideManager()

    def __enter__(self) -> None:
        if not backend.is_gpu:
            return
        # override numpy "asarray" function with own function
        self.asarray_org = np.asarray
        np.asarray = self.asarray_override_manager.asarray_override

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        if not backend.is_gpu:
            return
        # reset to original numpy function
        np.asarray = numpy_asarray
