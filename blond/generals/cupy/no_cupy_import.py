# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module without dependency to Cupy."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def is_cupy_array(arr: NumpyArray | CupyArray | Any) -> bool:
    """
    Check if the array is a Cupy array.

    Parameters
    ----------
    arr
        A Numpy or Cupy array.

    Returns
    -------
    is_cupy
        True if it's a GPU array, False otherwise.
    """
    if hasattr(arr, "device"):
        return arr.device != "cpu"  # type: ignore
    elif hasattr(arr, "gpu_data"):  # numba.cuda array
        # Overall there is no problem with numba-cuda arrays.
        # Its just that the entire code is tested against Cupy
        # So use of it is discouraged.
        raise TypeError(f"{type(arr)} not supported.")
    else:
        return False


def copy_to_cpu(array: NumpyArray | CupyArray) -> NumpyArray:
    """
    Copy array from GPU/CPU to CPU.

    Parameters
    ----------
    array
        Array to copy (can be on GPU or CPU).

    Returns
    -------
    array
        CPU copy of the original array.

    Notes
    -----
    Changes to the returned array will not
    be reflected on the original memory.
    """
    if is_cupy_array(array):
        return array.get()
    else:
        return array.copy()


class _AsarrayOverrideManager:
    def __init__(self) -> None:
        """Override functionality for 'np.asarray' to handle Cupy."""
        self.cache: dict[int, np.ndarray] = {}
        self._numpy_asarray_original = deepcopy(np.asarray)
        self._numpy_array_original = deepcopy(np.array)

    def asarray_override(
        self,
        a: NumpyArray | CupyArray,
        dtype: Any = None,
        order: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> NumpyArray:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            key = a.data.ptr
            if key not in self.cache:
                a = a.get()  # copy data from GPU
                self.cache[key] = a
            else:
                # DON'T copy data from GPU, because it was done already
                a = self.cache[key]

        return self._numpy_asarray_original(  # type: ignore
            a,
            dtype=dtype,
            order=order,
            *args,  # NOQA: B026
            **kwargs,
        )

    def array_override(
        self, p_object, dtype=None, *args, **kwargs
    ) -> NumpyArray:
        import cupy as cp  # type: ignore

        a = p_object
        if isinstance(a, cp.ndarray):
            key = a.data.ptr
            if key not in self.cache:
                a = a.get()  # copy data from GPU
                self.cache[key] = a
            else:
                # DON'T copy data from GPU, because it was done already
                a = self.cache[key]

        return self._numpy_array_original(  # type: ignore
            a,
            dtype=dtype,
            *args,  # NOQA: B026
            **kwargs,
        )


class AllowPlotting:
    """
    Allows implicitly casting of Cupy arrays to Numpy arrays.

    Notes
    -----
    This is only intended for plotting of arrays.
    The function temporarily overrides the numpy.asarray function.

    Examples
    --------
    >>> y = cupy.ones(12)
    >>> with AllowPlotting():
    ...     plt.plot(y)
    """

    def __init__(self) -> None:
        try:
            self.cupy_found = True

        except Exception:
            self.cupy_found = False
            return  # do nothing
        # initialize cache, make override function available
        self.asarray_override_manager = _AsarrayOverrideManager()

    def __enter__(self) -> None:
        """Override np.asarray with own function to handle .get() for Cupy arrays."""
        if not self.cupy_found:
            return
        # override numpy "asarray" function with own function
        self.asarray_org = deepcopy(np.asarray)
        np.asarray = self.asarray_override_manager.asarray_override

        self.array_org = deepcopy(np.array)
        np.array = self.asarray_override_manager.array_override

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ):
        """
        Reset np.asarray to original Numpy function.

        Parameters
        ----------
        exc_type
            Exception type if an exception occurred.
        exc_val
            Exception value if an exception occurred.
        exc_tb
            Exception traceback if an exception occurred.
        """
        if not backend.is_gpu:
            return  # do nothing
        # reset to original numpy function
        np.asarray = self.asarray_org
        np.array = self.array_org
