# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module without dependency to Cupy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def is_cupy_array(arr: NumpyArray | CupyArray | Any) -> bool:
    """Checks if the array is a Cupy array.

    Parameters
    ----------
    arr
        A Numpy or Cupy array

    Returns
    -------
    True, if it's an GPU array

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


# pin original numpy function for `AllowPlotting`
_numpy_asarray_original = np.asarray


class _AsarrayOverrideManager:
    def __init__(self) -> None:
        """Override functionality for 'np.asarray' to handle Cupy."""
        self.cache: dict[int, np.ndarray] = {}

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

        return _numpy_asarray_original(  # type: ignore
            a,
            dtype=dtype,
            order=order,
            *args,  # NOQA: B026
            **kwargs,
        )


class AllowPlotting:
    """Allows implicitly casting of Cupy arrays to Numpy arrays.

    Notes
    -----
    This is only intended for plotting of arrays.
    The function temporarily overrides the numpy.asarray function.

    Examples
    --------
    >>> y = cupy.ones(12)
    >>> with AllowPlotting():
    >>>     plt.plot(y)


    """

    def __init__(self) -> None:
        if not backend.is_gpu:
            return  # do nothing
        # initialize cache, make override function available
        self.asarray_override_manager = _AsarrayOverrideManager()

    def __enter__(self) -> None:
        """Override np.asarray with own function to handle .get() for Cupy arrays."""
        if not backend.is_gpu:
            return
        # override numpy "asarray" function with own function
        self.asarray_org = np.asarray
        np.asarray = self.asarray_override_manager.asarray_override

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ):
        """Reset np.asarray to original Numpy function."""
        if not backend.is_gpu:
            return  # do nothing
        # reset to original numpy function
        np.asarray = _numpy_asarray_original
