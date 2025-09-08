"""
This file should not have any dependency to Cupy
to allow cupy agnostic code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


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
