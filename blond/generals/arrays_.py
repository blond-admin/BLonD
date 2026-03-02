# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with Numpy/Cupy arrays."""

from typing import TYPE_CHECKING

from blond import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def is_linspace_like(
    arr: NumpyArray | CupyArray,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> bool:
    """
    Test whether the given 1D array is a `linspace` or not.

    Parameters
    ----------
    arr
        The array to be investigated.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.

    Returns
    -------
    is_linspace
        True, if the given array is an `linspace`.
    """
    # Must be 1D and have at least 2 elements
    if arr.ndim != 1 or arr.size < 2:  # noqa: PLR2004
        return False

    # Compute differences
    diffs = backend.diff(arr)

    # Check if all differences are equal (within floating tolerance)
    return bool(backend.allclose(diffs, diffs[0], rtol=rtol, atol=atol))
