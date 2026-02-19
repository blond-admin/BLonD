# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of helpers to develop new functions and modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from numpy import ndarray as numpyarray
from numpy.typing import NDArray as NumpyArray


class UnevenArraySizes(Exception):
    """Exception of uneven array sizes in function arguments."""

    pass


def raise_on_uneven_array_sizes(
    *args: tuple[float | Sequence | NumpyArray],
) -> Any:
    """
    Check if the tuple of arguments have the same length.

    Parameters
    ----------
    *args
        Tuple of Sequence.

    Returns
    -------
    UnevenArraySizes exception
        If any input arrays or sequences have different lengths.

    Examples
    --------
    >>> def function(*args):
    >>>     args = tuple(locals().values())
    >>>     raise_on_uneven_array_sizes(args)
    """
    lengths = []
    for a in args[0]:
        if isinstance(a, Sequence | numpyarray):
            lengths.append(len(a))
    if len(set(lengths)) > 1:
        raise UnevenArraySizes(
            "Input sequences of more than one element have different lengths."
        )
