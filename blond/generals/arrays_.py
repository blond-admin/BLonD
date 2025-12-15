# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with Numpy/Cupy arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy._typing import NDArray as NumpyArray


def _read_only(array: NumpyArray | CupyArray):
    """
    Create a read-only view of the given array.

    Parameters
    ----------
    array
        Numpy or Cupy array.

    Returns
    -------
    a_readonly
        Readonly view of the original array.
    """
    view = array.view()
    view.flags.writeable = False
    return view
