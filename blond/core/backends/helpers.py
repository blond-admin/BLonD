# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to improve usability of the backend."""

from __future__ import annotations

from typing import Literal


def setup_backend(
    mode: Literal["auto", "python", "cpp", "numba", "cuda"],
) -> None:
    """
    Set up the backend to be used.

    Parameters
    ----------
    mode
        Backend special mode to use.
        If "auto", the fasted backend will be automatically choose.

    Notes
    -----
    This should be called at the start of the user input script,
    as it will define the state of all internal arrays.
    """
    from blond import backend

    if mode == "auto":
        backend.autoselect_backend()
    elif mode == "cuda":
        from blond import Cupy64Bit

        backend.change_backend(Cupy64Bit)
        backend.set_specials(mode)
    elif mode in ("python", "cpp", "numba"):
        from blond import Numpy64Bit

        backend.change_backend(Numpy64Bit)
        backend.set_specials(mode)
    else:
        raise ValueError(mode)
