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

from blond import Cupy64Bit, Numpy64Bit, backend

_options = {  # Intentionally disregard 32 bit backends, as they might be removed in future
    "python": (Numpy64Bit, "python"),
    "cpp": (Numpy64Bit, "cpp"),
    "numba": (Numpy64Bit, "numba"),
    "cuda": (Cupy64Bit, "cuda"),
}


def setup_backend(
    mode: Literal[
        "auto",
        "python",
        "cpp",
        "numba",
        "cuda",
    ],
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
    if mode == "auto":
        backend.autoselect_backend()
    else:
        backend_, mode_ = _options[mode]
        backend.change_backend(backend_)
        backend.set_specials(mode_)
