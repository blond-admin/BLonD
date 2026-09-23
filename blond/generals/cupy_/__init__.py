# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Scripts that are useful to work with Cupy."""

from __future__ import annotations

from typing import Any

CUPY_MISSING_MESSAGE = (
    "The CUDA backend needs CuPy, which is not installed. Install the "
    "GPU extra matching your CUDA toolkit, e.g. "
    "`pip install blond[gpu_cuda12]` for CUDA 12 or "
    "`pip install blond[gpu_cuda13]` for CUDA 13. To stay on the CPU "
    "instead, pick a CPU backend, e.g. BLOND_BACKEND_MODE=numba "
    "(or 'cpp' / 'python')."
)


def import_cupy_with_error_hint() -> Any:
    """
    Import CuPy, or fail with an actionable error message.

    Returns
    -------
    cupy
        The imported ``cupy`` module.

    Raises
    ------
    ModuleNotFoundError
        If CuPy is not installed. The message names the extra to
        install and the CPU alternative; the original import error is
        kept as ``__cause__``.

    Notes
    -----
    Use this instead of a bare ``import cupy`` on any code path a
    CPU-only machine can reach. Most machines running BLonD have no
    GPU at all, so the bare ``ModuleNotFoundError: No module named
    'cupy'`` -- raised from deep inside the backend machinery -- is a
    common and needlessly confusing first contact with the library.
    """
    try:
        import cupy  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(CUPY_MISSING_MESSAGE) from exc
    return cupy


def __getattr__(name: str) -> Any:
    """
    Provide ``cupy`` as a module attribute when it is installed.

    Parameters
    ----------
    name
        The attribute being looked up.

    Returns
    -------
    cupy
        The imported ``cupy`` module.

    Raises
    ------
    AttributeError
        If ``name`` is not ``"cupy"`` or CuPy is not installed, so
        ``hasattr`` returns ``False``. ``from blond.generals.cupy_ import
        cupy`` then falls back to the ``cupy`` submodule, which raises
        the ``ModuleNotFoundError`` with the install hint.
    """
    if name != "cupy":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return import_cupy_with_error_hint()
    except ModuleNotFoundError as exc:
        raise AttributeError(str(exc)) from exc
