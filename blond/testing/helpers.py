# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Various helper scripts to support testing of BLonD."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from blond import Cupy64Bit, Numpy64Bit, backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray

    NumpyArray = NDArray[Any]


def pinned_values_helper(variable: NumpyArray, variable_name: str) -> None:
    """
    Use this to generate a code snippet to copy-paste into the tests.

    Workflow:
    1. Write a testcase.
    2. Execute the test with `pinned_values_helper` to get a printed output.
    3. Replace `pinned_values_helper` by the printed output.

    Parameters
    ----------
    variable
        The array to be pinned.
    variable_name
        The name of the variable.

    Examples
    --------
    >>> import numpy as np
    >>> array = np.ones(10)
    >>> pinned_values_helper(array, "array")
    """
    variable_name_nodot = variable_name.replace(".", "_")
    print(
        f"\n{variable_name_nodot}_pinned = {variable.tolist()}\n"
        f"""np.testing.assert_allclose(
    {variable_name},
    {variable_name_nodot}_pinned,
    rtol=1e-6 if backend.float == np.float32 else 1e-12,
)"""
    )


def pytest_active():
    """
    Return ``True``, if pytest is active.

    Returns
    -------
    pytest_is_active
        `True``, if pytest is active.
    """
    testing = os.environ.get("PYTEST_CURRENT_TEST", None) is not None
    if testing is None:
        return False
    else:
        return bool(testing)


def allclose_tolerances(
    expected: NumpyArray,
    rtol_32bit: float = 1e-6,
) -> dict[str, float]:
    """
    Generate keyword-arguments for the tolerances of `np.testing.assert_allclose`.

    Parameters
    ----------
    expected
        Expected array of `np.testing.assert_allclose`.
    rtol_32bit
        Relative tolerance for 32 bit backend.
        The 64-bit tolerance is double, e.g. `1e-6` and `1e-12`.

    Returns
    -------
    kwargs
        The `rtol` and `atol` keyword arguments.

    Examples
    --------
    >>> np.testing.assert_allclose(
    ...     actual,
    ...     expected,
    ...     **allclose_tolerances(expected),
    ... )
    """
    amplitude = float(np.max(expected) - np.min(expected))
    rtol = rtol_32bit if backend.float == np.float32 else (rtol_32bit**2)
    kwargs = {
        "rtol": 0,  # intentional 0, it makes problems at arrays that cross 0.
        "atol": amplitude * rtol,
    }
    return kwargs


def enforce_64_bit_backend():
    """Enforce 64-bit backend, GPU is taken into account."""
    if backend.float == np.float32:
        if backend.is_gpu:
            backend.change_backend(Cupy64Bit)
        else:
            backend.change_backend(Numpy64Bit)
