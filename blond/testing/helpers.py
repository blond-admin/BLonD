# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Various helper scripts to support testing of BLonD."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


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
    print(
        f"\n{variable_name}_pinned = {variable.tolist()}\n"
        f"""np.testing.assert_allclose(
    {variable_name},
    {variable_name}_pinned,
    rtol=1e-6 if backend.float == np.float32 else 1e-12,
)"""
    )
