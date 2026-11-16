# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Various helper scripts to support testing of BLonD."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend

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
    rtol=1e-12,
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
    # `PYTEST_VERSION` is exported by pytest (>=8) for the whole session,
    # collection included, and is the only reliable marker of a session.
    # `PYTEST_CURRENT_TEST` exists only while a test is *executing*: module
    # level code guarded by `if not pytest_active()` runs at import time,
    # i.e. during collection, when that variable is still absent, so the
    # guard would not hold and the module would mutate global state for the
    # rest of the session. `"pytest" in sys.modules` has the opposite flaw,
    # being true for any script that merely imports pytest (e.g. a test file
    # run directly with `python`), where no session is running at all.
    return os.environ.get("PYTEST_VERSION") is not None


def allclose_tolerances(
    expected: NumpyArray,
    rtol: float = 1e-12,
) -> dict[str, float]:
    """
    Generate keyword-arguments for the tolerances of `np.testing.assert_allclose`.

    Parameters
    ----------
    expected
        Expected array of `np.testing.assert_allclose`.
    rtol
        The required relative tolerance.

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
    kwargs = {
        "rtol": 0,  # intentional 0, it makes problems at arrays that cross 0.
        "atol": amplitude * rtol,
    }
    return kwargs


def enforce_64_bit_backend():
    """Enforce 64-bit backend, GPU is taken into account."""
    if backend.float == np.float32:
        raise TypeError("32-bit float and 64-bit complex have been removed.")


_BLOND2_REFERENCE_ENVIRONMENT_KEY = "blond2_reference_environment"


def _git_commit(directory: str) -> str:
    """
    Identify the git commit that `directory` is checked out at.

    Parameters
    ----------
    directory
        Any directory inside the git work tree.

    Returns
    -------
    commit
        The commit hash, suffixed with ``-dirty`` if the tree has
        uncommitted changes, or ``unknown`` outside of a git work tree.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def save_blond2_reference_file(path: str, **arrays: NumpyArray) -> None:
    """
    Save reference arrays together with the environment that produced them.

    BLonD 2 reference files freeze the output of code that should no longer run in the
    test suite (e.g. BLonD 2 in the legacy regression tests). Next to the
    arrays, the ``blond2_reference_environment`` entry stores a JSON string with the
    installed packages (``pip_list``), the Python version, the platform,
    the git commit and the creation time, so a later mismatch can be traced
    back to a version drift.

    Parameters
    ----------
    path
        Target ``.npz`` file.
    **arrays
        The arrays to store, as for `numpy.savez`.

    Raises
    ------
    ValueError
        If one of the arrays uses the reserved name ``blond2_reference_environment``.

    Examples
    --------
    >>> save_blond2_reference_file("resources/blond2_reference.npz", dt=beam.dt, dE=beam.dE)
    >>> with np.load("resources/blond2_reference.npz") as blond2_reference:
    ...     dt = blond2_reference["dt"]
    """
    import datetime
    import json
    import platform
    import sys
    from importlib import metadata

    if _BLOND2_REFERENCE_ENVIRONMENT_KEY in arrays:
        raise ValueError(
            f"'{_BLOND2_REFERENCE_ENVIRONMENT_KEY}' is a reserved name."
        )

    pip_list = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
    )
    environment = {
        "pip_list": pip_list,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(os.path.dirname(os.path.abspath(path))),
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    np.savez(
        path,
        **{
            _BLOND2_REFERENCE_ENVIRONMENT_KEY: json.dumps(
                environment, indent=1
            )
        },
        **arrays,
    )
