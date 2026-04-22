# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Convenience decorators to help with unit testing.

Notes
-----
Authors:
Simon Albright
"""

from __future__ import annotations

import os
import warnings
from functools import wraps
from typing import TYPE_CHECKING

from blond.core.backends import backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from blond.core.backends.backend import BackendBaseClass

try:
    import cupy  # noqa: F401  # ty: ignore[unresolved-import]
except ImportError:
    cupy_available = False
else:
    cupy_available = True


def _set_forcing() -> bool:
    flag = os.environ.get("BLOND_FORCE_TEST_ALL_BACKENDS", "False")
    if flag not in ("True", "False"):
        raise OSError(
            f"BLOND_FORCE_TEST_ALL_BACKENDS environment variable must be either True or False, not {flag}"
        )
    else:
        return flag == "True"


FORCE_ALL_BACKENDS = _set_forcing()


def _backend_selection(*args: str) -> list[type[BackendBaseClass]]:
    if FORCE_ALL_BACKENDS:
        # If FORCE_ALL_BACKENDS is True, the requested backends will all
        # be used, whether or not they can be initialised.  For backends
        # that cannot be initialised, they will fail when calling
        # backend.change_backend in multi_backend_testcase.
        backends = [backend.ALL_BACKENDS[b] for b in args]
    else:
        # If FORCE_ALL_BACKENDS is False, the requested backends will
        # only be used if they are available.  This allows the testcase
        # to run with the available subset of backends.
        backends = []
        for b in args:
            try:
                backends.append(backend.AVAILABLE_BACKENDS[b])
            except KeyError:
                warnings.warn(
                    f"Backend {b} was requested but is not available"
                    ", it will not be included in testing.  To force"
                    " all backends to be considered, set "
                    "`backend_testing.FORCE_ALL_BACKENDS = True`",
                    stacklevel=2,
                )

    return backends


def multi_backend_testcase(*args: Any) -> Callable:
    """
    Decorator to run a unittest testcase with multiple backends.

    If used as a bare decorator, the decorated unittest will be run
    with all available backends.  If called with a list of str, only
    the corresponding backends will be run.

    The test case function will be called multiple times from within
    a for loop with the backend changed before each call of the
    function.  At the end, the backend will be changed to its initial
    value.

    `setUp` and `tearDown` will be called before and after each run
    of the test function.

    Parameters
    ----------
    *args
        If only specific backends are desired, their name should be
        specified as a str.  E.g. @multi_backend_testcase("Numpy32Bit")
        If no name is specified, all known backends are used.

    Returns
    -------
    Callable
        The wrapped test function.

    Examples
    --------
    >>> # To run with all known backends:
    >>> @multi_backend_testcase
    ... def testcase(self):
    ...    # Unit test code
    >>> # To run with only Numpy32Bit and Numpy64Bit
    >>> @multi_backend_testcase("Numpy32Bit", "Numpy64Bit")
    ... def testcase(self):
    ...    # Unit test code
    """
    bare = bool(callable(args[0]))

    if bare:
        fn = args[0]
        tested_backends = _backend_selection(*backend.ALL_BACKENDS.keys())
    else:
        tested_backends = _backend_selection(*args)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def multi_test(self):
            init_backend = backend.backend.__class__.__name__
            for t in tested_backends:
                backend.backend.change_backend(t)
                self.setUp()
                try:
                    fn(self)
                except Exception:
                    # If a function call fails, force return to the
                    # initial condition, then re-raise the exception.
                    backend.backend.change_backend(
                        backend.ALL_BACKENDS[init_backend]
                    )
                    raise
                self.tearDown()
            backend.backend.change_backend(backend.ALL_BACKENDS[init_backend])

        return multi_test

    if bare:
        return decorator(fn)
    else:
        return decorator


def skip_if_no_cupy(fn: Callable) -> Callable:
    """
    Convenience wrapper to skip a test case if cupy is not available.

    Parameters
    ----------
    fn
        The function to wrapped.

    Returns
    -------
    Callable
        The wrapped function.

    Examples
    --------
    >>> @skip_if_no_cupy
    ... def testcase(self):
    ...    # Test that only runs if cupy available
    """

    @wraps(fn)
    def func(self):
        if not cupy_available:
            self.skipTest(f"{cupy_available=}")
        else:
            fn(self)

    return func
