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

from functools import wraps
from typing import TYPE_CHECKING

from blond.core.backends import backend

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import cupy  # noqa: F401
except ImportError:
    cupy_available = False
else:
    cupy_available = True


def multi_backend_testcase(*args: tuple[str]) -> Callable:
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
        tested_backends = backend.ALL_BACKENDS.values()
    else:
        tested_backends = [backend.ALL_BACKENDS[b] for b in args]

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def multi_test(self):
            init_backend = backend.backend.__class__.__name__
            for t in tested_backends:
                backend.backend.change_backend(t)
                self.setUp()
                fn(self)
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
