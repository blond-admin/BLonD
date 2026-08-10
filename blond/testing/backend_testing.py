# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
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
from functools import partial, wraps
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends import backend
from blond.generals.cupy import no_cupy_import as no_cupy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from blond.blond_typing import AnyArray
    from blond.core.backends.backend import BackendBaseClass

try:
    import cupy  # noqa: F401
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


def _backend_selection(*args: tuple[str]) -> dict[str, BackendBaseClass]:
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


def pin_fast_test_backends() -> None:
    """
    Pin the global numeric backends to a fast, deterministic state.

    Intended to be called from an autouse test fixture so that per-test
    timing and behaviour do not depend on test order (``pytest-randomly``) or
    on a backend left active by an earlier test.

    The two backend systems are handled asymmetrically:

    - **BLonD 2**: if the legacy ``bm`` singleton has already been imported,
      it is reset to the fastest available CPU backend (``use_cpu``:
      cpp > numba > python). The legacy backend is a mutable global that many
      regression tests change without restoring; leaving it in pure-python
      mode made those tests an order-dependent performance sink.
    - **BLonD 3**: only the slow pure-python kernels are guarded against being
      the *ambient* default; ``python`` specials are switched to ``numba``.
      Tests that deliberately exercise the python backend set it themselves
      after this helper runs and are therefore unaffected.

    Notes
    -----
    The legacy backend is only touched when its module is already imported, so
    backend-agnostic BLonD 3 tests do not pay the cost of importing the legacy
    code. The first legacy import runs ``use_cpu`` itself, so the state is
    clean either way.
    """
    import sys

    legacy_utils = sys.modules.get("blond.legacy.blond2.utils")
    if legacy_utils is not None:
        legacy_utils.bmath.use_cpu()

    if backend.backend.specials_mode == "python":
        try:
            backend.backend.set_specials("cpp")
        except Exception as exc:
            warnings.warn(str(exc), stacklevel=1)
            backend.backend.set_specials("numba")


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
        specified as a str.  E.g. @multi_backend_testcase("Numpy64Bit")
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
    >>> # To run with only Numpy64Bit
    >>> @multi_backend_testcase("Numpy64Bit")
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
                except Exception as exc:
                    failed_on = backend.backend.__class__.__name__
                    # If a function call fails, force return to the
                    # initial condition, then re-raise the exception.
                    backend.backend.change_backend(
                        backend.ALL_BACKENDS[init_backend]
                    )
                    raise RuntimeError(
                        f"Failed with backend {failed_on}"
                    ) from exc
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


class ArrayLikeScan:
    """
    Convenience object to simplify testing different `ArrayLike`s.

    Simplifies the process of iterating over common `ArrayLike` types for
    testing `ArrayLike` inputs.  When an `iterator` is created from it,
    the return is a `Generator` that yields casting functions to type
    cast the input.  By default, it will iterate over list, tuple,
    np.array and (if cupy is available) cp.array.  If different types
    are required, they can be given at input when creating the object.

    Parameters
    ----------
    array_likes
        The casting functions to use (e.g. `list`, `tuple`, ...).
        If None, will be replaced with `[list, tuple, np.array]`,
        `cp.array` will be appended if cupy is available.

    Examples
    --------
    >>> for inp_cast in ArrayLikeScan():
    ...     np.max(inp_cast([1, 2, 3]))
    """

    def __init__(self, array_likes: Iterable[type] | None = None):
        if array_likes is None:
            array_likes = [list, tuple, np.array]
            if cupy_available:
                array_likes.append(cupy.array)

        self.array_likes = array_likes

    def __iter__(self):
        """A generator to iterate over casting options."""
        for type_ in self.array_likes:
            func = partial(self._cast_to, type_)

            yield func

    def _cast_to(self, type_: type, value: AnyArray) -> AnyArray:
        # Wrapper to ensure cupy arrays are first converted to numpy
        # arrays, otherwise most conversions raise an error
        # because automatic convertion (`.get()`) is not possible.
        if no_cupy.is_cupy_array(value):
            value = no_cupy.copy_to_cpu(value)

        return type_(value)
