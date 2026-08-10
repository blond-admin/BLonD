"""
Pin the backend for the muon-collider feedback test package.

The mucol suites are written and validated against the import-time default
backend (``Numpy64Bit`` with the ``python`` specials): the cavity-feedback
signal processing is host-only by design, and the tests build/inspect
profile arrays with plain NumPy. In a full-suite session the global backend
can arrive in any state (``backend_mutation`` tests flip it mid-run, and
``pytest-randomly`` interleaves them), so every test here re-pins the
default backend first -- otherwise a leaked GPU backend makes e.g.
``profile.hist_x`` a CuPy array and the NumPy-based helpers fail with
``TypeError: Unsupported type <class 'numpy.ndarray'>``.

Pinning without restore is itself a backend mutation, so all tests in this
package are marked ``backend_mutation``, mirroring the LHC blond2-comparison
package.

The pin must land *before* ``setUpClass``: several suites here build their
profiles, beams and simulations there (a fixture, even an autouse one, runs
only after ``setUpClass`` and would leave that work on whatever backend
happened to be active). ``pytest_runtest_setup`` runs earlier than any
fixture, so the pin is applied from there and the autouse fixture only
re-applies it per test.

Setting ``BLOND_BACKEND_MODE=cuda`` (the repo-wide backend selector) pins to
the GPU backend instead, which is how the package's CuPy-safety is
validated -- the tests transfer device arrays explicitly with
``copy_to_cpu`` and must pass on either backend. Any other value keeps the
host pin.
"""

import os
from pathlib import Path

import pytest

from blond import Cupy64Bit, Numpy64Bit, backend

_THIS_DIR = Path(__file__).parent

#: True when the session was explicitly pointed at the GPU backend; the pin
#: then keeps CuPy instead of forcing the host backend back on.
_PIN_TO_GPU = os.environ.get("BLOND_BACKEND_MODE", "").lower() == "cuda"


def _pin_backend() -> None:
    """Re-pin this package's backend, host by default (GPU when selected)."""
    if _PIN_TO_GPU:
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
    else:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")


def _is_in_package(item) -> bool:
    """
    Whether a collected test item belongs to this package.

    Parameters
    ----------
    item
        The collected test item.

    Returns
    -------
    in_package
        True when the item's file lives under this package's directory.
    """
    try:
        return Path(item.path).is_relative_to(_THIS_DIR)
    except (TypeError, ValueError):
        return False


def pytest_collection_modifyitems(items):
    """
    Mark every test in this package as a backend mutator.

    Parameters
    ----------
    items
        The collected test items (session-wide; only items under this
        package's directory are marked).
    """
    for item in items:
        if _is_in_package(item):
            item.add_marker(pytest.mark.backend_mutation)


def pytest_runtest_setup(item):
    """
    Pin the backend before ``setUpClass``, which fixtures cannot precede.

    Parameters
    ----------
    item
        The test item about to run; only items of this package are pinned.
    """
    if _is_in_package(item):
        _pin_backend()


@pytest.fixture(autouse=True)
def _pin_default_backend():
    """Re-pin the backend before every mucol test."""
    _pin_backend()
    yield
