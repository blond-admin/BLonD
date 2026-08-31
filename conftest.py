# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Session-wide pytest configuration.

Keeps the global numeric backends in a fast, deterministic state before every
test. Both the legacy BLonD 2 ``bm`` singleton and the BLonD 3 backend are
process-global mutable objects; under ``pytest-randomly`` a test could
otherwise inherit a slow (pure-python) backend left active by whatever ran
before it, which made the BLonD 2 regression tests an order-dependent
performance sink (the same ~1000 s of slowness roamed between tests run to
run). See :func:`blond.testing.backend_testing.pin_fast_test_backends`.

Also pins a NON-INTERACTIVE matplotlib backend. Several tests call
``plt.show()`` without guarding it behind a debug flag; with a GUI
backend that call blocks on a window nobody is there to close, so a full
``pytest tests/unittests/`` run stops dead and looks exactly like a hang
(no output, high CPU). ``Agg`` makes ``show()`` a no-op. An explicit
``MPLBACKEND`` is honoured, so a developer can still
``MPLBACKEND=TkAgg pytest ...`` to look at a plot.
"""

import os

import matplotlib
import pytest

from blond.testing.backend_testing import pin_fast_test_backends

if not os.environ.get("MPLBACKEND"):
    # force=True: some import chain may already have selected a GUI
    # backend by the time conftest is imported.
    matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _pin_fast_test_backends():
    """Reset the numeric backends to a fast default before each test."""
    pin_fast_test_backends()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """
    Close any figures a test left open.

    Figures are held by a process-global registry, so without this a full
    run accumulates every figure every test ever drew -- memory that grows
    monotonically, plus matplotlib's "More than 20 figures" warning
    drowning the log.

    Yields
    ------
    None
        Control returns to the test; the figures are closed afterwards.
    """
    yield
    # Imported here, not at module scope: the backend is selected
    # above and importing pyplot is what locks it in.
    import matplotlib.pyplot as plt

    plt.close("all")
