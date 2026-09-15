# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Session-wide pytest configuration.

Pins the matplotlib backend and keeps the global numeric backends in a fast,
deterministic state before every
test. Both the legacy BLonD 2 ``bm`` singleton and the BLonD 3 backend are
process-global mutable objects; under ``pytest-randomly`` a test could
otherwise inherit a slow (pure-python) backend left active by whatever ran
before it, which made the BLonD 2 regression tests an order-dependent
performance sink (the same ~1000 s of slowness roamed between tests run to
run). See :func:`blond.testing.backend_testing.pin_fast_test_backends`.
"""

import os

import matplotlib
import pytest

from blond.testing.backend_testing import pin_fast_test_backends

# Tests must never need a window server. An interactive backend (`tkagg` is
# the default wherever tkinter is installed) makes the plotting tests depend
# on Tk teardown order, which `pytest-randomly` reshuffles every run: a
# figure collected after Tk is gone raises
# `TclError: application has been destroyed`, so a test that passes alone
# fails in a full run under some seeds. `.gitlab-ci.yml` sets `MPLBACKEND`
# for most jobs already; doing it here covers the remaining ones and every
# developer machine. An explicitly chosen backend is left alone.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _pin_fast_test_backends():
    """Reset the numeric backends to a fast default before each test."""
    pin_fast_test_backends()
