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
"""

import pytest

from blond.testing.backend_testing import pin_fast_test_backends


@pytest.fixture(autouse=True)
def _pin_fast_test_backends():
    """Reset the numeric backends to a fast default before each test."""
    pin_fast_test_backends()
