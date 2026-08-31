# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
The test session must run on a non-interactive matplotlib backend.

Several tests call ``plt.show()`` without guarding it behind a debug flag.
On a GUI backend that blocks on a window nobody is there to close, so a full
``pytest tests/unittests/`` run stops dead and looks exactly like a hang --
no output, high CPU, indistinguishable from a slow test. The root
``conftest.py`` pins ``Agg`` unless ``MPLBACKEND`` says otherwise; these
tests pin that arrangement.
"""

import unittest

import matplotlib
import matplotlib.pyplot as plt

#: Backend name prefixes that open a window and therefore block on show().
INTERACTIVE_PREFIXES = (
    "tk",
    "qt",
    "gtk",
    "wx",
    "macosx",
    "web",
    "nbagg",
)


class TestMatplotlibBackend(unittest.TestCase):
    """The session backend must not be able to block on ``show()``."""

    def test_backend_is_non_interactive(self):
        """No GUI backend, whatever the developer's environment."""
        backend = matplotlib.get_backend().lower()
        self.assertFalse(
            backend.startswith(INTERACTIVE_PREFIXES),
            msg=(
                f"matplotlib backend {backend!r} opens windows; an "
                "unguarded plt.show() in any test would block the whole "
                "run. See the backend pin in the root conftest.py."
            ),
        )

    def test_show_returns_instead_of_blocking(self):
        """``show()`` is a no-op, so an unguarded call cannot hang."""
        figure = plt.figure()
        try:
            plt.show()
        finally:
            plt.close(figure)
