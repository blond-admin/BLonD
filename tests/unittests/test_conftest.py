import os
import unittest

import matplotlib


class TestSessionConfiguration(unittest.TestCase):
    """The session-wide setup of the root ``conftest.py``."""

    @unittest.skipIf(
        "MPLBACKEND" in os.environ,
        "the backend was chosen explicitly through the environment",
    )
    def test_plotting_runs_on_a_non_interactive_backend(self):
        """Tests must not depend on a window server.

        An interactive backend makes the plotting tests fail in a full run
        under some ``pytest-randomly`` seeds, when a figure is collected
        after Tk has been torn down.
        """
        self.assertEqual(matplotlib.get_backend().lower(), "agg")


if __name__ == "__main__":
    unittest.main()
