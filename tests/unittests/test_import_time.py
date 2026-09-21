# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import subprocess
import sys
import unittest

from blond.testing.backend_testing import BLonDTestCase

LINE_DENSITY_NUMBA_MODULE = (
    "blond.experimental.beam_preparation.semi_empiric_matcher_extensions"
    ".line_density.callables_numba"
)


class TestImportTime(BLonDTestCase):
    """Modules that compile Numba kernels eagerly (explicit signatures)
    cost seconds at import. They must only be imported just in time, not
    as a side effect of ``import blond``.
    """

    def test_import_blond_does_not_compile_line_density_kernels(self):
        # Fresh interpreter: ``sys.modules`` of the test process is
        # already populated by other tests.
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore",
                "-c",
                "import sys; import blond; "
                f"print({LINE_DENSITY_NUMBA_MODULE!r} in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # ``import blond`` may itself write to stdout (e.g. "Using
        # environment variable BLOND_BACKEND_MODE = ..." in CI), so only
        # the last line is the answer.
        self.assertEqual(completed.stdout.splitlines()[-1], "False")


if __name__ == "__main__":
    unittest.main()
