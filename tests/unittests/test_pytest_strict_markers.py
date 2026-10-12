# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from blond.testing.backend_testing import BLonDTestCase

PYPROJECT_TOML = Path(__file__).parents[2] / "pyproject.toml"

TEST_WITH_UNREGISTERED_MARKER = """
import pytest


@pytest.mark.not_a_registered_marker
def test_dummy():
    pass
"""


class TestPytestStrictMarkers(BLonDTestCase):
    """A misspelled marker (e.g. ``@pytest.mark.backend_mutaton``) is
    silently ignored by pytest unless ``--strict-markers`` is set, so a
    test would quietly escape the ``-m "not backend_mutation"`` filter.
    """

    def test_unregistered_marker_fails_collection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test_unregistered_marker.py"
            test_file.write_text(TEST_WITH_UNREGISTERED_MARKER)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-c",
                    str(PYPROJECT_TOML),
                    "--rootdir",
                    tmp_dir,
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "no:randomly",
                    str(test_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(
            "'not_a_registered_marker' not found in `markers`",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
