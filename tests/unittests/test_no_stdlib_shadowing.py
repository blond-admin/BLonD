# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import sys
import unittest
from pathlib import Path

import blond

BLOND_PACKAGE_DIR = Path(blond.__file__).parent


class TestNoStdlibShadowing(unittest.TestCase):
    """If ``blond/`` itself ends up on ``sys.path`` (e.g. an IDE or a
    script adds the package directory instead of its parent), any
    top-level module inside it that shares a name with a Python
    standard library module silently shadows that module for the
    whole interpreter. ``blond/typing.py`` is such a case.
    """

    def test_top_level_modules_do_not_shadow_stdlib(self):
        top_level_module_names = {
            path.stem
            for path in BLOND_PACKAGE_DIR.glob("*.py")
            if path.name != "__init__.py"
        }

        conflicts = sorted(top_level_module_names & sys.stdlib_module_names)

        self.assertEqual(
            conflicts,
            [],
            "The following files in blond/ shadow Python standard "
            "library modules if blond/ is added to sys.path: "
            f"{conflicts}",
        )


if __name__ == "__main__":
    unittest.main()
