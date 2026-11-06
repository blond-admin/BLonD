# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for ``dev_tools/precommit_check_docstrings.py``."""

import contextlib
import importlib.util
import io
import shutil
import tempfile
import unittest
from pathlib import Path

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so pre-commit can run it without blond).
_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "dev_tools"
    / "precommit_check_docstrings.py"
)
_spec = importlib.util.spec_from_file_location(
    "precommit_check_docstrings", _SCRIPT
)
check_docstrings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docstrings)

RUFF_AVAILABLE = shutil.which("ruff") is not None


class TestPublicPath(unittest.TestCase):
    def test_private_folders_and_files_lose_their_underscore(self):
        self.assertEqual(
            check_docstrings.public_path("blond/_backends/_helpers.py"),
            "blond/backends/helpers.py",
        )

    def test_dunder_files_are_kept(self):
        self.assertEqual(
            check_docstrings.public_path("blond/_backends/__init__.py"),
            "blond/backends/__init__.py",
        )

    def test_public_paths_are_unchanged(self):
        self.assertEqual(
            check_docstrings.public_path("blond/generals/cupy_/x.py"),
            "blond/generals/cupy_/x.py",
        )


@unittest.skipUnless(RUFF_AVAILABLE, "ruff is not installed")
class TestMain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.package = Path(self._tmp.name) / "_private_package"
        self.package.mkdir()
        (self.package / "__init__.py").write_text('"""Doc."""\n', "utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, source: str) -> tuple[int, str, Path]:
        path = self.package / "_private_module.py"
        path.write_text(source, encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = check_docstrings.main([str(path)])
        return exit_code, output.getvalue(), path

    def test_public_function_in_private_path_is_reported(self):
        exit_code, output, path = self._run(
            '"""Doc."""\n\n\ndef public():\n    pass\n'
        )
        self.assertEqual(exit_code, 1)
        self.assertIn(f"{path}:4:5: D103", output)

    def test_ruff_rules_still_apply_to_names(self):
        exit_code, output, _ = self._run(
            '"""Doc."""\n\n\n'
            "def _private():\n    pass\n\n\n"
            "def suppressed():  # NOQA: D103\n    pass\n"
        )
        self.assertEqual((exit_code, output), (0, ""))

    def test_public_paths_are_left_to_ruff(self):
        path = Path(self._tmp.name) / "public_module.py"
        path.write_text("def public():\n    pass\n", encoding="utf-8")
        self.assertEqual(check_docstrings.main([str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
