# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for ``dev_tools/precommit_check_env_vars.py``."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from blond.testing.backend_testing import BLonDTestCase

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so pre-commit can run it without blond).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "dev_tools" / "precommit_check_env_vars.py"
_spec = importlib.util.spec_from_file_location(
    "precommit_check_env_vars", _SCRIPT
)
check_env_vars = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_env_vars)


class TestEnvVarsInSource(BLonDTestCase):
    def test_finds_name_passed_to_os_environ(self):
        source = 'import os\nos.environ.get("BLOND_FOO", "1")\n'
        self.assertEqual(
            check_env_vars.env_vars_in_source(source), {"BLOND_FOO": 2}
        )

    def test_finds_name_bound_to_a_constant(self):
        source = 'ENV_KEY = "BLOND_FOO"\n'
        self.assertEqual(
            check_env_vars.env_vars_in_source(source), {"BLOND_FOO": 1}
        )

    def test_reports_first_line_of_each_name(self):
        source = 'A = "BLOND_FOO"\nB = "BLOND_BAR"\nC = "BLOND_FOO"\n'
        self.assertEqual(
            check_env_vars.env_vars_in_source(source),
            {"BLOND_FOO": 1, "BLOND_BAR": 2},
        )

    def test_ignores_identifiers_docstrings_and_messages(self):
        source = (
            '"""Set ``BLOND_FOO`` to change this."""\n'
            "BLOND_ROOT = 1\n"
            'message = f"BLOND_FOO = {BLOND_ROOT}"\n'
        )
        self.assertEqual(check_env_vars.env_vars_in_source(source), {})


class TestPerformCheck(BLonDTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.package = root / "blond"
        self.package.mkdir()
        self.pyproject = root / "pyproject.toml"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_module(self, relative_path, source):
        path = self.package / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    def _write_declared(self, *entries):
        lines = "".join(f'  "{entry}",\n' for entry in entries)
        self.pyproject.write_text(f"[tool.blond]\nenv-vars = [\n{lines}]\n")

    def _check(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = check_env_vars.perform_check(
                self.package, self.pyproject
            )
        return exit_code, output.getvalue()

    def test_passes_when_used_and_declared_match(self):
        self._write_module("a.py", 'import os\nos.getenv("BLOND_FOO")\n')
        self._write_declared("BLOND_FOO: Does foo.")

        exit_code, output = self._check()

        self.assertEqual(exit_code, 0, output)

    def test_fails_on_undeclared_variable(self):
        self._write_module("a.py", 'import os\nos.getenv("BLOND_FOO")\n')
        self._write_declared()

        exit_code, output = self._check()

        self.assertEqual(exit_code, 1)
        self.assertIn('"BLOND_FOO: <what it does>",', output)
        # Repo-relative file:line, so it is short and clickable
        self.assertIn("blond/a.py:2", output)
        self.assertNotIn(self._tmp.name, output)

    def test_fails_on_declared_but_unused_variable(self):
        self._write_declared("BLOND_GONE: Not read anywhere any more.")

        exit_code, output = self._check()

        self.assertEqual(exit_code, 1)
        self.assertIn("BLOND_GONE", output)

    def test_fails_on_entry_without_description(self):
        self._write_module("a.py", 'import os\nos.getenv("BLOND_FOO")\n')
        self._write_declared("BLOND_FOO")

        exit_code, output = self._check()

        self.assertEqual(exit_code, 1)
        self.assertIn("description", output)
        self.assertIn('"BLOND_FOO: <what it does>",', output)
        # Only the description is missing; asking to add the variable
        # again would make the developer create a duplicate entry.
        self.assertNotIn("Undeclared", output)

    def test_fails_without_env_vars_table(self):
        self.pyproject.write_text("[tool.other]\n")

        exit_code, output = self._check()

        self.assertEqual(exit_code, 1)
        # A paste-ready snippet, not just the name of the missing table
        self.assertIn("[tool.blond]\nenv-vars = [", output)

    def test_skips_legacy_and_experimental(self):
        self._write_module("legacy/a.py", 'import os\nos.getenv("BLOND_X")\n')
        self._write_module(
            "experimental/b.py", 'import os\nos.getenv("BLOND_Y")\n'
        )
        self._write_declared()

        exit_code, output = self._check()

        self.assertEqual(exit_code, 0, output)


class TestRepository(BLonDTestCase):
    def test_every_env_var_of_blond_is_declared(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = check_env_vars.perform_check(
                _REPO_ROOT / "blond", _REPO_ROOT / "pyproject.toml"
            )

        self.assertEqual(exit_code, 0, output.getvalue())


if __name__ == "__main__":
    unittest.main()
