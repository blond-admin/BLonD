import os
import subprocess
import sys
import unittest
from unittest import mock

from blond.testing import pytest_active


class TestCallables(unittest.TestCase):
    def test_pytest_active(self):
        self.assertTrue(pytest_active())


class TestPytestActiveOutsideRunningTest(unittest.TestCase):
    """`pytest_active` must track the pytest *session*, nothing else.

    Two failure modes are guarded against here. Reporting ``False``
    during collection lets module level code guarded by
    ``if not pytest_active()`` mutate global state for the whole
    session. Reporting ``True`` merely because ``pytest`` is importable
    disables that code in ordinary scripts that happen to import pytest.
    """

    @staticmethod
    def _run_outside_pytest(code: str) -> str:
        """Run `code` in a child interpreter with no pytest session."""
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PYTEST_")
            and not key.startswith("COV_")
            and not key.startswith("COVERAGE_")
        }
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return completed.stdout

    def test_active_when_current_test_env_var_is_absent(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            self.assertTrue(pytest_active())

    def test_inactive_in_a_plain_interpreter(self):
        stdout = self._run_outside_pytest(
            "from blond.testing import pytest_active;"
            "print('PYTEST_ACTIVE', pytest_active())"
        )
        self.assertIn("PYTEST_ACTIVE False", stdout)

    def test_inactive_when_pytest_is_merely_imported(self):
        stdout = self._run_outside_pytest(
            "import pytest;"
            "from blond.testing import pytest_active;"
            "print('PYTEST_ACTIVE', pytest_active())"
        )
        self.assertIn("PYTEST_ACTIVE False", stdout)


if __name__ == "__main__":
    unittest.main()
