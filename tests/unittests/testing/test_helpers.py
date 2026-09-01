import os
import sys
import unittest
from unittest import mock

from blond.testing import pytest_active


class TestCallables(unittest.TestCase):
    def test_pytest_active(self):
        self.assertTrue(pytest_active())


class TestPytestActiveTracksTheSession(unittest.TestCase):
    """`pytest_active` must track the pytest *session*, nothing else.

    Reporting ``False`` during collection lets module level code guarded
    by ``if not pytest_active()`` mutate global state for the rest of the
    session. Reporting ``True`` merely because ``pytest`` is importable
    disables that code in ordinary scripts that happen to import pytest.
    """

    def test_active_while_no_test_is_executing(self):
        # `PYTEST_CURRENT_TEST` is absent during collection, but the
        # session is running and must be reported as such.
        with mock.patch.dict(os.environ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            self.assertTrue(pytest_active())

    def test_inactive_when_pytest_is_merely_imported(self):
        # A test file run directly with `python` imports pytest without
        # ever starting a session.
        self.assertIn("pytest", sys.modules)
        with mock.patch.dict(os.environ):
            for name in [
                key for key in os.environ if key.startswith("PYTEST_")
            ]:
                os.environ.pop(name)
            self.assertFalse(pytest_active())


if __name__ == "__main__":
    unittest.main()
