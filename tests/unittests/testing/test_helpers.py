import os
import sys
import unittest
from unittest import mock

from blond.testing import pytest_active
from blond.testing.backend_testing import BLonDTestCase


class TestCallables(BLonDTestCase):
    def test_pytest_active(self):
        self.assertTrue(pytest_active())


class TestPytestActiveTracksTheSession(BLonDTestCase):
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


class TestSaveGoldenFile(BLonDTestCase):
    """`save_golden_file` stores arrays plus the producing environment."""

    def setUp(self):
        import tempfile

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, "golden.npz")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_arrays_round_trip(self):
        import numpy as np

        from blond.testing.helpers import save_golden_file

        save_golden_file(self.path, dt=np.arange(3.0), dE=np.ones(2))

        with np.load(self.path) as golden:
            np.testing.assert_array_equal(golden["dt"], np.arange(3.0))
            np.testing.assert_array_equal(golden["dE"], np.ones(2))

    def test_environment_is_stored_without_pickle(self):
        import json

        import numpy as np

        from blond.testing.helpers import save_golden_file

        save_golden_file(self.path, dt=np.arange(3.0))

        with np.load(self.path, allow_pickle=False) as golden:
            environment = json.loads(str(golden["golden_environment"]))
        self.assertIn(f"numpy=={np.__version__}", environment["pip_list"])
        self.assertEqual(environment["python"], sys.version)
        for key in ("platform", "git_commit", "created"):
            self.assertIn(key, environment)

    def test_refuses_reserved_key(self):
        import numpy as np

        from blond.testing.helpers import save_golden_file

        with self.assertRaises(ValueError):
            save_golden_file(self.path, golden_environment=np.ones(1))
