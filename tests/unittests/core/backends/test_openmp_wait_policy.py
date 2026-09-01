# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the OpenMP wait-policy default applied on importing BLonD."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from blond.core.backends.openmp_env import (
    OMP_WAIT_POLICY,
    set_default_openmp_wait_policy,
)


class TestSetDefaultOpenmpWaitPolicy(unittest.TestCase):
    """`set_default_openmp_wait_policy` sets a default without overriding."""

    def setUp(self) -> None:
        self._saved = os.environ.get(OMP_WAIT_POLICY)
        os.environ.pop(OMP_WAIT_POLICY, None)

    def tearDown(self) -> None:
        os.environ.pop(OMP_WAIT_POLICY, None)
        if self._saved is not None:
            os.environ[OMP_WAIT_POLICY] = self._saved

    def test_sets_passive_when_unset(self) -> None:
        """An unset wait policy defaults to 'passive'."""
        set_default_openmp_wait_policy()
        self.assertEqual(os.environ[OMP_WAIT_POLICY], "passive")

    def test_does_not_override_user_choice(self) -> None:
        """A wait policy chosen by the user is left untouched."""
        os.environ[OMP_WAIT_POLICY] = "active"
        set_default_openmp_wait_policy()
        self.assertEqual(os.environ[OMP_WAIT_POLICY], "active")

    def test_does_not_override_empty_user_choice(self) -> None:
        """An explicitly emptied wait policy is left untouched."""
        os.environ[OMP_WAIT_POLICY] = ""
        set_default_openmp_wait_policy()
        self.assertEqual(os.environ[OMP_WAIT_POLICY], "")


class TestImportingBlondSetsWaitPolicy(unittest.TestCase):
    """Importing `blond` applies the wait-policy default."""

    @staticmethod
    def _clean_env() -> dict[str, str]:
        """Return the environment without BLonD-specific overrides.

        `test_backend.test_apply_environment_variables` leaks
        `BLOND_BACKEND_MODE=fail` into `os.environ`, which would make the
        subprocess below fail on import. `PYCHARM_HOSTED` makes colorama
        treat the captured pipe as a TTY and pollute stdout.
        """
        env = os.environ.copy()
        for key in (
            "BLOND_BACKEND_MODE",
            "BLOND_BACKEND_BITS",
            "PYCHARM_HOSTED",
            OMP_WAIT_POLICY,
        ):
            env.pop(key, None)
        return env

    def _wait_policy_after_import(self, env: dict[str, str]) -> str:
        """Return `OMP_WAIT_POLICY` seen by a subprocess that imports blond."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import blond, os;"
                f" print(repr(os.environ.get({OMP_WAIT_POLICY!r})))",
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return result.stdout.strip().splitlines()[-1]

    def test_import_sets_passive(self) -> None:
        """A fresh interpreter importing blond ends up with 'passive'."""
        env = self._clean_env()
        self.assertEqual(self._wait_policy_after_import(env), "'passive'")

    def test_import_keeps_user_value(self) -> None:
        """A user-provided wait policy survives importing blond."""
        env = self._clean_env()
        env[OMP_WAIT_POLICY] = "active"
        self.assertEqual(self._wait_policy_after_import(env), "'active'")


if __name__ == "__main__":
    unittest.main()
