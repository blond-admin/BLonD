"""Tests for the juliacall bootstrap of the Julia backends."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest

from blond.core.backends.julia import julia_env

JULIACALL_AVAILABLE = importlib.util.find_spec("juliacall") is not None

KERNELS_DIR = Path(julia_env.__file__).resolve().parent / "BLonDKernels"


def _skip_without_juliacall(test_case: unittest.TestCase) -> None:
    if not JULIACALL_AVAILABLE:
        test_case.skipTest("juliacall is not installed")


class TestIsJuliaAvailable(unittest.TestCase):
    """`is_julia_available` must be cheap and must not start Julia."""

    def test_returns_bool(self) -> None:
        self.assertIsInstance(julia_env.is_julia_available(), bool)

    def test_matches_find_spec(self) -> None:
        self.assertEqual(julia_env.is_julia_available(), JULIACALL_AVAILABLE)

    def test_false_when_juliacall_missing(self) -> None:
        with mock.patch.object(
            julia_env.importlib.util, "find_spec", return_value=None
        ):
            self.assertFalse(julia_env.is_julia_available())

    def test_does_not_import_juliacall(self) -> None:
        """Probing availability must not import (and thus start) Julia."""
        code = (
            "import sys;"
            "from blond.core.backends.julia.julia_env "
            "import is_julia_available;"
            "is_julia_available();"
            "print('juliacall' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")


class TestErrorsWithoutJuliacall(unittest.TestCase):
    """Both entry points must fail with an actionable `ImportError`."""

    def test_julia_kernels_raises_import_error(self) -> None:
        with mock.patch.object(
            julia_env, "is_julia_available", return_value=False
        ):
            with self.assertRaises(ImportError) as context:
                julia_env.julia_kernels()
        self.assertIn("blond[julia]", str(context.exception))

    def test_julia_cuda_kernels_raises_import_error(self) -> None:
        with mock.patch.object(
            julia_env, "is_julia_available", return_value=False
        ):
            with self.assertRaises(ImportError) as context:
                julia_env.julia_cuda_kernels()
        self.assertIn("blond[julia]", str(context.exception))


@pytest.mark.julia
class TestEnsureJuliaEnvironment(unittest.TestCase):
    """Bootstrap the Julia session and load `BLonDKernels`."""

    def setUp(self) -> None:
        _skip_without_juliacall(self)
        try:
            julia_env.ensure_julia_environment()
        except OSError as error:  # e.g. an incompatible libstdc++
            self.skipTest(str(error))

    def test_environment_variables_are_set(self) -> None:
        julia_env.ensure_julia_environment()
        self.assertEqual(os.environ["PYTHON_JULIACALL_THREADS"], "auto")
        self.assertEqual(os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"], "yes")

    def test_julia_kernels_is_cached(self) -> None:
        first = julia_env.julia_kernels()
        second = julia_env.julia_kernels()
        self.assertIs(first, second)

    def test_host_device_and_max_threads(self) -> None:
        kernels = julia_env.julia_kernels()
        device = kernels.host_device()
        self.assertGreaterEqual(int(kernels.max_threads(device)), 1)


@pytest.mark.julia
@pytest.mark.integration
class TestJuliaPackageTests(unittest.TestCase):
    """Run the `BLonDKernels` Julia test suite.

    Opt-in via ``BLOND_RUN_JULIA_PACKAGE_TESTS=True`` because
    ``Pkg.test`` instantiates a separate Julia environment, which is far
    too slow for the default unit-test run.
    """

    def setUp(self) -> None:
        _skip_without_juliacall(self)
        if os.environ.get("BLOND_RUN_JULIA_PACKAGE_TESTS", "") not in (
            "1",
            "True",
            "true",
        ):
            self.skipTest("BLOND_RUN_JULIA_PACKAGE_TESTS is not set")
        if not (KERNELS_DIR / "test" / "runtests.jl").is_file():
            self.skipTest(f"No Julia test suite in {KERNELS_DIR}")

    def test_pkg_test(self) -> None:
        julia_env.ensure_julia_environment()
        from juliacall import Main as jl  # type: ignore

        jl.seval("using Pkg")
        jl.seval(f'Pkg.test("{julia_env.JULIA_PACKAGE_NAME}")')
