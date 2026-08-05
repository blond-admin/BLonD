"""Tests for the C++ backend loading helpers."""

import os
import unittest
from unittest.mock import patch

import pytest

from blond.core.backends.backend import Numpy64Bit, backend


@unittest.skipUnless(
    hasattr(os, "add_dll_directory"),
    "os.add_dll_directory only exists on Windows",
)
class TestDllDirectoryIsAddedOnce(unittest.TestCase):
    """
    The Windows DLL search path must not grow on repeated backend loads.

    ``os.add_dll_directory`` *appends* to the process-wide DLL search path
    and returns a handle that keeps the entry alive. Adding the same
    directory again on every ``set_specials("cpp")`` grows that internal
    path until Windows refuses further additions with ``OSError``
    (``WinError 206``, "filename or extension is too long") -- naming the
    directory just offered, which makes a short path look like the culprit.
    In a test session that repeatedly activates the C++ backend this
    surfaces as a spurious "C++ backend was not found" plus a recompile
    attempt on every subsequent load.
    """

    @pytest.mark.backend_mutation
    def test_repeated_activation_adds_the_directory_only_once(self):
        """Activating the C++ backend repeatedly adds one search path."""
        real_add_dll_directory = os.add_dll_directory
        added_directories = []

        def spy(path):
            added_directories.append(path)
            return real_add_dll_directory(path)

        with patch.object(os, "add_dll_directory", spy):
            for _ in range(5):
                backend.set_specials("cpp")

        self.assertLessEqual(
            len(added_directories),
            1,
            "os.add_dll_directory was called "
            f"{len(added_directories)} times for "
            f"{set(added_directories)}; each call appends to the "
            "process-wide DLL search path, which eventually overflows "
            "with WinError 206.",
        )

    def tearDown(self) -> None:
        """Restore the default backend for the rest of the session."""
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
