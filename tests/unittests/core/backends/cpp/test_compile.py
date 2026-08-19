"""Tests for the C++ backend loading helpers."""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp import compile as cpp_compile
from blond.core.backends.cpp.compile import add_dll_directory_once


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

    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

    @pytest.mark.backend_mutation
    def test_repeated_activation_adds_the_directory_only_once(self):
        """Activating the C++ backend repeatedly adds one search path."""
        directory = os.path.normcase(os.path.abspath("/path/to/be/tested"))
        mock_add_dll_directory = MagicMock()

        with (
            patch.object(cpp_compile, "_added_dll_directory_keys", set()),
            patch.object(
                os, "add_dll_directory", mock_add_dll_directory, create=True
            ),
        ):
            for _ in range(500):
                add_dll_directory_once(directory)

            self.assertSetEqual(
                cpp_compile._added_dll_directory_keys, {directory}
            )
            mock_add_dll_directory.assert_called_once_with(directory)

    def tearDown(self) -> None:
        """
        Restore the default CPU backend for the rest of the session.

        ``tearDown`` also runs after a skip, so it must not force a numpy
        backend onto a session that is running on the GPU -- there the test
        body changed nothing.
        """
        if backend.is_gpu:
            return
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
