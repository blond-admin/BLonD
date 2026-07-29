# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the build-option-aware helpers `reload_cpp_backend` uses to
locate and load the compiled C++ backend."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from blond.core.backends.cpp import callables
from blond.core.backends.cpp import compiled_dir_handler as lc

_HAS_GPP = shutil.which("g++") is not None


def _default_options(**overrides):
    options = {
        "compiler": "g++",
        "optimize": True,
        "flags": "",
        "libs": "",
        "with_fftw": False,
        "with_fftw_threads": False,
        "with_fftw_omp": False,
        "with_fftw_lib": None,
        "with_fftw_header": None,
        "boost": None,
    }
    options.update(overrides)
    return options


class TestCallables(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.folder = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_get_platform(self):
        with mock.patch.object(os, "name", "posix"):
            self.assertEqual(callables._get_platform(), "posix")

        with (
            mock.patch.object(os, "name", "nt"),
            mock.patch.object(sys, "platform", "win32"),
        ):
            self.assertEqual(callables._get_platform(), "win")

        with (
            mock.patch.object(os, "name", "nt"),
            mock.patch.object(sys, "platform", "some-other-os"),
        ):
            with self.assertRaises(ValueError):
                callables._get_platform()

    def test_cpp_basepath_default(self):
        with mock.patch(
            "blond.core.backends.cpp.compile.compile_cpp_library"
        ) as mocked_compile:
            basepath = callables._resolve_cpp_basepath(self.folder)
        self.assertEqual(basepath, lc.cpp_compiled_dir(self.folder))
        mocked_compile.assert_not_called()

    def test_cpp_basepath_valid_options(self):
        if not _HAS_GPP:
            self.skipTest("needs g++")
        custom_options = _default_options(flags="-DFOO")
        lc.save_build_options(self.folder, **custom_options)
        # The directory built for these options must exist on disk for
        # them to be used -- otherwise `_resolve_cpp_basepath` falls back
        # to the default, see `test_cpp_basepath_valid_options_missing_dir`.
        os.makedirs(
            lc.cpp_compiled_dir(self.folder, **custom_options),
            exist_ok=True,
        )

        with mock.patch(
            "blond.core.backends.cpp.compile.compile_cpp_library"
        ) as mocked_compile:
            basepath = callables._resolve_cpp_basepath(self.folder)

        self.assertEqual(
            basepath, lc.cpp_compiled_dir(self.folder, **custom_options)
        )
        mocked_compile.assert_not_called()

    def test_cpp_basepath_valid_options_missing_dir(self):
        if not _HAS_GPP:
            self.skipTest("needs g++")
        custom_options = _default_options(flags="-DFOO")
        lc.save_build_options(self.folder, **custom_options)
        # Deliberately do not create the directory these options hash to
        # (e.g. it was evicted from the compiled-directory LRU cache).

        with mock.patch(
            "blond.core.backends.cpp.compile.compile_cpp_library"
        ) as mocked_compile:
            with self.assertWarns(UserWarning) as ctx:
                basepath = callables._resolve_cpp_basepath(self.folder)

        # Falls back to the default directory rather than recompiling
        # with, or reusing, the stale custom options.
        self.assertEqual(basepath, lc.cpp_compiled_dir(self.folder))
        self.assertIn("-DFOO", str(ctx.warning))
        mocked_compile.assert_not_called()

    def test_cpp_basepath_invalid_options(self):

        bad_options = _default_options(
            compiler="definitely-not-a-real-compiler-xyz"
        )
        lc.save_build_options(self.folder, **bad_options)

        with mock.patch(
            "blond.core.backends.cpp.compile.compile_cpp_library"
        ) as mocked_compile:
            with self.assertWarns(UserWarning):
                basepath = callables._resolve_cpp_basepath(self.folder)

        self.assertEqual(basepath, lc.cpp_compiled_dir(self.folder))
        mocked_compile.assert_not_called()

    def test_make_libblond_path(self):
        with (
            mock.patch.object(
                callables, "_resolve_cpp_basepath", return_value="/base"
            ),
            mock.patch.object(os, "name", "posix"),
        ):
            basepath, libblond_path = callables._make_libblond_path(
                "/folder", "double"
            )
        self.assertEqual(basepath, "/base")
        self.assertEqual(
            libblond_path, os.path.join("/base", "libblond_double.so")
        )

        with (
            mock.patch.object(
                callables, "_resolve_cpp_basepath", return_value="/base"
            ),
            mock.patch.object(os, "name", "nt"),
            mock.patch.object(sys, "platform", "win32"),
        ):
            basepath, libblond_path = callables._make_libblond_path(
                "/folder", "double_noOMP"
            )
        self.assertEqual(
            libblond_path,
            os.path.join("/base", "libblond_double_noOMP.dll"),
        )

    def test_get_libblond_posix(self):
        with (
            mock.patch.object(
                callables, "_get_platform", return_value="posix"
            ),
            mock.patch.object(
                callables.ct, "CDLL", return_value="FAKE_LIB"
            ) as mocked_cdll,
        ):
            result = callables._get_libblond("/path/to/lib.so")
        self.assertEqual(result, "FAKE_LIB")
        mocked_cdll.assert_called_once_with("/path/to/lib.so")

    def test_get_libblond_win_with_add_dll(self):
        with (
            mock.patch.object(callables, "_get_platform", return_value="win"),
            mock.patch.object(os, "add_dll_directory", create=True),
            mock.patch.object(
                callables.ct, "CDLL", return_value="FAKE_LIB"
            ) as mocked_cdll,
        ):
            result = callables._get_libblond("/path/to/lib.dll")
        self.assertEqual(result, "FAKE_LIB")
        mocked_cdll.assert_called_once_with("/path/to/lib.dll", winmode=0)

    def test_get_libblond_win_without_add_dll(self):
        if hasattr(os, "add_dll_directory"):
            self.skipTest(
                "host has add_dll_directory; cannot exercise the "
                "posix-Python-on-Windows fallback branch"
            )
        with (
            mock.patch.object(callables, "_get_platform", return_value="win"),
            mock.patch.object(
                callables.ct, "CDLL", return_value="FAKE_LIB"
            ) as mocked_cdll,
        ):
            result = callables._get_libblond("/path/to/lib.dll")
        self.assertEqual(result, "FAKE_LIB")
        mocked_cdll.assert_called_once_with("/path/to/lib.dll")


if __name__ == "__main__":
    unittest.main()
