# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for build-option persistence and validity checks used to decide
whether a saved C++ build can be reused instead of recompiling."""

import inspect
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from blond.core.backends.cpp import compiled_dir_handler as lc

EXPECTED_KEYS = sorted(
    set(inspect.signature(lc.cpp_compiled_dir).parameters) - {"folder"}
)

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


class TestSaveAndLoadBuildOptions(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.folder = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_round_trip(self):
        options = _default_options(flags="-DFOO")
        lc.save_build_options(self.folder, **options)
        self.assertEqual(lc.load_build_options(self.folder), options)

    def test_options_overwrite(self):
        first_options = _default_options(flags="-DFOO")
        first_options["stale_key"] = "should not survive"
        lc.save_build_options(self.folder, **first_options)

        second_options = _default_options(flags="-DBAR")
        lc.save_build_options(self.folder, **second_options)

        loaded = lc.load_build_options(self.folder)
        self.assertDictEqual(loaded, second_options)
        self.assertNotIn("stale_key", loaded)

    def test_none_if_missing(self):
        self.assertIsNone(lc.load_build_options(self.folder))

    def test_invalid_cache(self):
        compiled_dir = os.path.join(self.folder, "compiled")
        os.makedirs(compiled_dir)
        path = os.path.join(compiled_dir, lc._BUILD_OPTIONS_NAME)

        bad_payloads = [
            "{not valid json",  # unparseable
            "[1, 2, 3]",  # valid JSON, not a dict
            '"not-a-dict"',
            "42",
            "null",
        ]
        for payload in bad_payloads:
            with open(path, "w") as file:
                file.write(payload)
            with self.assertWarns(RuntimeWarning):
                result = lc.load_build_options(self.folder)
            self.assertIsNone(result)

    def test_save_failure(self):
        # Cache bookkeeping must never break a compile: a write failure
        # (e.g. read-only filesystem) is reported via warning, not raised.
        with mock.patch("os.makedirs", side_effect=OSError("boom")):
            with self.assertWarns(RuntimeWarning):
                lc.save_build_options(self.folder, **_default_options())
        self.assertIsNone(lc.load_build_options(self.folder))


class TestCheckHelpers(unittest.TestCase):
    def test_check_build_keys(self):
        self.assertTrue(lc._check_build_keys({"a": 1, "b": 2}, ["a", "b"]))
        self.assertFalse(lc._check_build_keys({"a": 1}, ["a", "b"]))
        self.assertTrue(lc._check_build_keys({}, []))

    def test_check_compiler(self):
        self.assertTrue(lc._check_compiler(sys.executable))
        self.assertFalse(
            lc._check_compiler("definitely-not-a-real-compiler-xyz")
        )

        if _HAS_GPP:
            self.assertTrue(lc._check_compiler("g++"))
        else:
            self.skipTest("needs g++ for the bare-name PATH-lookup case")

    def test_check_lib_dirs(self):

        with tempfile.TemporaryDirectory() as tmp_dir:
            boost_dir = os.path.join(tmp_dir, "boost")
            os.makedirs(boost_dir)
            self.assertTrue(
                lc._check_lib_dirs(
                    {"boost": boost_dir, "with_fftw_lib": None},
                    ("boost", "with_fftw_lib"),
                )
            )

            self.assertFalse(
                lc._check_lib_dirs(
                    {"boost": os.path.join(tmp_dir, "does-not-exist")},
                    ("boost",),
                )
            )

        # boost="" means "use the system default", not a path to check.
        self.assertTrue(lc._check_lib_dirs({"boost": ""}, ("boost",)))
        self.assertTrue(lc._check_lib_dirs({}, ("boost", "with_fftw_lib")))

    def test_check_dry_run_compile(self):
        self.assertFalse(
            lc._check_dry_run_compile("definitely-not-a-real-compiler-xyz", [])
        )

        if _HAS_GPP:
            self.assertTrue(lc._check_dry_run_compile("g++", ["-O2", "-Wall"]))
            self.assertFalse(
                lc._check_dry_run_compile("g++", ["-this-is-not-a-real-flag"])
            )
        else:
            self.skipTest("needs g++")


class TestBuildOptionsValid(unittest.TestCase):
    def test_default_options(self):
        if not _HAS_GPP:
            self.skipTest("needs g++")
        self.assertTrue(
            lc.build_options_valid(_default_options(), EXPECTED_KEYS)
        )

    def test_missing_key(self):
        options = _default_options()
        del options["boost"]
        self.assertFalse(lc.build_options_valid(options, EXPECTED_KEYS))

    def test_invalid_compiler(self):
        options = _default_options(
            compiler="definitely-not-a-real-compiler-xyz"
        )
        self.assertFalse(lc.build_options_valid(options, EXPECTED_KEYS))

    def test_missing_boost_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = _default_options(
                boost=os.path.join(tmp_dir, "no-such-path")
            )
        self.assertFalse(lc.build_options_valid(options, EXPECTED_KEYS))

    def test_invalid_flag(self):
        if not _HAS_GPP:
            self.skipTest("needs g++")
        options = _default_options(flags="-this-is-not-a-real-flag")
        self.assertFalse(lc.build_options_valid(options, EXPECTED_KEYS))

    def test_optimize_false(self):
        if not _HAS_GPP:
            self.skipTest("needs g++")
        options = _default_options(optimize=False)
        self.assertTrue(lc.build_options_valid(options, EXPECTED_KEYS))

    def test_dry_run_flags(self):

        with mock.patch.object(
            lc, "_check_dry_run_compile", return_value=True
        ) as mocked_dry_run:
            lc.build_options_valid(
                _default_options(optimize=True, flags="-Wall"),
                EXPECTED_KEYS,
            )
            flags_with_optimize = mocked_dry_run.call_args.args[1]

            lc.build_options_valid(
                _default_options(optimize=False, flags="-Wall"),
                EXPECTED_KEYS,
            )
            flags_without_optimize = mocked_dry_run.call_args.args[1]

        self.assertIn("-march=native", flags_with_optimize)
        self.assertIn("-ffast-math", flags_with_optimize)
        self.assertNotIn("-march=native", flags_without_optimize)
        self.assertNotIn("-ffast-math", flags_without_optimize)
        self.assertEqual(flags_without_optimize, ["-Wall"])


if __name__ == "__main__":
    unittest.main()
