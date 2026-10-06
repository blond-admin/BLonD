# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""BLonD 2's compile.py, as far as BLonD 3 runs it (run_comparison.py)."""

import platform
import shutil
import unittest

from blond.legacy.blond2 import compile as blond2_compile
from blond.testing.backend_testing import BLonDTestCase


class TestNativeVectorizationFlags(BLonDTestCase):
    @unittest.skipUnless(
        shutil.which("g++") and platform.machine() in ("x86_64", "AMD64"),
        "needs g++ on x86",
    )
    def test_flags_of_the_local_cpu(self):
        # --optimize asks the compiler which SIMD extensions -march=native
        # enables; every x86-64 CPU has at least SSE2.
        flags = blond2_compile.native_vectorization_flags("g++")
        self.assertTrue(
            any(flag.startswith(("-msse", "-mavx")) for flag in flags), flags
        )

    def test_missing_compiler_gives_no_flags(self):
        self.assertEqual(
            blond2_compile.native_vectorization_flags("no-such-compiler"), []
        )
