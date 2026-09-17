# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests that the per-kernel 512-bit vector width macro really widens code.

``BLOND_PREFER_VECTOR_WIDTH_512`` is easy to get wrong in a way that looks
exactly like success: GCC accepts ``__attribute__((target(...)))`` in the
wrong position with only a warning and then silently emits 256-bit code,
and a plain ``-mavx2`` alongside ``-march=native`` changes nothing at all.
These tests therefore check the emitted registers, not just that the build
succeeded.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from blond.core.backends.cpp import compile as cpp_compile
from blond.testing.backend_testing import BLonDTestCase

_HAS_GPP = shutil.which("g++") is not None
_CPP_DIR = os.path.dirname(os.path.abspath(cpp_compile.__file__))

# A loop that GCC vectorizes readily, so the only thing under test is the
# width it picks.
_PROBE_SOURCE = """
#include "blond_common.h"

extern "C" MACRO_PLACEHOLDER void probe_loop(
    real_t *__restrict__ out, const real_t *__restrict__ in,
    const index_t n_elements) {
  for (index_t i = 0; i < n_elements; i++) {
    out[i] += in[i] * in[i] + 1.0;
  }
}
"""


def _host_supports_avx512():
    """Whether ``-march=native`` on this host enables AVX-512."""
    if not _HAS_GPP:
        return False
    proc = subprocess.run(
        ["g++", "-march=native", "-dM", "-E", "-x", "c++", "-"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return "__AVX512F__" in proc.stdout


_HAS_AVX512 = _host_supports_avx512()


def _compile_probe_to_asm(macro, extra_flags=()):
    """Compile the probe with `macro` in front of it, return its assembly."""
    source = _PROBE_SOURCE.replace("MACRO_PLACEHOLDER", macro)
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_path = os.path.join(tmp_dir, "probe.cpp")
        asm_path = os.path.join(tmp_dir, "probe.s")
        with open(source_path, "w", encoding="utf-8") as file:
            file.write(source)
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++11",
                "-march=native",
                "-ffast-math",
                "-ftree-vectorize",
                "-Werror=attributes",
                *extra_flags,
                "-I",
                _CPP_DIR,
                "-S",
                "-o",
                asm_path,
                source_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(asm_path, encoding="utf-8") as file:
            return file.read()


@unittest.skipUnless(_HAS_GPP, "Requires g++ to inspect generated code")
class TestPreferVectorWidthMacro(BLonDTestCase):
    """The macro must widen vectorization, and must do so only where used."""

    def test_macro_compiles_on_every_host(self):
        """The macro is usable even where the CPU has no AVX-512."""
        asm = _compile_probe_to_asm("BLOND_PREFER_VECTOR_WIDTH_512")
        self.assertIn("probe_loop", asm)

    @unittest.skipUnless(_HAS_AVX512, "Host CPU has no AVX-512")
    def test_macro_emits_512_bit_registers(self):
        """With the macro, the loop uses zmm registers."""
        asm = _compile_probe_to_asm("BLOND_PREFER_VECTOR_WIDTH_512")
        self.assertIn("%zmm", asm)

    @unittest.skipUnless(_HAS_AVX512, "Host CPU has no AVX-512")
    def test_without_macro_stays_256_bit(self):
        """Without the macro, GCC's default keeps the loop at 256 bits.

        This is the control: without it, a test asserting ``%zmm`` would
        still pass if every kernel were widened globally.
        """
        asm = _compile_probe_to_asm("")
        self.assertNotIn("%zmm", asm)

    @unittest.skipUnless(_HAS_AVX512, "Host CPU has no AVX-512")
    def test_macro_can_be_disabled_from_the_command_line(self):
        """Defining the macro empty switches the whole feature off.

        This is the escape hatch for a CPU where AVX-512 downclocking makes
        wider vectors a loss, and the screening script depends on it to
        compile an already-widened kernel at 256 bits.
        """
        asm = _compile_probe_to_asm(
            "BLOND_PREFER_VECTOR_WIDTH_512",
            extra_flags=("-DBLOND_PREFER_VECTOR_WIDTH_512=",),
        )
        self.assertNotIn("%zmm", asm)
