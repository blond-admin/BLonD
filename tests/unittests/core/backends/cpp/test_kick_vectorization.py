# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests that the C++ multi-harmonic kick vectorizes over particles.

With a run-time number of harmonics, GCC prefers to vectorize the inner
loop over the harmonics as a reduction, 4 or 8 harmonics per vector
instruction, and leaves the particle loop scalar. For the few harmonics a
real RF system has, that runs almost entirely in the reduction's scalar
remainder: 5 harmonics took 9x as long as 4 (i5-11500, 1e6 particles).
Results stay correct, so only the generated code shows it: a loop that
evaluates the sine in vector registers but never stores a vector of
particles is such a reduction.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from blond.core.backends.cpp import compile as cpp_compile
from blond.testing.backend_testing import BLonDTestCase

_HAS_GPP = shutil.which("g++") is not None
_CPP_DIR = os.path.dirname(os.path.abspath(cpp_compile.__file__))
_DEFERRED_DIR = os.path.join(os.path.dirname(_CPP_DIR), "deferred")

# Only the multi-harmonic kick is instantiated, so every loop in the
# assembly belongs to it.
_PROBE_SOURCE = """
#include "particle_kernels.h"

extern "C" void probe_kick(const KickMultiHarmonicArgs &args,
                           const real_t *beam_dt, real_t *beam_dE,
                           const index_t begin, const index_t end) {
  apply_to_chunk(args, beam_dt, beam_dE, begin, end);
}
"""

# The flags `blond-compile-cpp` builds with, at fixed CPU targets so the
# result does not depend on the host: one 256-bit, one 512-bit.
_FLAGS = (
    "-O3",
    "-std=c++11",
    "-funroll-loops",
    "-ftree-vectorize",
    "-ffast-math",
    "-fopenmp",
    "-DPARALLEL",
    "-D_USE_MATH_DEFINES",
    "-Wno-unknown-pragmas",
    "-fPIC",
)
_TARGETS = ("haswell", "cascadelake")

_PACKED_ARITHMETIC = re.compile(r"^\s*v\w+pd\s+.*%[yz]mm")
_PACKED_STORE = re.compile(r"^\s*vmov[au]pd\s+%[yz]mm\d+,\s*[^%\s]")
_SINE_RANGE_REDUCTION = re.compile(r"^\s*vcvttpd2dq[xy]?\s+%[yz]mm")


def _compile_probe_to_asm(march):
    """Assembly of the multi-harmonic kick for `-march=<march>`."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_path = os.path.join(tmp_dir, "probe.cpp")
        asm_path = os.path.join(tmp_dir, "probe.s")
        with open(source_path, "w", encoding="utf-8") as file:
            file.write(_PROBE_SOURCE)
        subprocess.run(
            [
                "g++",
                *_FLAGS,
                f"-march={march}",
                "-I",
                _CPP_DIR,
                "-I",
                _DEFERRED_DIR,
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
            return file.read().split("\n")


def _ends_function(line):
    """Whether `line` returns or starts the next function."""
    is_function_label = re.match(r"^[A-Za-z_][\w.]*:$", line) is not None
    return line.strip().startswith("ret") or is_function_label


def _loops(asm_lines):
    """Bodies of the loops: a label up to the jump back to it."""
    loops = []
    for start, line in enumerate(asm_lines):
        label = re.match(r"^(\.L\d+):", line)
        if not label:
            continue
        back_jump = re.compile(rf"^\s*j\w+\s+{re.escape(label.group(1))}$")
        for end in range(start + 1, len(asm_lines)):
            if back_jump.match(asm_lines[end]):
                loops.append(asm_lines[start + 1 : end + 1])
                break
            if _ends_function(asm_lines[end]):
                break
    return loops


def _stores_particles(loop):
    """Whether the loop stores a vector to memory other than the stack."""
    return any(
        _PACKED_STORE.match(line)
        and "(%rsp" not in line
        and "(%rbp" not in line
        for line in loop
    )


@unittest.skipUnless(_HAS_GPP, "Requires g++ to inspect generated code")
class TestKickMultiHarmonicVectorization(BLonDTestCase):
    """Every vectorized sine loop of the kick must store particles."""

    def test_sine_loops_vectorize_over_particles(self):
        """No loop reduces over harmonics instead of particles."""
        for march in _TARGETS:
            with self.subTest(march=march):
                sine_loops = [
                    loop
                    for loop in _loops(_compile_probe_to_asm(march))
                    if any(_SINE_RANGE_REDUCTION.match(x) for x in loop)
                    and any(_PACKED_ARITHMETIC.match(x) for x in loop)
                ]
                self.assertTrue(sine_loops, "no vectorized sine loop found")
                reductions = [
                    loop for loop in sine_loops if not _stores_particles(loop)
                ]
                self.assertEqual(
                    len(reductions),
                    0,
                    f"{len(reductions)} of {len(sine_loops)} vectorized "
                    "sine loops store no particles (reduction over "
                    "harmonics)",
                )
