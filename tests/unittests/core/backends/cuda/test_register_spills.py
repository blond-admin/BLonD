# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests that the eager CUDA kernels do not spill registers to local memory.

The library is built with ``-maxrregcount 32``, so a small change in how a
kernel holds its parameters (a per-thread struct copy, a shared-memory
staging with a barrier) can push its per-particle loop into local-memory
spills. Results stay identical, only the kernel gets slower, so nothing
else catches it. These tests therefore read ``ptxas -v``'s resource usage
for the kernels in question.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

import pytest

from blond.core.backends.cuda import compile as cuda_compile
from blond.core.backends.cuda.compiled_dir_handler import resolve_nvcc
from blond.testing.backend_testing import BLonDTestCase

_NVCC = resolve_nvcc()
_HAS_NVCC = shutil.which(_NVCC) is not None
_CUDA_DIR = os.path.dirname(os.path.abspath(cuda_compile.__file__))
_DEFERRED_DIR = os.path.join(os.path.dirname(_CUDA_DIR), "deferred")

# Register allocation depends on the target, so pin the architecture the
# limits below were measured on (T400) instead of whatever GPU is present;
# nvcc compiles for it without a device.
_ARCH = "sm_75"

_PROPERTIES_PATTERN = re.compile(
    r"Function properties for (?P<name>\w+)\s*\n"
    r"\s*(?P<stack>\d+) bytes stack frame, "
    r"(?P<spill_stores>\d+) bytes spill stores, "
    r"(?P<spill_loads>\d+) bytes spill loads\s*\n"
    r"[^\n]*Used (?P<registers>\d+) registers"
)

# The GPUs the deferred executor targets: V100 (sm_70), T4/T400 (sm_75),
# A100 (sm_80), H100/H200 (sm_90). All have 64 Ki registers per SM.
_EXECUTOR_ARCHS = ("sm_70", "sm_75", "sm_80", "sm_90")
_REGISTERS_PER_SM = 64 * 1024
# `_deferred_block_size` and the blocks per SM of `grid_size` in
# cuda/callables.py (`default_blocks`).
_EXECUTOR_BLOCK_SIZE = 256
_EXECUTOR_BLOCKS_PER_SM = 2
# Spill bytes (stores, loads) tolerated per target for nvcc older than
# 12.9. nvcc 12.8 builds the sm_90 executor at the 128-register cap with
# one 4-byte value spilled: stored once at kernel start, reloaded once per
# record of a tile (each then applied to a whole tile of particles), so
# it costs next to nothing. nvcc 12.9 fits the same code in 124 registers.
_SPILL_TOLERANCE_BEFORE_NVCC_12_9 = {"sm_90": (4, 16)}


def _nvcc_version() -> tuple[int, int]:
    """The ``(major, minor)`` release of the installed nvcc."""
    output = subprocess.run(
        [_NVCC, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"release (\d+)\.(\d+)", output)
    return int(match[1]), int(match[2])


def _executor_spill_tolerance(arch: str) -> tuple[int, int]:
    """The (stores, loads) spill bytes the executor may have on `arch`."""
    if _nvcc_version() >= (12, 9):
        return 0, 0
    return _SPILL_TOLERANCE_BEFORE_NVCC_12_9.get(arch, (0, 0))


def _nvcc_archs() -> set[str]:
    """The ``sm_XX`` targets the installed nvcc can compile for."""
    listed = subprocess.run(
        [_NVCC, "--list-gpu-arch"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return {arch.replace("compute_", "sm_") for arch in listed}


def _ptxas_resource_usage(arch=_ARCH):
    """Compile kernels.cu verbosely, return usage per kernel name."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        proc = subprocess.run(
            [
                _NVCC,
                *cuda_compile.NVCC_FLAGS,
                "-arch",
                arch,
                "-Xptxas",
                "-v",
                "-I" + _DEFERRED_DIR,
                "-o",
                os.path.join(tmp_dir, "kernels.cubin"),
                os.path.join(_CUDA_DIR, "kernels.cu"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    usage = {}
    for match in _PROPERTIES_PATTERN.finditer(proc.stderr + proc.stdout):
        usage[match["name"]] = {
            key: int(match[key])
            for key in ("stack", "spill_stores", "spill_loads", "registers")
        }
    return usage


@pytest.mark.cupy
@unittest.skipUnless(_HAS_NVCC, "Requires nvcc to inspect generated code")
class TestEagerKernelRegisterSpills(BLonDTestCase):
    """The eager kernels keep their per-particle loop in registers."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.usage = _ptxas_resource_usage()

    def test_kick_multi_harmonic_does_not_spill(self):
        """Reading the RF batch in place keeps the harmonic loop spill-free.

        Staging the batch in shared memory behind a barrier made the loop
        spill under the 32-register cap.
        """
        usage = self.usage["kick_multi_harmonic"]
        self.assertEqual(usage["spill_stores"], 0, usage)
        self.assertEqual(usage["spill_loads"], 0, usage)

    def test_drift_exact_has_no_stack_frame(self):
        """The by-value alphas stay constant-bank operands.

        NVVM copies a small by-value parameter that is indexed dynamically
        into local memory; the unrolled Horner loop indexes it with
        compile-time indices instead.
        """
        usage = self.usage["drift_exact"]
        self.assertEqual(usage["stack"], 0, usage)
        self.assertEqual(usage["spill_stores"], 0, usage)
        self.assertEqual(usage["spill_loads"], 0, usage)

    def test_drift_exact_global_alphas_keeps_small_stack_frame(self):
        """The alpha coefficients are read in place from global memory.

        A per-thread copy of them (a DriftExactArgs record) is dynamically
        indexed, so it lives in local memory and the stack frame grows.
        32 bytes is what the kernel used before that copy was introduced.
        """
        usage = self.usage["drift_exact_global_alphas"]
        self.assertLessEqual(usage["stack"], 32, usage)


@pytest.mark.cupy
@unittest.skipUnless(_HAS_NVCC, "Requires nvcc to inspect generated code")
class TestDeferredExecutorRegisters(BLonDTestCase):
    """The deferred executor fits its launch in registers on every target.

    ``execute_kernel_call_batch`` keeps a tile of particles per thread in
    registers (every tile width is inlined into the one kernel), and is
    launched with two blocks per SM, which must be resident together.
    """

    def test_no_spills_and_two_blocks_per_sm(self):
        """No local-memory spills; the registers of two blocks fit an SM.

        Except the small spill nvcc < 12.9 leaves on sm_90, see
        `_SPILL_TOLERANCE_BEFORE_NVCC_12_9`.
        """
        available = _nvcc_archs()
        for arch in _EXECUTOR_ARCHS:
            with self.subTest(arch=arch):
                if arch not in available:
                    self.skipTest(f"nvcc cannot compile for {arch}")
                usage = _ptxas_resource_usage(arch)[
                    "execute_kernel_call_batch"
                ]
                max_stores, max_loads = _executor_spill_tolerance(arch)
                self.assertLessEqual(usage["spill_stores"], max_stores, usage)
                self.assertLessEqual(usage["spill_loads"], max_loads, usage)
                self.assertLessEqual(
                    usage["registers"]
                    * _EXECUTOR_BLOCK_SIZE
                    * _EXECUTOR_BLOCKS_PER_SM,
                    _REGISTERS_PER_SM,
                    usage,
                )
