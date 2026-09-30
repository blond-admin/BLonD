# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests the machine code of the deferred CUDA executor.

``execute_kernel_call_batch`` applies every record of a batch to a tile
of ``PARTICLES_PER_THREAD`` particles at a time. A record's loop-invariant
factors (the FP64 reciprocals of the drifts) must be computed once per
record and block, before the tile loop, not once per tile: on GPUs with a
low FP64 rate a division per record per tile dominates the cheap drift
records. Results are identical either way, only the executor gets slower,
so nothing else catches it. This test therefore reads the SASS.
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
_NVCC_PATH = shutil.which(_NVCC)
_CUOBJDUMP = (
    os.path.join(os.path.dirname(_NVCC_PATH), "cuobjdump")
    if _NVCC_PATH is not None
    else None
)
_HAS_TOOLS = _CUOBJDUMP is not None and os.path.exists(_CUOBJDUMP)
_CUDA_DIR = os.path.dirname(os.path.abspath(cuda_compile.__file__))
_DEFERRED_DIR = os.path.join(os.path.dirname(_CUDA_DIR), "deferred")
_KERNELS_CU = os.path.join(_CUDA_DIR, "kernels.cu")

# The target the codegen was measured on (T400); nvcc compiles for it
# without a device.
_ARCH = "sm_75"

_INSTRUCTION_PATTERN = re.compile(
    r"/\*(?P<address>[0-9a-f]{4,})\*/\s+(?P<text>[^;]*);"
)
_CALL_TARGET_PATTERN = re.compile(
    r"CALL\.REL\.NOINC\s+0x(?P<target>[0-9a-f]+)"
)


def _executor_sass() -> list[tuple[int, str]]:
    """Return ``(address, instruction)`` of the executor, in order."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cubin = os.path.join(tmp_dir, "kernels.cubin")
        subprocess.run(
            [
                _NVCC,
                *cuda_compile.NVCC_FLAGS,
                "-arch",
                _ARCH,
                "-I" + _DEFERRED_DIR,
                "-o",
                cubin,
                _KERNELS_CU,
            ],
            check=True,
            capture_output=True,
        )
        sass = subprocess.run(
            [_CUOBJDUMP, "-sass", "-fun", "execute_kernel_call_batch", cubin],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return [
        (int(match["address"], 16), match["text"].strip())
        for match in _INSTRUCTION_PATTERN.finditer(sass)
    ]


def _particles_per_thread() -> int | None:
    """Return ``PARTICLES_PER_THREAD`` of kernels.cu, None if absent."""
    with open(_KERNELS_CU) as file:
        match = re.search(
            r"constexpr int PARTICLES_PER_THREAD = (\d+);", file.read()
        )
    return None if match is None else int(match[1])


@pytest.mark.cupy
@unittest.skipUnless(_HAS_TOOLS, "Requires nvcc and cuobjdump")
class TestDeferredExecutorCodegen(BLonDTestCase):
    """The executor's tile loop holds only per-particle divisions."""

    def test_record_factors_are_computed_before_the_tile_loop(self):
        """No per-record reciprocal is left after the last barrier.

        Everything after the executor's last ``BAR.SYNC`` is the tile
        loop. Its only legitimate FP64 division is drift_exact's
        per-particle ``/ (1 + delta)``, once per particle of a tile. The
        reciprocals of the drift factors (``T eta_0 / (beta^2 E)``,
        ``1 / beta^2``, ``1 / E``) belong before that barrier. Division
        slow paths are subroutines after the kernel body (CALL targets)
        and are not counted.
        """
        particles_per_thread = _particles_per_thread()
        self.assertIsNotNone(particles_per_thread)
        instructions = _executor_sass()
        call_targets = [
            int(match["target"], 16)
            for _, text in instructions
            if (match := _CALL_TARGET_PATTERN.search(text))
        ]
        body_end = min(call_targets, default=instructions[-1][0] + 1)
        body = [text for address, text in instructions if address < body_end]
        last_barrier = max(
            i for i, text in enumerate(body) if text.startswith("BAR.SYNC")
        )
        tile_loop_reciprocals = sum(
            "MUFU.RCP64H" in text for text in body[last_barrier + 1 :]
        )
        self.assertLessEqual(
            tile_loop_reciprocals,
            particles_per_thread,
            "per-record FP64 divisions are computed inside the tile loop",
        )
