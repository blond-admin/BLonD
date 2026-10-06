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
of ``PARTICLES_PER_THREAD`` particles at a time (the beam's tail to tiles
of its halvings). A record's loop-invariant
factors (the FP64 reciprocals of the drifts) must be computed once per
record and block, before the tile loop, not once per tile: on GPUs with a
low FP64 rate a division per record per tile dominates the cheap drift
records. Results are identical either way, only the executor gets slower,
so nothing else catches it. This test therefore reads the SASS.

The same goes for the FP64 arithmetic of each record: on those GPUs the
fused batch is bound by it, so whatever a record can do once per block
instead of once per particle is checked here too, on probe kernels that
apply a single record type to a tile the way the executor does.
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
# A load of a beam coordinate (dt or dE), not of a constant.
_COORDINATE_LOAD_PATTERN = re.compile(r"\bLDG\.E\.64(?!\.CONSTANT)\b")


# Width of the probe tiles, a width the executor uses.
_PROBE_TILE = 8

# One kernel per record type: the record staged in shared memory and
# prepared by one thread behind a barrier, then applied to a tile, as in
# `execute_kernel_call_batch` but without the switch over the record
# types, so the SASS after the last barrier is that record's alone.
_PROBE_SOURCE = f"""
#include "kernels.cu"

template <class Args>
__device__ __forceinline__ void probe(const KernelCallBatch &batch,
                                      real_t *beam_dt, real_t *beam_dE) {{
  __shared__ KernelCallBatch staged;
  __shared__ RecordFactors factors[1];
  for (int j = threadIdx.x; j < KERNEL_CALL_BATCH_CAPACITY_BYTES / 8;
       j += blockDim.x) {{
    staged.slots[j] = batch.slots[j];
  }}
  __syncthreads();
  const auto *record = reinterpret_cast<const KernelCallHeader *>(
      staged.slots);
  if (threadIdx.x == 0) {{
    PrepareRecord{{&factors[0]}}(record_args<Args>(record));
  }}
  __syncthreads();
  real_t dt[{_PROBE_TILE}];
  real_t dE[{_PROBE_TILE}];
  const index_t start = particle_loop_start();
  const index_t stride = particle_loop_stride();
#pragma unroll
  for (int k = 0; k < {_PROBE_TILE}; ++k) {{
    dt[k] = beam_dt[start + k * stride];
    dE[k] = beam_dE[start + k * stride];
  }}
  ApplyToParticleTile<{_PROBE_TILE}>{{&dt, &dE, &factors[0]}}(
      record_args<Args>(record));
#pragma unroll
  for (int k = 0; k < {_PROBE_TILE}; ++k) {{
    beam_dt[start + k * stride] = dt[k];
    beam_dE[start + k * stride] = dE[k];
  }}
}}

#define PROBE(ARGS)                                                    \\
  extern "C" __global__ void __launch_bounds__(EXECUTOR_BLOCK_SIZE,    \\
                                               EXECUTOR_BLOCKS_PER_SM) \\
      probe_##ARGS(const KernelCallBatch batch, real_t *beam_dt,       \\
                   real_t *beam_dE) {{                                  \\
    probe<ARGS>(batch, beam_dt, beam_dE);                              \\
  }}
PROBE(KickSingleHarmonicArgs)
PROBE(KickMultiHarmonicArgs)
PROBE(DriftLikeLineSegmentArgs)

// A bare double reciprocal square root per particle, as a yardstick.
extern "C" __global__ void probe_rsqrt(real_t *beam_dt, real_t *beam_dE) {{
  __syncthreads();
  const index_t start = particle_loop_start();
  const index_t stride = particle_loop_stride();
#pragma unroll
  for (int k = 0; k < {_PROBE_TILE}; ++k) {{
    beam_dt[start + k * stride] = rsqrt(beam_dE[start + k * stride]);
  }}
}}
"""


def _sass(source: str, function: str) -> list[tuple[int, str]]:
    """Return ``(address, instruction)`` of `function`, in order.

    `source` is compiled with the flags of the library build, with the
    CUDA backend's directory on the include path.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cubin = os.path.join(tmp_dir, "kernels.cubin")
        subprocess.run(
            [
                _NVCC,
                *cuda_compile.NVCC_FLAGS,
                "-arch",
                _ARCH,
                "-I" + _DEFERRED_DIR,
                "-I" + _CUDA_DIR,
                "-o",
                cubin,
                source,
            ],
            check=True,
            capture_output=True,
        )
        sass = subprocess.run(
            [_CUOBJDUMP, "-sass", "-fun", function, cubin],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return [
        (int(match["address"], 16), match["text"].strip())
        for match in _INSTRUCTION_PATTERN.finditer(sass)
    ]


def _executor_sass() -> list[tuple[int, str]]:
    """Return ``(address, instruction)`` of the executor, in order."""
    return _sass(_KERNELS_CU, "execute_kernel_call_batch")


def _probe_tile_loop(args_type: str) -> list[str]:
    """Return the tile loop of the probe kernel of record `args_type`."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "probe.cu")
        with open(source, "w") as file:
            file.write(_PROBE_SOURCE)
        return _tile_loop(_sass(source, f"probe_{args_type}"))


def _fp64_arithmetic(instructions: list[str]) -> int:
    """Return how many of `instructions` are FP64 adds and multiplies."""
    return sum(_count(instructions, op) for op in ("DADD", "DMUL", "DFMA"))


def _count(instructions: list[str], opcode: str) -> int:
    """Return how many of `instructions` have the opcode `opcode`.

    Predicated instructions (``@P0 DMUL ...``) count too; modifiers
    after a dot (``MUFU.RSQ64H``) are part of the opcode only if
    `opcode` names them.
    """
    count = 0
    for text in instructions:
        mnemonic = re.sub(r"^@!?U?P\w+\s+", "", text).split()[0]
        if mnemonic == opcode or mnemonic.startswith(opcode + "."):
            count += 1
    return count


def _tile_loop(
    instructions: list[tuple[int, str]], barriers_after: int = 0
) -> list[str]:
    """Return the instructions after the barrier before the tile loop.

    That is the last barrier of a probe kernel. The executor has one more
    after its tile loop (``barriers_after=1``), before it adds the block's
    histogram counts to the profile. Division slow paths are subroutines
    after the kernel body (CALL targets) and are left out.
    """
    call_targets = [
        int(match["target"], 16)
        for _, text in instructions
        if (match := _CALL_TARGET_PATTERN.search(text))
    ]
    body_end = min(call_targets, default=instructions[-1][0] + 1)
    body = [text for address, text in instructions if address < body_end]
    barriers = [
        i for i, text in enumerate(body) if text.startswith("BAR.SYNC")
    ]
    start = barriers[-1 - barriers_after] + 1
    end = barriers[-barriers_after] if barriers_after else len(body)
    return body[start:end]


def _tile_widths(particles_per_thread: int) -> list[int]:
    """Widths of the executor's tiles: the full one, then the halvings."""
    widths = [particles_per_thread]
    while widths[-1] > 1:
        widths.append(widths[-1] // 2)
    return widths


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

    def test_no_fp64_division_in_the_tile_loop(self):
        """No FP64 reciprocal is left after the last barrier.

        Everything after the executor's last ``BAR.SYNC`` is the tile
        loops. The reciprocals of the drift factors (``T eta_0 / (beta^2
        E)``, ``1 / beta^2``, ``1 / E``) belong before that barrier, and
        drift_exact divides by ``1 + delta`` as a multiplication with the
        reciprocal square root it needs anyway. Division slow paths are
        subroutines after the kernel body (CALL targets) and are not
        counted.
        """
        tile_loop_reciprocals = sum(
            "MUFU.RCP64H" in text
            for text in _tile_loop(_executor_sass(), barriers_after=1)
        )
        self.assertEqual(
            tile_loop_reciprocals,
            0,
            "FP64 divisions are computed inside the tile loop",
        )

    def test_beam_tail_is_covered_by_narrower_tiles(self):
        """Every halving of the tile has its own tile loop.

        The beam past the last whole sweep of full tiles covers fewer
        particles per thread than a full tile. Covering it with tiles of
        half, a quarter, ... of the width (as many as its size needs)
        makes each thread apply the batch to the ceiling of
        ``n_macroparticles / n_threads`` particles, where full tiles
        would round that up to a multiple of ``PARTICLES_PER_THREAD``.
        Each tile loads the two coordinates of each of its particles, so
        the tile loops load ``2 * sum(widths)`` coordinates.
        """
        particles_per_thread = _particles_per_thread()
        self.assertIsNotNone(particles_per_thread)
        coordinate_loads = sum(
            bool(_COORDINATE_LOAD_PATTERN.search(text))
            for text in _tile_loop(_executor_sass(), barriers_after=1)
        )
        self.assertEqual(
            coordinate_loads, 2 * sum(_tile_widths(particles_per_thread))
        )


@pytest.mark.cupy
@unittest.skipUnless(_HAS_TOOLS, "Requires nvcc and cuobjdump")
class TestDeferredRecordArithmetic(BLonDTestCase):
    """Per-particle FP64 arithmetic of each record in the fused tile loop.

    Counted on probe kernels (see `_PROBE_SOURCE`). Every inlined double
    ``sin`` has exactly one call to its slow-path argument reduction, so
    the calls count the sines of a kick.
    """

    def test_multi_harmonic_kick_folds_charge_into_voltages(self):
        """A harmonic multiplies no more than a single-harmonic kick.

        Both kicks cost a ``sin`` plus a fused multiply-add per harmonic.
        The single-harmonic kick's ``charge * voltage`` is the same for
        the whole tile. The multi-harmonic kick's ``charge * voltage[j]``
        is too, but, inside the harmonic loop, it used to be recomputed
        for every particle and harmonic: a DMUL per ``sin`` more. The
        executor folds the charge into the staged voltages once per
        block instead.
        """
        single = _probe_tile_loop("KickSingleHarmonicArgs")
        multi = _probe_tile_loop("KickMultiHarmonicArgs")
        # one float-to-int conversion (the quadrant) per `fast_sin`
        single_sines = _count(single, "F2I")
        multi_sines = _count(multi, "F2I")
        self.assertEqual(single_sines, _PROBE_TILE)
        self.assertGreater(multi_sines, 0)
        self.assertLessEqual(
            _count(multi, "DMUL") / multi_sines,
            _count(single, "DMUL") / single_sines,
        )

    def test_line_segment_drift_costs_an_rsqrt_and_four_operations(self):
        """drift_like_line_segment: an ``rsqrt`` plus 4 FP64 operations.

        ``delta = sqrt(1 + (dE^2 / E^2 + 2 dE / E) / beta^2) - 1`` is,
        with the prepared factors ``a = 1 / (beta E)^2`` and
        ``b = 2 / (beta^2 E)`` and ``d = fma(dE, fma(a, dE, b), 1)``,
        ``fma(d, rsqrt(d), -1)``: two FMAs, the reciprocal square root
        and one more FMA; the drift is another FMA. Written as in the
        formula, it took five FP64 operations before a full-precision
        square root (three more than ``rsqrt``) and a subtraction.
        """
        drift = _probe_tile_loop("DriftLikeLineSegmentArgs")
        yardstick = _probe_tile_loop("rsqrt")
        self.assertLessEqual(
            _fp64_arithmetic(drift) / _PROBE_TILE,
            _fp64_arithmetic(yardstick) / _PROBE_TILE + 4,
        )
