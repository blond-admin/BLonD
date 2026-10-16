# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CudaSpecials` and helper functions."""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np
from cupy_backends.cuda.libs import cublas  # type: ignore

from blond.core.backends.backend import INDEX_DTYPE, Specials
from blond.core.backends.cuda.compiled_dir_handler import cuda_compiled_dir
from blond.core.backends.flags import BeamFlags
from blond.generals.compiled_cache import mark_used

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

_filepath = os.path.realpath(__file__)
_compute_capability = cp.cuda.Device(0).compute_capability

FLOAT = np.float64

folder = os.path.dirname(os.path.abspath(__file__))

# Same toolchain-aware directory the compiler writes to.
_basepath = str(cuda_compiled_dir(folder))


path = os.path.join(
    _basepath,
    f"kernels_sm_{_compute_capability}_double.cubin",
)
if not os.path.isfile(path):
    from blond.core.backends.cuda.compile import compile_cuda_library

    print("CUDA backend was not found.. Trying to compile.")
    compile_cuda_library()

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"The compiled CUDA backend was not found at {path=}.\n"
            f"Has the backend been compiled?"
            f"{__file__.replace('callables.py', 'compile.py')}:1"  # :1 to
            # make PyCharm automatically link the correct file
        )
gpu_module = cp.RawModule(
    path=path,
)
# Refresh the LRU stamp on the cache dir we loaded from.
mark_used(_basepath)

_drift_simple = gpu_module.get_function("drift_simple")
_drift_like_line_segment = gpu_module.get_function("drift_like_line_segment")
_drift_exact = gpu_module.get_function("drift_exact")
_beam_phase = gpu_module.get_function("beam_phase")
_kick_multi_harmonic = gpu_module.get_function("kick_multi_harmonic")
_kick_single_harmonic = gpu_module.get_function("kick_single_harmonic")
_sm_histogram = gpu_module.get_function("sm_histogram")
_hybrid_histogram = gpu_module.get_function("hybrid_histogram")
_gm_linear_interp_kick_help = gpu_module.get_function("lik_only_gm_copy")
_gm_linear_interp_kick_comp = gpu_module.get_function("lik_only_gm_comp")
_gm_linear_interp_kick_sparse_help = gpu_module.get_function(
    "lik_sparse_gm_copy"
)
_gm_linear_interp_kick_sparse_comp = gpu_module.get_function(
    "lik_sparse_gm_comp"
)
_loss_box = gpu_module.get_function("loss_box")
_histogram_sparse = gpu_module.get_function("histogram_sparse")
_pole_residue_chunk_maps = gpu_module.get_function("pole_residue_chunk_maps")
_wake_from_pole_residue = gpu_module.get_function("wake_from_pole_residue")
_twc_scan_chunks = gpu_module.get_function("twc_scan_chunks")
_twc_scan_totals = gpu_module.get_function("twc_scan_totals")
_wake_from_twc_fir = gpu_module.get_function("wake_from_twc_fir")
_pole_gemm_scan_local = gpu_module.get_function("pole_gemm_scan_local")
_pole_gemm_scan_groups = gpu_module.get_function("pole_gemm_scan_groups")
_pole_gemm_scan_fix = gpu_module.get_function("pole_gemm_scan_fix")
_pole_gemm_prep = gpu_module.get_function("pole_gemm_prep")
_pole_gemm_readout = gpu_module.get_function("pole_gemm_readout")
_pole_gemm_handover = gpu_module.get_function("pole_gemm_handover")
_pole_near_field = gpu_module.get_function("pole_near_field")
_apply_sr_without_quantum_excitation = gpu_module.get_function(
    "apply_sr_without_quantum_excitation"
)
_apply_sr_with_quantum_excitation = gpu_module.get_function(
    "apply_sr_with_quantum_excitation"
)

default_blocks = 2 * cp.cuda.Device(0).attributes["MultiProcessorCount"]
default_threads = cp.cuda.Device(0).attributes["MaxThreadsPerBlock"]
max_shared_memory_per_block = cp.cuda.Device(0).attributes[
    "MaxSharedMemoryPerBlock"
]
blocks = int(os.environ.get("GPU_BLOCKS", default_blocks))
threads = int(os.environ.get("GPU_THREADS", default_threads))
grid_size = (blocks, 1, 1)
block_size = (threads, 1, 1)
# Bytes per bin of the shared-memory histogram counters (`block_hist` in
# kernels.cu), which are `int` wide -- deliberately *not* `index_t`: a
# 64-bit counter would halve the number of bins that fit in shared memory
# for no benefit, since CUDA has no signed 64-bit `atomicAdd` anyway.
_HIST_COUNT_ITEMSIZE = np.dtype(np.int32).itemsize
# Per-harmonic RF parameters are passed to `kick_multi_harmonic` by value,
# as a struct in the kernel's parameter space (`RFParamsBatch` in
# kernels.cu): no host-to-device copy per turn. Both must match their
# counterparts in kernels.cu.
# The 32 is unrelated to the warp size: every thread loops over all
# harmonics of the batch. It is bounded by the kernel parameter limit
# (4 KiB before CUDA 12.1 / Volta), which would fit ~160 harmonics at
# 64 bit; 32 keeps the struct small (768 B) while still covering any
# realistic RF system in a single launch.
MAX_RF_HARMONICS_PER_LAUNCH = 32
_RF_PARAMS_BATCH_DTYPE = np.dtype(
    [
        ("voltage", FLOAT, (MAX_RF_HARMONICS_PER_LAUNCH,)),
        ("omega_rf", FLOAT, (MAX_RF_HARMONICS_PER_LAUNCH,)),
        ("phi_rf", FLOAT, (MAX_RF_HARMONICS_PER_LAUNCH,)),
    ]
)
_quantum_excitation_seed_counter = itertools.count(time.time_ns())


# Shape of the pole kernels; must match POLE_TILE, POLE_BLOCK and
# POLE_THREADS in kernels.cu.
_POLE_TILE = 64  # bins per tile, at most
_POLE_BLOCK = 16  # poles per block
_POLE_THREADS = 64  # threads per block
# Blocks per multiprocessor the pole kernels aim for (tuned on a T400).
_POLE_BLOCKS_PER_MULTIPROCESSOR = 16
# Fewest tiles per time chunk.
_MIN_POLE_CHUNK_TILES = 4
_N_MULTIPROCESSORS = cp.cuda.Device(0).attributes["MultiProcessorCount"]


def _pole_chunk_count(
    n_bins: int, n_poles: int, n_multiprocessors: int
) -> int:
    """
    Return into how many time chunks the pole kernels split the bins.

    A block holds `_POLE_BLOCK` poles; too few blocks leave the GPU idle,
    so the bins are split into chunks run in parallel, until there are
    enough. Chunks keep at least `_MIN_POLE_CHUNK_TILES` tiles, and stay
    fewer than the square root of the tiles: each chunk first folds the
    maps of all earlier chunks.

    Parameters
    ----------
    n_bins
        Number of bins of the profile.
    n_poles
        Number of poles.
    n_multiprocessors
        Number of streaming multiprocessors of the GPU.

    Returns
    -------
    n_chunks
        Number of chunks, at least 1.
    """
    pole_blocks = -(-n_poles // _POLE_BLOCK)
    target_blocks = _POLE_BLOCKS_PER_MULTIPROCESSOR * n_multiprocessors
    if pole_blocks >= target_blocks:
        return 1
    n_chunks = min(
        -(-target_blocks // pole_blocks),
        n_bins // (_MIN_POLE_CHUNK_TILES * _POLE_TILE),
        math.isqrt(n_bins // _POLE_TILE),
    )
    return max(n_chunks, 1)


def _wake_from_pole_residue_in_chunks(
    bin_offsets: CupyArray,
    profile: CupyArray,
    carried_charge: CupyArray,
    carried_is_counterrotating: bool,
    state_lag_dt: float,
    carried_lag_dt: float,
    poles: CupyArray,
    residues: CupyArray,
    is_counterrotating_beam: bool,
    counterrotating_pole_signs: CupyArray,
    factor: float,
    bin_dt: float,
    states: CupyArray,
    voltage: CupyArray,
    n_chunks: int,
) -> None:
    """
    Launch the pole-residue kernels with the bins in `n_chunks` chunks.

    See `Specials.wake_from_pole_residue` for every other parameter and
    kernels.cu for the scheme. `voltage` is overwritten.

    Parameters
    ----------
    bin_offsets
        Position of each bin in whole bins from the first bin, of type
        `INDEX_DTYPE`; replaces ``profile_time``.
    n_chunks
        Number of time chunks, run in parallel; 1 walks all bins in one
        pass.
    """
    n_bins = int(profile.shape[0])
    n_poles = int(poles.shape[0])
    chunk_length = -(-n_bins // n_chunks)
    n_chunks = -(-n_bins // chunk_length)

    # Each block adds its poles' read-outs via atomicAdd; must start at zero.
    voltage.fill(0)
    # The poles' wake within a tile, summed over all poles, at lags below
    # a tile: Re sum_p rho_p pair_p d_p^lag (the counter-rotation flips of
    # injection and read-out cancel). Contiguous: `.real` of a complex
    # array is a strided view, and the kernel reads the raw buffer.
    pair_factors = cp.where(poles.imag == 0, 1.0, 2.0)
    lags = cp.arange(_POLE_TILE, dtype=FLOAT)
    wake = cp.ascontiguousarray(
        (
            (residues * pair_factors)[None, :]
            * cp.exp(lags[:, None] * (poles * bin_dt)[None, :])
        )
        .sum(axis=1)
        .real
    )
    # From CuPy's memory pool, so no allocation after the first call.
    chunk_maps = cp.empty(max(n_chunks - 1, 1) * n_poles * 4, dtype=FLOAT)

    block = (_POLE_THREADS, 1, 1)
    grid = (-(-n_poles // _POLE_BLOCK), n_chunks, 1)
    # Both kernels start with these arguments; see kernels.cu. Complex
    # arrays are viewed as interleaved real/imag float arrays, no copy.
    shared_args = (
        bin_offsets,
        profile,
        poles.view(FLOAT),
        residues.view(FLOAT),
        counterrotating_pole_signs,
        np.bool_(is_counterrotating_beam),
        np.bool_(carried_is_counterrotating),
        carried_charge,
        FLOAT(carried_lag_dt),
        FLOAT(factor),
        FLOAT(bin_dt),
        np.int32(n_bins),
        np.int32(n_poles),
        np.int32(chunk_length),
        np.int32(n_chunks),
        chunk_maps,
    )
    if n_chunks > 1:
        _pole_residue_chunk_maps(args=shared_args, block=block, grid=grid)
    _wake_from_pole_residue(
        args=(
            *shared_args,
            wake,
            FLOAT(state_lag_dt),
            states.view(FLOAT),
            voltage,
        ),
        block=block,
        grid=grid,
    )


# Threads per block of the TWC kernels, TWC_THREADS in kernels.cu.
_TWC_THREADS = 256
# Bins per chunk of the TWC prefix scan, TWC_CHUNK in kernels.cu.
_TWC_SCAN_CHUNK = 4 * _TWC_THREADS


# Tile lengths of the GEMM far field, longest first.
_GEMM_TILE_LENGTHS = (128, 64, 32, 16)
# Fewest poles for which the GEMM far field beats `wake_from_pole_residue`
# (measured on a T400, between 10 and 20 poles; below, the GEMM's fixed
# cost per call dominates). A100, H100 and H200 were only measured from
# 200 poles on, where the GEMM is faster.
_MIN_GEMM_POLES = 20
# Memory budget of the GEMM's per-slab workspace: at most 2 GiB and an
# eighth of the GPU's memory.
_MAX_GEMM_SLAB_BYTES = min(2**31, cp.cuda.Device(0).mem_info[1] // 8)
# Threads per block of the GEMM kernels: one pole, or one charge or bin,
# per thread; GEMM_THREADS in kernels.cu.
_GEMM_THREADS = 128
# Threads per block of `pole_near_field`; POLE_NEAR_THREADS in kernels.cu.
_NEAR_FIELD_THREADS = 256
# cuBLAS scalars, read from the host (CUBLAS_POINTER_MODE_HOST).
_ONE = np.ones(1, dtype=FLOAT)
_ZERO = np.zeros(1, dtype=FLOAT)


def _blocks(n_threads: int, threads_per_block: int) -> int:
    """Blocks of `threads_per_block` covering `n_threads`, at least 1."""
    return max(1, -(-n_threads // threads_per_block))


def _gemm_rows(
    a: int,
    lda: int,
    b: int,
    ldb: int,
    c: int,
    ldc: int,
    m: int,
    n: int,
    k: int,
) -> None:
    """
    C = A @ B for row-major [m, k] A, [k, n] B and [m, n] C, by cuBLAS.

    The arguments are device pointers and row strides, in elements, so
    the matrices can be column ranges of wider arrays. Row-major C = A B is
    column-major C^T = B^T A^T, which is what is handed to cuBLAS.
    """
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    cublas.dgemm(
        handle,
        cublas.CUBLAS_OP_N,
        cublas.CUBLAS_OP_N,
        n,
        m,
        k,
        _ONE.ctypes.data,
        b,
        ldb,
        a,
        lda,
        _ZERO.ctypes.data,
        c,
        ldc,
    )


@dataclass(frozen=True, eq=False)
class _GemmRotation:
    """The tables of `PoleTileGemmPlan` for one direction of the beam."""

    # Per pole: the injection weight (pair factor, flipped).
    injection: CupyArray
    # [K + 1, 2 poles]: F, charges to [Z_re | Z_im].
    f: CupyArray
    # [2 poles + K + 1, K]: [E_re; -E_im; T], [Y_re | Y_im | h] to reads.
    e: CupyArray
    # [length, poles]: the hand-over's charges to state, M.
    m_re: CupyArray
    m_im: CupyArray


@dataclass(frozen=True, eq=False)
class _GemmWorkspace:
    """Device buffers of `wake_from_pole_residue_gemm`, reused every call."""

    # [slab tiles, 2 poles + K + 1]: rows [Y_re | Y_im | h].
    work: CupyArray
    # [slab tiles, 2 poles]: rows [Z_re | Z_im].
    z: CupyArray
    # [slab tiles * K]: the slab's read-outs, bin by bin.
    reads: CupyArray
    # Per group of tiles and pole: the scan's end, propagator and start.
    end_re: CupyArray
    end_im: CupyArray
    prop_re: CupyArray
    prop_im: CupyArray
    start_re: CupyArray
    start_im: CupyArray
    # Per pole: the scan's state, the carried charge's injection after
    # tile 0, and the carried charge at -1.
    state_re: CupyArray
    state_im: CupyArray
    carry_re: CupyArray
    carry_im: CupyArray
    carried_re: CupyArray
    carried_im: CupyArray


@dataclass(frozen=True, eq=False)
class PoleTileGemmPlan:
    """
    Grid, pole tables and workspace of `CudaNoParitySpecials`' GEMM far field.

    Built once per grid and set of poles by
    `CudaNoParitySpecials.pole_tile_gemm_plan`; see
    `CudaNoParitySpecials.wake_from_pole_residue_gemm` for the scheme.
    Device arrays unless noted; complex tables are planar (re, im), except
    `pole_dt`.
    """

    n_bins: int
    n_tiles: int
    n_poles: int
    tile_length: int
    bin_dt: float
    # Host: per slab, its tiles [first, stop), and the scan's group length
    # and number of groups.
    slabs: tuple[tuple[int, int, int, int], ...]
    # Bins in the last tile.
    last_tile_length: int
    # Position, in whole bins from the first bin, of the second tile's
    # first bin.
    second_start: int
    # Per tile: the bins of its charges, the halo (the previous bin, if
    # adjacent) first; `n_bins` stands for no charge.
    gather: CupyArray
    # Per tile: its last bin (`n_bins` for the last tile).
    last_bin: CupyArray
    # Per tile: the type of the step to the next tile, -1 for the last.
    type_ids: CupyArray
    # Per pole: p bin_dt (complex), the conjugate-pair factor and the
    # counter-rotation sign.
    pole_dt: CupyArray
    pair: CupyArray
    signs: CupyArray
    # Per step type [types, poles]: the state's decay over the step, Z's
    # over the gap, and the tile's last bin's over the gap but one.
    step: tuple[CupyArray, CupyArray]
    own: tuple[CupyArray, CupyArray]
    last: tuple[CupyArray, CupyArray]
    # Per pole: d^length, the last tile's state to the hand-over, and
    # exp(p bin_dt last_offset), the carried charge's for a single tile.
    advance: tuple[CupyArray, CupyArray]
    carried_advance: tuple[CupyArray, CupyArray]
    # Per direction of the beam (is counter-rotating).
    by_rotation: dict[bool, _GemmRotation]
    workspace: _GemmWorkspace


def _gemm_tile_length(run_lengths: NumpyArray) -> int | None:
    """The longest tile length dividing every run but the last, or None."""
    return next(
        (k for k in _GEMM_TILE_LENGTHS if np.all(run_lengths[:-1] % k == 0)),
        None,
    )


def _planar(array: CupyArray) -> tuple[CupyArray, CupyArray]:
    """Contiguous real and imaginary parts of a complex array."""
    return (
        cp.ascontiguousarray(array.real, dtype=FLOAT),
        cp.ascontiguousarray(array.imag, dtype=FLOAT),
    )


def _gemm_rotation(
    pair: CupyArray,
    signs: CupyArray,
    residues: CupyArray,
    powers: CupyArray,
    toeplitz: CupyArray,
    last_tile_length: int,
    is_counterrotating_beam: bool,
) -> _GemmRotation:
    """The tables for one direction of the beam."""
    k = len(powers) - 1
    flip = cp.where((signs == -1) & is_counterrotating_beam, -1.0, 1.0)
    injection = flip * pair
    readout = flip * residues
    source = cp.zeros((k + 1, len(injection)), dtype=np.complex128)
    source[:k] = injection[None, :] * powers[k - 1 :: -1]
    e_re, e_im = _planar((readout[None, :] * powers[:k]).T)
    length = last_tile_length
    m_re, m_im = _planar(
        injection[None, :] * powers[length - 1 :: -1][:length]
    )
    return _GemmRotation(
        injection=cp.ascontiguousarray(injection, dtype=FLOAT),
        f=cp.ascontiguousarray(
            cp.concatenate((source.real, source.imag), axis=1), dtype=FLOAT
        ),
        e=cp.ascontiguousarray(
            cp.concatenate((e_re, -e_im, toeplitz), axis=0), dtype=FLOAT
        ),
        m_re=m_re,
        m_im=m_im,
    )


def _gemm_slabs(
    n_tiles: int, slab_tiles: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Per slab: first and stop tile, the scan's group length and groups."""
    slabs = []
    for first in range(0, n_tiles, slab_tiles):
        stop = min(first + slab_tiles, n_tiles)
        n_slab_tiles = stop - first
        # ~sqrt(2 n_tiles) groups: the local and the stitching pass balance
        n_groups = max(
            1, min(n_slab_tiles, round(math.sqrt(2 * n_slab_tiles)))
        )
        group_length = -(-n_slab_tiles // n_groups)
        n_groups = -(-n_slab_tiles // group_length)
        slabs.append((first, stop, group_length, n_groups))
    return tuple(slabs)


def _gemm_workspace(
    slab_tiles: int, n_groups: int, n_poles: int, tile_length: int
) -> _GemmWorkspace:
    """Allocate the buffers of `wake_from_pole_residue_gemm`."""

    def per_pole(rows: int = 1) -> CupyArray:
        return cp.zeros((rows, n_poles) if rows > 1 else n_poles, dtype=FLOAT)

    return _GemmWorkspace(
        work=cp.zeros(
            (slab_tiles, 2 * n_poles + tile_length + 1), dtype=FLOAT
        ),
        z=cp.zeros((slab_tiles, 2 * n_poles), dtype=FLOAT),
        reads=cp.zeros(slab_tiles * tile_length, dtype=FLOAT),
        end_re=per_pole(n_groups),
        end_im=per_pole(n_groups),
        prop_re=per_pole(n_groups),
        prop_im=per_pole(n_groups),
        start_re=per_pole(n_groups),
        start_im=per_pole(n_groups),
        state_re=per_pole(),
        state_im=per_pole(),
        carry_re=per_pole(),
        carry_im=per_pole(),
        carried_re=per_pole(),
        carried_im=per_pole(),
    )


class CudaNoParitySpecials:
    """
    CUDA kernels outside the `Specials` parity contract.

    Reached as ``backend.specials.no_parity`` on the CUDA backend (None on
    the others, see `Specials.no_parity`).
    """

    @staticmethod
    def pole_tile_gemm_plan(
        bin_offsets: NumpyArray,
        poles: CupyArray,
        residues: CupyArray,
        counterrotating_pole_signs: CupyArray,
        bin_dt: float,
        max_slab_bytes: int = _MAX_GEMM_SLAB_BYTES,
    ) -> PoleTileGemmPlan | None:
        """
        Plan the GEMM far field on a fixed grid, if it pays off.

        Parameters
        ----------
        bin_offsets
            Host array: position of each bin in whole bins from the first
            bin, increasing.
        poles
            Complex poles, in [rad/s].
        residues
            Complex residues, as for `Specials.wake_from_pole_residue`.
        counterrotating_pole_signs
            Per pole, -1 if its sign flips for a counter-rotating beam.
        bin_dt
            Bin width, in [s].
        max_slab_bytes
            Memory budget of the per-slab workspace, in bytes.

        Returns
        -------
        plan
            None if there are no poles or fewer than `_MIN_GEMM_POLES`, or
            if the runs of contiguous bins, but the last, share no tile
            length of 16 bins or more; `wake_from_pole_residue` is then the
            faster or the only choice.
        """
        offsets = np.asarray(bin_offsets, dtype=np.int64)
        offsets = offsets - offsets[0]
        n_bins = len(offsets)
        breaks = np.flatnonzero(np.diff(offsets) != 1) + 1
        run_starts = np.concatenate(([0], breaks))
        run_lengths = np.diff(np.concatenate((run_starts, [n_bins])))
        k = _gemm_tile_length(run_lengths)
        if len(poles) == 0 or len(poles) < _MIN_GEMM_POLES or k is None:
            return None
        first = np.concatenate(
            [
                np.arange(s, s + n, k)
                for s, n in zip(run_starts, run_lengths, strict=True)
            ]
        )
        n_tiles = len(first)
        # every run but the last is whole tiles: tile t starts at bin t K
        assert np.array_equal(first, np.arange(n_tiles) * k)
        starts = offsets[first]
        # charges per tile: halo, then the tile's bins; n_bins is a zero
        halo = np.full(n_tiles, n_bins)
        adjacent = offsets[first[1:] - 1] + 1 == starts[1:]
        halo[1:][adjacent] = first[1:][adjacent] - 1
        columns = first[:, None] + np.arange(k)[None, :]
        columns[columns >= np.append(first[1:], n_bins)[:, None]] = n_bins
        last_bin = np.minimum(first + k - 1, n_bins)
        last_bin[-1] = n_bins  # no step after the last tile
        # steps between tiles: their gap beyond a tile, as types
        gaps = starts[1:] - starts[:-1] - k
        type_gaps, type_ids = np.unique(gaps, return_inverse=True)

        poles = cp.ascontiguousarray(poles, dtype=np.complex128)
        signs = cp.ascontiguousarray(counterrotating_pole_signs, dtype=FLOAT)
        n_poles = len(poles)
        pole_dt = poles * bin_dt
        pair = cp.ascontiguousarray(2.0 - (poles.imag == 0), dtype=FLOAT)
        powers = cp.exp(cp.arange(k + 1)[:, None] * pole_dt[None, :])
        type_gaps = cp.asarray(type_gaps, dtype=FLOAT)[:, None]
        step = cp.exp((type_gaps + k) * pole_dt[None, :])
        own = cp.exp(type_gaps * pole_dt[None, :])
        last = cp.where(
            type_gaps > 0, cp.exp((type_gaps - 1) * pole_dt[None, :]), 0.0
        )
        # the flips of injection and read-out cancel in the summed wake
        wake = cp.asnumpy((powers[:k] * (residues * pair)[None, :]).sum(1))
        lag = np.arange(k)[None, :] - np.arange(k + 1)[:, None] - 1
        toeplitz = cp.asarray(
            np.where(lag >= 0, wake.real[np.clip(lag, 0, k - 1)], 0),
            dtype=FLOAT,
        )
        last_tile_length = n_bins - int(first[-1])
        bytes_per_tile = (4 * n_poles + 2 * k + 1) * 8
        slab_tiles = max(1, min(n_tiles, max_slab_bytes // bytes_per_tile))
        slabs = _gemm_slabs(n_tiles, slab_tiles)
        return PoleTileGemmPlan(
            n_bins=n_bins,
            n_tiles=n_tiles,
            n_poles=n_poles,
            tile_length=k,
            bin_dt=bin_dt,
            slabs=slabs,
            last_tile_length=last_tile_length,
            second_start=int(starts[1]) if n_tiles > 1 else 0,
            gather=cp.asarray(
                np.concatenate((halo[:, None], columns), axis=1),
                dtype=np.int64,
            ),
            last_bin=cp.asarray(last_bin, dtype=np.int64),
            type_ids=cp.asarray(np.append(type_ids, -1), dtype=np.int32),
            pole_dt=pole_dt,
            pair=pair,
            signs=signs,
            step=_planar(step),
            own=_planar(own),
            last=_planar(last),
            advance=_planar(powers[last_tile_length]),
            carried_advance=_planar(cp.exp(pole_dt * int(offsets[-1]))),
            by_rotation={
                rotation: _gemm_rotation(
                    pair,
                    signs,
                    residues,
                    powers,
                    toeplitz,
                    last_tile_length,
                    rotation,
                )
                for rotation in (False, True)
            },
            workspace=_gemm_workspace(
                slab_tiles,
                max(slab[3] for slab in slabs),
                n_poles,
                k,
            ),
        )

    @staticmethod
    def wake_from_pole_residue_gemm(
        # read
        plan: PoleTileGemmPlan,
        profile: CupyArray,
        carried_charge: CupyArray,
        carried_is_counterrotating: bool,
        state_lag_dt: float,
        carried_lag_dt: float,
        is_counterrotating_beam: bool,
        factor: float,
        # write
        states: CupyArray,
        voltage: CupyArray,
    ) -> None:
        """
        `Specials.wake_from_pole_residue` as tiled GEMMs, on `plan`'s grid.

        Same contract and result as `Specials.wake_from_pole_residue`; the
        grid, poles and residues come from `plan`. The poles are a diagonal
        linear state-space model, evaluated in tiles of K contiguous bins:

            Z_t     = h_t . F                     tile t's own sum
            Y_{t+1} = A_t Y_t + G_t Z_t + ...     scan over the tiles
            V_t     = Re(Y_t . E) + h_t . T       read-outs of tile t

        with h_t the tile's charges (and a halo), d = exp(p bin_dt),
        F = inj d^(K - 1 - s), E = r d^k, and T the wake summed over the
        poles at lags below a tile. The sum over the poles is the inner
        dimension of both GEMMs, which cuBLAS runs on the FP64 tensor cores
        of sm_80+ GPUs; the scan is the only sequential part (kernels.cu).
        The tiles are processed in slabs of the plan's workspace. Per slab
        that is a kernel for the charges, a GEMM, three for the scan, a
        GEMM and one for the read-outs, and one more for the hand-over: the
        call is bound by the GPU, not by launching it.

        Parameters
        ----------
        plan
            From `pole_tile_gemm_plan`, for this grid and these poles.
        profile
            Beam profile histogram, one value per bin.
        carried_charge
            Length 1: charge of the previous call's last bin.
        carried_is_counterrotating
            Whether the call of the carried charge was counter-rotating.
        state_lag_dt
            How far before the first read-out the states are referenced,
            in [s].
        carried_lag_dt
            How long before the first bin the carried charge was emitted,
            in [s].
        is_counterrotating_beam
            Whether the beam is counter-rotating.
        factor
            To convert `profile` to charge per bin.
        states
            Complex state per pole; read and overwritten with the state one
            bin before the last bin.
        voltage
            The far-field voltage per bin, in [V]; overwritten.
        """
        for array in (profile, carried_charge, voltage):
            assert array.dtype == FLOAT
            assert array.flags.c_contiguous
        assert states.dtype == np.complex128
        assert states.flags.c_contiguous
        assert profile.shape[0] == plan.n_bins
        assert voltage.shape[0] == plan.n_bins

        k, n_poles, n_bins = plan.tile_length, plan.n_poles, plan.n_bins
        rotation = plan.by_rotation[is_counterrotating_beam]
        ws = plan.workspace
        work, z, reads = ws.work, ws.z, ws.reads
        work_ld, z_ld = work.shape[1], z.shape[1]
        h_col = 2 * n_poles
        work_ptr, z_ptr = work.data.ptr, z.data.ptr
        factor = FLOAT(factor)
        n_bins_arg, n_poles_arg = np.int32(n_bins), np.int32(n_poles)
        pole_blocks = _blocks(n_poles, _GEMM_THREADS)
        block = (_GEMM_THREADS, 1, 1)
        for first, stop, group_length, n_groups in plan.slabs:
            n_slab_tiles = stop - first
            # the per-pole quantities once, with the first slab's charges
            n_prep_poles = n_poles if first == 0 else 0
            n_prep = n_slab_tiles * (k + 1) + n_prep_poles
            _pole_gemm_prep(
                (_blocks(n_prep, _GEMM_THREADS),),
                block,
                (
                    profile,
                    factor,
                    plan.gather,
                    n_bins_arg,
                    np.int32(first),
                    np.int32(n_slab_tiles),
                    np.int32(k),
                    work,
                    np.int32(work_ld),
                    np.int32(h_col),
                    np.int32(n_prep_poles),
                    states,
                    plan.pole_dt,
                    plan.pair,
                    plan.signs,
                    carried_charge,
                    np.int32(carried_is_counterrotating),
                    FLOAT(carried_lag_dt / plan.bin_dt - 1.0),
                    FLOAT(state_lag_dt / plan.bin_dt),
                    FLOAT(plan.second_start - 1),
                    ws.state_re,
                    ws.state_im,
                    ws.carry_re,
                    ws.carry_im,
                    ws.carried_re,
                    ws.carried_im,
                ),
            )
            # Z = h . F
            _gemm_rows(
                work_ptr + 8 * h_col,
                work_ld,
                rotation.f.data.ptr,
                z_ld,
                z_ptr,
                z_ld,
                n_slab_tiles,
                z_ld,
                k + 1,
            )
            scan_grid = (pole_blocks, n_groups, 1)
            _pole_gemm_scan_local(
                scan_grid,
                block,
                (
                    z,
                    np.int32(z_ld),
                    plan.type_ids,
                    plan.last_bin,
                    profile,
                    factor,
                    n_bins_arg,
                    *plan.step,
                    *plan.own,
                    *plan.last,
                    rotation.injection,
                    ws.carry_re,
                    ws.carry_im,
                    np.int32(first),
                    np.int32(n_slab_tiles),
                    n_poles_arg,
                    np.int32(group_length),
                    work,
                    np.int32(work_ld),
                    ws.end_re,
                    ws.end_im,
                    ws.prop_re,
                    ws.prop_im,
                ),
            )
            _pole_gemm_scan_groups(
                (pole_blocks,),
                block,
                (
                    ws.end_re,
                    ws.end_im,
                    ws.prop_re,
                    ws.prop_im,
                    np.int32(n_groups),
                    n_poles_arg,
                    ws.start_re,
                    ws.start_im,
                    ws.state_re,
                    ws.state_im,
                ),
            )
            _pole_gemm_scan_fix(
                scan_grid,
                block,
                (
                    plan.type_ids,
                    *plan.step,
                    ws.start_re,
                    ws.start_im,
                    np.int32(first),
                    np.int32(n_slab_tiles),
                    n_poles_arg,
                    np.int32(group_length),
                    work,
                    np.int32(work_ld),
                ),
            )
            # reads = [Y_re | Y_im | h] . [E_re; -E_im; T]
            _gemm_rows(
                work_ptr,
                work_ld,
                rotation.e.data.ptr,
                k,
                reads.data.ptr,
                k,
                n_slab_tiles,
                k,
                work_ld,
            )
            bin_first = first * k
            n_slab_bins = min(stop * k, n_bins) - bin_first
            # tile 0, from its second bin on, takes the carried charge too
            n_carried_bins = min(k - 1, n_slab_bins - 1) if first == 0 else 0
            _pole_gemm_readout(
                (n_carried_bins + _blocks(n_slab_bins, _GEMM_THREADS),),
                block,
                (
                    reads,
                    np.int64(bin_first),
                    np.int64(n_slab_bins),
                    np.int32(n_carried_bins),
                    np.int32(k),
                    n_poles_arg,
                    ws.carried_re,
                    ws.carried_im,
                    rotation.e,
                    voltage,
                ),
            )
        # hand over: one bin before the last bin, every bin but the last in
        _pole_gemm_handover(
            (pole_blocks,),
            block,
            (
                work,
                np.int32(work_ld),
                np.int32(plan.n_tiles - 1 - plan.slabs[-1][0]),
                np.int32(h_col),
                np.int32(plan.last_tile_length),
                n_poles_arg,
                rotation.m_re,
                rotation.m_im,
                *plan.advance,
                np.int32(plan.n_tiles == 1),
                ws.carried_re,
                ws.carried_im,
                *plan.carried_advance,
                states,
            ),
        )

    @staticmethod
    def pole_near_field(
        # read
        profile: CupyArray,
        factor: float,
        taps: CupyArray,
        adjacent: CupyArray,
        # read and write
        carried_charge: CupyArray,
        voltage: CupyArray,
        # the carried charge's wake at the first bin: a constant, ...
        carried_wake: float = 0.0,
        # ... or summed over these poles (all None for the constant)
        carried_poles: CupyArray | None = None,
        carried_residues: CupyArray | None = None,
        carried_weights: CupyArray | None = None,
        carried_shift_dt: float = 0.0,
    ) -> None:
        """
        Near field of a pole-residue model and the carried charge, in one go.

        Adds to `voltage` the near field of `MultiPoleSparseSolve`, the taps
        at lags of +1, 0 and -1 bin applied to a bin's neighbours where they
        are adjacent, and the carried charge's whole contribution to the
        first bin; then carries the last bin's charge to the next call.

        Parameters
        ----------
        profile
            Beam profile histogram, one value per bin.
        factor
            To convert `profile` to charge per bin.
        taps
            Length 3: the wake at lags of +1, 0 and -1 bin, in [V/C].
        adjacent
            Bool, per pair of consecutive bins: one bin apart.
        carried_charge
            Length 1: charge of the previous call's last bin; overwritten
            with ``factor * profile[-1]``.
        voltage
            Induced voltage per bin, in [V]; added to.
        carried_wake
            The carried charge's wake at the first bin, in [V/C], when
            `carried_poles` is None.
        carried_poles
            Complex poles, in [rad/s]; with them the carried charge's wake
            is ``sum(carried_weights * Re(carried_residues * exp(
            carried_poles * carried_shift_dt)))``.
        carried_residues
            Complex residues, one per pole.
        carried_weights
            Real weight per pole.
        carried_shift_dt
            Time by which the residues are advanced, in [s].
        """
        for array in (profile, taps, carried_charge, voltage):
            assert array.dtype == FLOAT
            assert array.flags.c_contiguous
        assert adjacent.dtype == np.bool_
        n_bins = int(profile.shape[0])
        assert voltage.shape[0] == n_bins
        assert adjacent.shape[0] == max(n_bins - 1, 0)
        if carried_poles is None:
            n_carried_poles = 0
            carried_poles = carried_residues = carried_weights = taps
        else:
            n_carried_poles = int(carried_poles.shape[0])
            assert carried_poles.dtype == np.complex128
            assert carried_residues.dtype == np.complex128
            assert carried_weights.dtype == FLOAT
            assert carried_residues.shape[0] == n_carried_poles
            assert carried_weights.shape[0] == n_carried_poles
        _pole_near_field(
            (_blocks(n_bins, _NEAR_FIELD_THREADS),),
            (_NEAR_FIELD_THREADS,),
            (
                profile,
                FLOAT(factor),
                taps,
                adjacent,
                np.int32(n_bins),
                carried_charge,
                np.int32(n_carried_poles),
                carried_poles,
                carried_residues,
                carried_weights,
                FLOAT(carried_shift_dt),
                FLOAT(carried_wake),
                voltage,
            ),
        )


class CudaSpecials(Specials):  # NOQA: D101
    no_parity = CudaNoParitySpecials()

    @staticmethod
    def get_max_threads() -> int:  # NOQA: D102
        """
        Return the max number of threads this backend's kernels may use.

        Returns
        -------
        max_threads
            Maximum number of threads this backend's kernels may use.
        """
        return 1

    @staticmethod
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: CupyArray,
        dE: CupyArray,
        flags: CupyArray,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert flags.device != "cpu", (
            f"Requires Cupy array, but got {type(flags)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert flags.dtype == np.int32

        _loss_box(
            args=(
                FLOAT(e_max),
                FLOAT(e_min),
                FLOAT(t_min),
                FLOAT(t_max),
                dt,
                dE,
                flags,
                np.int32(BeamFlags.LOST.value),  # flag_lost
                INDEX_DTYPE(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def kick_single_harmonic(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        _kick_single_harmonic(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                FLOAT(charge),  # charge
                FLOAT(voltage),  # voltage
                FLOAT(omega_rf),  # omega_RF
                FLOAT(phi_rf),  # phi_RF
                INDEX_DTYPE(len(dE)),  # n_macroparticles
                FLOAT(acceleration_kick),  # acc_kick
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def kick_multi_harmonic(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        # The per-harmonic parameters stay on the host: they are passed to
        # the kernel by value, see `_RF_PARAMS_BATCH_DTYPE`.
        assert isinstance(voltage, np.ndarray), type(voltage)
        assert isinstance(omega_rf, np.ndarray), type(omega_rf)
        assert isinstance(phi_rf, np.ndarray), type(phi_rf)

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        assert len(voltage) == len(omega_rf) == len(phi_rf) == n_rf

        # One launch per `MAX_RF_HARMONICS_PER_LAUNCH` harmonics, and at
        # least one so that `acceleration_kick` is applied for n_rf == 0.
        # Each launch adds its harmonics to `dE`; only the last one adds
        # `acceleration_kick`.
        for first in range(0, max(n_rf, 1), MAX_RF_HARMONICS_PER_LAUNCH):
            last = min(first + MAX_RF_HARMONICS_PER_LAUNCH, n_rf)
            rf_params_batch = np.zeros((), dtype=_RF_PARAMS_BATCH_DTYPE)
            rf_params_batch["voltage"][: last - first] = voltage[first:last]
            rf_params_batch["omega_rf"][: last - first] = omega_rf[first:last]
            rf_params_batch["phi_rf"][: last - first] = phi_rf[first:last]
            is_last_launch = last == n_rf
            _kick_multi_harmonic(
                args=(
                    dt,  # beam_dt
                    dE,  # beam_dE
                    rf_params_batch,  # rf_params_batch
                    np.int32(last - first),  # n_rf_in_batch
                    FLOAT(charge),  # charge
                    INDEX_DTYPE(len(dE)),  # n_macroparticles
                    FLOAT(acceleration_kick if is_last_launch else 0.0),
                ),
                block=block_size,
                grid=grid_size,
            )

    @staticmethod
    def sum_1d_array(array: CupyArray) -> float:
        """Return the sum of 1d array."""
        assert array.device != "cpu", (
            f"Requires Cupy array, but got {type(array)}."
        )
        return cp.sum(array)

    @staticmethod
    def dot_product_1d_array(array_1: CupyArray, array_2: CupyArray):  # NOQA: D102
        assert array_1.device != "cpu", (
            f"Requires Cupy array, but got {type(array_1)}."
        )
        assert array_2.device != "cpu", (
            f"Requires Cupy array, but got {type(array_2)}."
        )

        """Return the sum of dot product of two 1d arrays."""
        return cp.dot(array_1, array_2)

    @staticmethod
    def drift_simple(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        # Cast Python floats to backend floattype
        T = FLOAT(T)
        eta_0 = FLOAT(eta_0)
        beta = FLOAT(beta)
        energy = FLOAT(energy)

        _drift_simple(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                T,  # T
                eta_0,  # eta_zero
                beta,  # beta
                energy,  # energy
                INDEX_DTYPE(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def drift_like_line_segment(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        # Cast Python floats to backend floattype
        T = FLOAT(T)
        eta_0 = FLOAT(eta_0)
        beta = FLOAT(beta)
        energy = FLOAT(energy)

        _drift_like_line_segment(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                T,  # T
                eta_0,  # eta_zero
                beta,  # beta
                energy,  # energy
                INDEX_DTYPE(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def drift_exact(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        alpha_0: float,
        higher_alpha: CupyArray,
        beta: float,
        energy: float,
    ) -> None:
        assert dt.device != "cpu"
        assert dE.device != "cpu"
        assert higher_alpha.device != "cpu"

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert higher_alpha.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert higher_alpha.flags.c_contiguous

        T = FLOAT(T)
        alpha_0 = FLOAT(alpha_0)
        beta = FLOAT(beta)
        energy = FLOAT(energy)

        _drift_exact(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                T,  # t_rev
                alpha_0,  # alpha_zero
                higher_alpha,  # higher_alpha
                np.int32(len(higher_alpha)),  # n_alpha
                beta,  # beta
                energy,  # energy
                INDEX_DTYPE(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

    @staticmethod
    def kick_interpolated(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: CupyArray,
        bin_centers: CupyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: CupyArray | None = None,
        bucket_index_to_memory_index: CupyArray | None = None,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert voltage.device != "cpu", (
            f"Requires Cupy array, but got {type(voltage)}."
        )
        assert bin_centers.device != "cpu", (
            f"Requires Cupy array, but got {type(bin_centers)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert bin_centers.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert bin_centers.flags.c_contiguous

        # Cast Python floats to backend floattype
        charge = FLOAT(charge)
        acceleration_kick = FLOAT(acceleration_kick)

        if first_left_cut is None:
            n_slices = bin_centers.size
            assert n_slices >= 2, (  # noqa: PLR2004
                "kick_interpolated needs at least 2 bins to interpolate "
                f"across, got {n_slices}"
            )

            glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), FLOAT)
            _gm_linear_interp_kick_help(
                args=(
                    dt,
                    dE,
                    voltage,
                    bin_centers,
                    charge,
                    np.int32(bin_centers.size),
                    INDEX_DTYPE(dt.size),
                    acceleration_kick,
                    glob_vkick_factor,
                ),
                grid=grid_size,
                block=block_size,
            )

            _gm_linear_interp_kick_comp(
                args=(
                    dt,
                    dE,
                    voltage,
                    bin_centers,
                    FLOAT(charge),
                    np.int32(bin_centers.size),
                    INDEX_DTYPE(dt.size),
                    acceleration_kick,
                    glob_vkick_factor,
                ),
                grid=grid_size,
                block=block_size,
            )
            return

        assert filling_pattern.device != "cpu", (
            f"Requires Cupy array, but got {type(filling_pattern)}."
        )
        assert bucket_index_to_memory_index.device != "cpu", (
            f"Requires Cupy array, but got "
            f"{type(bucket_index_to_memory_index)}."
        )
        assert filling_pattern.dtype == np.bool_
        assert bucket_index_to_memory_index.dtype == np.int32
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), FLOAT)
        _gm_linear_interp_kick_sparse_help(
            args=(
                voltage,
                bin_centers,
                charge,
                np.int32(bin_centers.size),
                acceleration_kick,
                FLOAT(cut_width),
                np.int32(bins_per_profile),
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )

        _gm_linear_interp_kick_sparse_comp(
            args=(
                dt,
                dE,
                INDEX_DTYPE(dt.size),
                FLOAT(first_left_cut),
                FLOAT(left_cut_distance),
                FLOAT(cut_width),
                np.int32(bins_per_profile),
                np.int32(len(filling_pattern)),
                filling_pattern,
                bucket_index_to_memory_index,
                acceleration_kick,
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )

    @staticmethod
    def histogram(  # NOQA: D102
        array_read: CupyArray,
        array_write: CupyArray,
        start: float,
        stop: float,
    ) -> None:
        assert array_read.device != "cpu", (
            f"Requires Cupy array, but got {type(array_read)}."
        )
        assert array_write.device != "cpu", (
            f"Requires Cupy array, but got {type(array_write)}."
        )

        assert array_read.dtype == FLOAT
        assert array_write.dtype == FLOAT
        assert array_read.flags.c_contiguous
        assert array_write.flags.c_contiguous

        # Cast Python floats to backend floattype
        start = FLOAT(start)
        stop = FLOAT(stop)

        n_slices = array_write.size
        array_write.fill(0)

        if _HIST_COUNT_ITEMSIZE * n_slices < max_shared_memory_per_block:
            _sm_histogram(
                args=(
                    array_read,
                    array_write,
                    start,
                    stop,
                    np.uint32(n_slices),
                    INDEX_DTYPE(len(array_read)),
                ),
                grid=grid_size,
                block=block_size,
                shared_mem=_HIST_COUNT_ITEMSIZE * n_slices,
            )
        else:
            _hybrid_histogram(
                args=(
                    array_read,
                    array_write,
                    start,
                    stop,
                    np.uint32(n_slices),
                    INDEX_DTYPE(len(array_read)),
                    np.int32(
                        max_shared_memory_per_block // _HIST_COUNT_ITEMSIZE
                    ),
                ),
                grid=grid_size,
                block=block_size,
                shared_mem=max_shared_memory_per_block,
            )

    @staticmethod
    def beam_phase(  # NOQA: D102
        hist_x: CupyArray,
        hist_y: CupyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        assert hist_x.device != "cpu", (
            f"Requires Cupy array, but got {type(hist_x)}."
        )
        assert hist_y.device != "cpu", (
            f"Requires Cupy array, but got {type(hist_y)}."
        )

        assert hist_x.dtype == FLOAT
        assert hist_y.dtype == FLOAT
        assert hist_x.flags.c_contiguous
        assert hist_y.flags.c_contiguous
        assert len(hist_x) >= 2, (  # noqa: PLR2004
            "beam_phase requires at least two bins for the trapezoidal rule, "
            f"got {len(hist_x)}"
        )

        # Cast Python floats to backend floattype
        alpha = FLOAT(alpha)
        omega_rf = FLOAT(omega_rf)
        phi_rf = FLOAT(phi_rf)
        bin_size = FLOAT(bin_size)

        result = cp.zeros(2, dtype=FLOAT)
        # The kernel reduces each block in shared memory instead of
        # striding over the array like the other kernels, so it needs one
        # block per `threads` bins: the fixed `grid_size` would silently
        # drop every bin beyond `blocks * threads`.
        n_blocks = (len(hist_x) + threads - 1) // threads
        _beam_phase(
            args=(
                hist_x,  # hist_x
                hist_y,  # hist_y
                result,  # result
                alpha,  # alpha
                omega_rf,  # omega_rf
                phi_rf,  # phi_rf
                bin_size,  # bin_size
                np.int32(len(hist_x)),  # n_bins
            ),
            block=block_size,
            grid=(n_blocks, 1, 1),
            shared_mem=2 * block_size[0] * np.dtype(FLOAT).itemsize,
        )
        return FLOAT((result[0] / result[1]).get())

    @staticmethod
    def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
        beam_dE: CupyArray,
        energy_lost: float,
        longitudinal_damping_time: float,
        natural_energy_spread: float,
        total_energy: float,
        disable_quantum_excitation: bool = False,
    ) -> None:
        """
        Apply synchrotron radiation and quantum excitation energy kicks.

        Single fused CUDA kernel — one launch, one pass over ``beam_dE``,
        no auxiliary noise buffer. The Gaussian noise is drawn with
        NVIDIA's cuRAND device library (one state per thread), so the
        memory footprint is just the beam itself.

        Parameters
        ----------
        beam_dE
            Macro-particle energy coordinates, in [eV]. CuPy array,
            modified in place.
        energy_lost
            Energy lost through the considered synchrotron segment,
            in [eV per turn].
        longitudinal_damping_time
            Longitudinal damping time, in [turn].
        natural_energy_spread
            Natural energy spread, [dimensionless].
        total_energy
            Beam total reference energy, in [eV].
        disable_quantum_excitation
           Disables the quantum excitation kick.
        """
        assert beam_dE.device != "cpu", (
            f"Requires Cupy array, but got {type(beam_dE)}."
        )
        assert beam_dE.dtype == FLOAT
        assert beam_dE.flags.c_contiguous

        damping_factor = FLOAT(1.0 - 2.0 / longitudinal_damping_time)
        energy_lost_typed = FLOAT(energy_lost)
        n_macroparticles = INDEX_DTYPE(len(beam_dE))
        if disable_quantum_excitation:
            _apply_sr_without_quantum_excitation(
                args=(
                    beam_dE,
                    damping_factor,
                    energy_lost_typed,
                    n_macroparticles,
                ),
                block=block_size,
                grid=grid_size,
            )
        else:
            noise_scale = FLOAT(
                2.0
                * natural_energy_spread
                / float(np.sqrt(longitudinal_damping_time))
                * total_energy
            )
            # The counter guarantees a distinct seed per launch even on
            # platforms where consecutive clock reads return the same value.
            # Each thread uses its tid as the cuRAND subsequence.
            base_seed = np.uint64(next(_quantum_excitation_seed_counter))
            _apply_sr_with_quantum_excitation(
                args=(
                    beam_dE,
                    damping_factor,
                    energy_lost_typed,
                    noise_scale,
                    base_seed,
                    n_macroparticles,
                ),
                block=block_size,
                grid=grid_size,
            )

    @staticmethod
    def move_flagged_elements_to_end(  # NOQA: D102
        flag: int,
        flags: CupyArray,
        dt: CupyArray,
        dE: CupyArray,
        ids: CupyArray,
    ):
        assert flags.device != "cpu", (
            f"Requires Cupy array, but got {type(flags)}."
        )
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert ids.device != "cpu", (
            f"Requires Cupy array, but got {type(ids)}."
        )

        # TODO write a kernel that works with gpu kernels
        #  to have a smaller memory footprint.
        flag = np.int32(flag)
        assert flags.dtype == np.int32
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert ids.dtype == INDEX_DTYPE

        select = flags == flag
        order = cp.argsort(select)

        flags[:] = flags[order]
        dt[:] = dt[order]
        dE[:] = dE[order]
        ids[:] = ids[order]

        # Host `int` like the other backends. The device sync is
        # unavoidable: the caller slices the arrays with `n_new`.
        n_new = len(ids) - int(cp.sum(select))
        return n_new

    @staticmethod
    def histogram_sparse(  # NOQA: D102
        x: CupyArray,
        out: CupyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: CupyArray,
        bucket_index_to_memory_index: CupyArray,
    ) -> None:
        assert x.device != "cpu", f"Requires Cupy array, but got {type(x)}."
        assert out.device != "cpu", (
            f"Requires Cupy array, but got {type(out)}."
        )
        assert filling_pattern.device != "cpu", (
            f"Requires Cupy array, but got {type(filling_pattern)}."
        )
        assert bucket_index_to_memory_index.device != "cpu", (
            f"Requires Cupy array, but got {type(bucket_index_to_memory_index)}."
        )

        assert x.dtype == FLOAT
        assert out.dtype == FLOAT
        assert filling_pattern.dtype == np.bool
        assert bucket_index_to_memory_index.dtype == np.int32

        assert x.flags.c_contiguous
        assert out.flags.c_contiguous
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        out[:] = 0
        _histogram_sparse(
            args=(
                x,  # input
                out,  # output
                FLOAT(first_left_cut),  # first_left_cut
                FLOAT(left_cut_distance),  # left_cut_distance
                FLOAT(cut_width),  # cut_width
                np.int32(bins_per_profile),  # bins_per_profile
                np.int32(len(filling_pattern)),  # n_buckets
                INDEX_DTYPE(len(x)),  # n_macroparticles
                filling_pattern,  # input
                bucket_index_to_memory_index,  # input
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def wake_from_twc_fir(
        # read
        profile: CupyArray,
        grid_index: CupyArray,
        r_shunt: CupyArray,
        a_tilde: CupyArray,
        omega_r: CupyArray,
        bin_dt: float,
        factor: float,
        # write
        voltage: CupyArray,
        voltage_threaded: CupyArray,
    ) -> None:
        """
        Travelling-wave-cavity wake, as window sums of prefix sums.

        See `Specials.wake_from_twc_fir` for the contract and every
        parameter. The other backends' recursion is a convolution on the
        lattice with an undamped phasor, so the wake at a bin is two sums
        over the bins within its support, each the difference of two
        prefix sums over the bins: a parallel scan, three kernels for all
        bins and modes (kernels.cu).

        Parameters
        ----------
        voltage_threaded
            Unused on the CUDA backend (kept for API parity with CPU
            backends); one thread per bin sums the modes.
        """
        for array, dtype in (
            (profile, FLOAT),
            (grid_index, np.int32),
            (r_shunt, FLOAT),
            (a_tilde, FLOAT),
            (omega_r, FLOAT),
            (voltage, FLOAT),
        ):
            assert array.device != "cpu", (
                f"Requires Cupy array, but got {type(array)}."
            )
            assert array.dtype == dtype
            assert array.flags.c_contiguous

        n_bins = int(profile.shape[0])
        n_modes = int(r_shunt.shape[0])
        assert grid_index.shape[0] == n_bins
        assert a_tilde.shape[0] == n_modes
        assert omega_r.shape[0] == n_modes
        assert voltage.shape[0] == n_bins

        if n_modes == 0 or n_bins == 0:
            voltage.fill(0)
            return
        # From CuPy's memory pool, so no allocation after the first call.
        n_chunks = -(-n_bins // _TWC_SCAN_CHUNK)
        prefix = cp.empty((n_modes, n_bins, 4), dtype=FLOAT)
        chunk_totals = cp.empty((n_modes, n_chunks, 4), dtype=FLOAT)
        offsets = cp.empty((n_modes, n_chunks, 4), dtype=FLOAT)
        phasors = cp.empty((n_modes, n_bins, 2), dtype=FLOAT)
        block = (_TWC_THREADS, 1, 1)
        _twc_scan_chunks(
            (n_chunks, n_modes, 1),
            block,
            (
                profile,
                grid_index,
                omega_r,
                FLOAT(bin_dt),
                np.int32(n_bins),
                prefix,
                chunk_totals,
                phasors,
            ),
        )
        _twc_scan_totals(
            (n_modes, 1, 1),
            block,
            (chunk_totals, np.int32(n_chunks), offsets),
        )
        _wake_from_twc_fir(
            (-(-n_bins // _TWC_THREADS), 1, 1),
            block,
            (
                profile,
                grid_index,
                r_shunt,
                a_tilde,
                omega_r,
                FLOAT(bin_dt),
                FLOAT(factor),
                np.int32(n_bins),
                np.int32(n_modes),
                np.int32(n_chunks),
                prefix,
                offsets,
                phasors,
                voltage,
            ),
        )

    @staticmethod
    def wake_from_pole_residue(
        # read
        profile_time: CupyArray,
        profile: CupyArray,
        carried_charge: CupyArray,
        carried_is_counterrotating: bool,
        state_lag_dt: float,
        carried_lag_dt: float,
        poles: CupyArray,
        residues: CupyArray,
        is_counterrotating_beam: bool,
        counterrotating_pole_signs: CupyArray,
        factor: float,
        bin_dt: float,
        # write
        states: CupyArray,
        voltage: CupyArray,
        voltage_threaded: CupyArray,
    ) -> None:
        """
        Far field of a pole-residue wake, one complex state per pole.

        See `Specials.wake_from_pole_residue` for the contract and every
        parameter.

        Parameters
        ----------
        voltage_threaded
            Unused on the CUDA backend (kept for API parity with CPU
            backends); pole contributions are summed per warp and
            reduced into `voltage` directly via atomic adds.
        """
        complex_dtype = np.complex64 if np.float32 == FLOAT else np.complex128
        for array, dtype in (
            (profile_time, FLOAT),
            (profile, FLOAT),
            (carried_charge, FLOAT),
            (poles, complex_dtype),
            (residues, complex_dtype),
            (counterrotating_pole_signs, FLOAT),
            (states, complex_dtype),
            (voltage, FLOAT),
        ):
            assert array.device != "cpu", (
                f"Requires Cupy array, but got {type(array)}."
            )
            assert array.dtype == dtype
            assert array.flags.c_contiguous

        n_bins = int(profile.shape[0])
        n_poles = int(poles.shape[0])
        assert profile_time.shape[0] == n_bins
        assert carried_charge.shape[0] == 1
        assert states.shape[0] == n_poles
        assert residues.shape[0] == n_poles
        assert counterrotating_pole_signs.shape[0] == n_poles
        assert voltage.shape[0] == n_bins

        if n_poles == 0:
            voltage.fill(0)
            return
        # The kernels step along the bin grid in whole bins. Unlike the
        # other backends they do not take bins off that grid: the caller,
        # `MultiPoleSparseSolve`, checks that its bins lie a whole number
        # of bins apart.
        bin_offsets = cp.rint((profile_time - profile_time[0]) / bin_dt)
        _wake_from_pole_residue_in_chunks(
            bin_offsets=bin_offsets.astype(INDEX_DTYPE),
            profile=profile,
            carried_charge=carried_charge,
            carried_is_counterrotating=carried_is_counterrotating,
            state_lag_dt=state_lag_dt,
            carried_lag_dt=carried_lag_dt,
            poles=poles,
            residues=residues,
            is_counterrotating_beam=is_counterrotating_beam,
            counterrotating_pole_signs=counterrotating_pole_signs,
            factor=factor,
            bin_dt=bin_dt,
            states=states,
            voltage=voltage,
            n_chunks=_pole_chunk_count(n_bins, n_poles, _N_MULTIPROCESSORS),
        )

    @staticmethod
    def music_track(  # NOQA: D102 inherited from `Specials.music_track`
        beam_dt: CupyArray,
        beam_dE: CupyArray,
        induced_voltage: CupyArray,
        parameter_array: CupyArray,
        alpha: float,
        omega_bar: float,
        const: float,
        coeff1: float,
        coeff2: float,
        coeff3: float,
        coeff4: float,
        time_since_last_track: float,
        multiturn: bool,
    ) -> None:
        # TODO 20260629.0 : Fix Notes when implementing CUDA/NUMBA backend
        raise NotImplementedError
