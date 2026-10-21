# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CudaNoParitySpecials`: the GEMM far field and the near field."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np
from cupy_backends.cuda.libs import cublas  # type: ignore

from blond.core.backends.cuda.gpu_module import FLOAT, gpu_module

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

_pole_gemm_scan_local = gpu_module.get_function("pole_gemm_scan_local")
_pole_gemm_scan_groups = gpu_module.get_function("pole_gemm_scan_groups")
_pole_gemm_scan_fix = gpu_module.get_function("pole_gemm_scan_fix")
_pole_gemm_prep = gpu_module.get_function("pole_gemm_prep")
_pole_gemm_readout = gpu_module.get_function("pole_gemm_readout")
_pole_gemm_handover = gpu_module.get_function("pole_gemm_handover")
_pole_near_field = gpu_module.get_function("pole_near_field")

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
# per thread; GEMM_THREADS in pole_gemm.cuh.
_GEMM_THREADS = 128
# Threads per block of `pole_near_field`; POLE_NEAR_THREADS in pole_gemm.cuh.
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
        poles at lags below a tile. The first GEMM sums over a tile's
        charges, the second over the poles (and the charges, for T); both
        need no atomics, and cuBLAS runs them on the FP64 tensor cores of
        sm_80+ GPUs. The scan is the only sequential part (pole_gemm.cuh).
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
