# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Launches the tiled CUDA kernels of `CudaSpecials.wake_from_pole_residue`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np

from blond.core.backends.cuda.gpu_module import FLOAT, gpu_module

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore

_pole_residue_chunk_maps = gpu_module.get_function("pole_residue_chunk_maps")
_wake_from_pole_residue = gpu_module.get_function("wake_from_pole_residue")


# Shape of the pole kernels; must match POLE_TILE, POLE_BLOCK and
# POLE_THREADS in pole_residue.cuh.
_POLE_TILE = 64  # bins per tile, at most
_POLE_BLOCK = 16  # poles per block
_POLE_THREADS = 64  # threads per block
# Blocks per multiprocessor the pole kernels aim for (tuned on a T400).
_POLE_BLOCKS_PER_MULTIPROCESSOR = 16
# Fewest tiles per time chunk.
_MIN_POLE_CHUNK_TILES = 4
_N_MULTIPROCESSORS = cp.cuda.Device(0).attributes["MultiProcessorCount"]


def pole_chunk_count(
    n_bins: int,
    n_poles: int,
    n_multiprocessors: int = _N_MULTIPROCESSORS,
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


def wake_from_pole_residue_in_chunks(
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
    pole_residue.cuh for the scheme. `voltage` is overwritten.

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
    # Both kernels start with these arguments; see pole_residue.cuh. Complex
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
