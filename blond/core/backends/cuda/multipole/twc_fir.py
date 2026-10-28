# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Launches the CUDA prefix-scan kernels of `CudaSpecials.wake_from_twc_fir`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np

from blond.core.backends.cuda.gpu_module import FLOAT, gpu_module

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore

_twc_scan_chunks = gpu_module.get_function("twc_scan_chunks")
_twc_scan_totals = gpu_module.get_function("twc_scan_totals")
_wake_from_twc_fir = gpu_module.get_function("wake_from_twc_fir")

# Threads per block of the TWC kernels, TWC_THREADS in twc_fir.cuh.
_TWC_THREADS = 256
# Bins per chunk of the TWC prefix scan, TWC_CHUNK in twc_fir.cuh.
_TWC_SCAN_CHUNK = 4 * _TWC_THREADS


def wake_from_twc_fir_scan(
    profile: CupyArray,
    grid_index: CupyArray,
    r_shunt: CupyArray,
    a_tilde: CupyArray,
    omega_r: CupyArray,
    bin_dt: float,
    factor: float,
    voltage: CupyArray,
) -> None:
    """
    Launch the three TWC kernels for all bins and modes.

    See `Specials.wake_from_twc_fir` for every parameter and twc_fir.cuh
    for the scheme. `voltage` is overwritten. Needs at least one bin and
    one mode.
    """
    n_bins = int(profile.shape[0])
    n_modes = int(r_shunt.shape[0])
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
