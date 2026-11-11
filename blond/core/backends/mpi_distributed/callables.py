# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions to interface with MPI distributed arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.mpi_distributed.helpers import (
    MPI,
    mpi_cuda_aware,
    mpi_is_distributed,
)
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.backends.mpi_distributed.distributed_array import (
        DistributedArray,
    )

from blond.core.backends.backend import backend
from blond.core.backends.flags import BeamFlags


def global_reduce(*local_values: float, op: str = "SUM") -> NumpyArray:
    """
    Reduce a few per-rank scalars over all MPI ranks in one collective.

    The values travel as one float64 buffer, not as pickled Python
    objects: one `Allreduce` instead of a length and a payload each.
    float64 holds any count up to 2**53 exactly.

    Parameters
    ----------
    *local_values
        The values on this rank.
    op
        Name of the MPI reduction, e.g. ``"SUM"``, ``"MIN"`` or ``"MAX"``.

    Returns
    -------
    global_values
        The reduced values, or `local_values` without MPI.
    """
    buffer = np.array(local_values, dtype=np.float64)
    if mpi_is_distributed():
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, buffer, op=getattr(MPI, op))
    return buffer


# Page-locked host buffers for `global_sum_array`, one per array shape
# and dtype; reused every turn, and copies from/to them are ~2x faster.
_PINNED_HOST_BUFFERS: dict[tuple, NumpyArray] = {}


def _pinned_host_buffer(like: CupyArray) -> NumpyArray:
    """
    Return a reusable page-locked host array shaped like `like`.

    Parameters
    ----------
    like
        CuPy array whose shape and dtype the buffer gets.

    Returns
    -------
    buffer
        Host array in pinned memory, shared by every call with the same
        shape and dtype.
    """
    key = (like.shape, like.dtype.str)
    buffer = _PINNED_HOST_BUFFERS.get(key)
    if buffer is None:
        import cupy as cp  # only reached for CuPy arrays

        memory = cp.cuda.alloc_pinned_memory(like.nbytes)
        buffer = np.frombuffer(memory, like.dtype, like.size)
        buffer = buffer.reshape(like.shape)
        _PINNED_HOST_BUFFERS[key] = buffer
    return buffer


def global_sum_array(
    local_array: NumpyArray | CupyArray,
) -> NumpyArray | CupyArray:
    """
    Sum a per-rank array element-wise over all MPI ranks, in place.

    By default a CuPy array is summed via a pinned host copy: handing
    device memory to an MPI library that is not CUDA-aware crashes or
    gives wrong sums. With `BLOND_MPI_CUDA_AWARE=True`, MPI gets the
    device memory directly, which skips both copies.

    Parameters
    ----------
    local_array
        The contiguous array on this rank; overwritten with the sum.

    Returns
    -------
    global_array
        `local_array`, now holding the sum over all ranks.
    """
    if not mpi_is_distributed():
        return local_array
    assert local_array.flags.c_contiguous, "MPI needs a contiguous buffer"
    if not is_cupy_array(local_array):
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, local_array, op=MPI.SUM)
    elif mpi_cuda_aware():
        import cupy as cp  # only reached for CuPy arrays

        # MPI cannot see CuPy's queued kernels: wait until the array
        # is written before MPI reads it.
        cp.cuda.get_current_stream().synchronize()
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, local_array, op=MPI.SUM)
    else:
        host_array = _pinned_host_buffer(local_array)
        local_array.get(out=host_array)  # blocking: waits for the kernels
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, host_array, op=MPI.SUM)
        local_array.set(host_array)
    return local_array


def rms_emittance(dt: DistributedArray, dE: DistributedArray) -> float:
    """
    Calculate the Root-Mean-Square emittance of the beam.

    Parameters
    ----------
    dt
        The beam time coordinates, in [s].
    dE
        The beam energy coordinates, in [eV].

    Returns
    -------
    rms_emittance
        The Root-Mean-Square emittance in [s eV] of the beam.
    """
    n = dt.global_size
    sum_dt = backend.specials.sum_1d_array(dt.array_local)
    sum_dE = backend.specials.sum_1d_array(dE.array_local)
    dt_mean, dE_mean = global_reduce(float(sum_dt), float(sum_dE)) / n

    # Centre on the global mean before taking second moments. The
    # one-pass form sum(x**2)/n - mean**2 cancels catastrophically when
    # the spread is small compared to the mean (e.g. a bunch far from
    # dt = 0) and can even turn negative.
    dt_centred = dt.array_local - dt_mean
    dE_centred = dE.array_local - dE_mean

    # use dot(x,x) for faster calculation of sum(x**2)
    dot = backend.specials.dot_product_1d_array
    sigma_dt_squared, sigma_dE_squared, sigma_dE_dt = (
        global_reduce(
            float(dot(dt_centred, dt_centred)),
            float(dot(dE_centred, dE_centred)),
            float(dot(dt_centred, dE_centred)),
        )
        / n
    )

    # The determinant is >= 0 in exact arithmetic (Cauchy-Schwarz), but
    # rounding can push it slightly negative for a zero-emittance beam.
    determinant = sigma_dt_squared * sigma_dE_squared - sigma_dE_dt**2
    return float(np.sqrt(max(determinant, 0.0)))


def count_active_macroparticles(flags: DistributedArray) -> int:
    """
    Count the macro-particles flagged as active across all processes.

    Parameters
    ----------
    flags
        The beam macro-particle flags.

    Returns
    -------
    n_active_macroparticles
        Number of macro-particles with the flag
        ``BeamFlags.ACTIVE``.
    """
    local_count = int(backend.sum(flags.array_local == BeamFlags.ACTIVE.value))
    return int(global_reduce(local_count)[0])
