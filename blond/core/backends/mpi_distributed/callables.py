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
    mpi_is_distributed,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.backends.mpi_distributed.distributed_array import (
        DistributedArray,
    )

from blond.core.backends.backend import backend
from blond.core.backends.flags import BeamFlags


def _global_sum(local_value: float) -> float:
    """
    Sum a per-rank value over all MPI ranks.

    Parameters
    ----------
    local_value
        The value on this rank.

    Returns
    -------
    global_value
        The sum over all ranks, or `local_value` without MPI.
    """
    if mpi_is_distributed():
        return MPI.COMM_WORLD.allreduce(local_value, op=MPI.SUM)
    return local_value


def _global_dot(
    array_1: NumpyArray | CupyArray, array_2: NumpyArray | CupyArray
) -> float:
    """
    Dot product of two rank-local arrays, summed over all MPI ranks.

    Parameters
    ----------
    array_1
        Rank-local input array 1.
    array_2
        Rank-local input array 2.

    Returns
    -------
    global_dot
        The dot product over the global arrays.
    """
    local_dot = backend.specials.dot_product_1d_array(array_1, array_2)
    return _global_sum(float(local_dot))


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
    n = _global_sum(dt.local_size)
    dt_mean = (
        _global_sum(float(backend.specials.sum_1d_array(dt.array_local))) / n
    )
    dE_mean = (
        _global_sum(float(backend.specials.sum_1d_array(dE.array_local))) / n
    )

    # Centre on the global mean before taking second moments. The
    # one-pass form sum(x**2)/n - mean**2 cancels catastrophically when
    # the spread is small compared to the mean (e.g. a bunch far from
    # dt = 0) and can even turn negative.
    dt_centred = dt.array_local - dt_mean
    dE_centred = dE.array_local - dE_mean

    # use dot(x,x) for faster calculation of sum(x**2)
    sigma_dt_squared = _global_dot(dt_centred, dt_centred) / n
    sigma_dE_squared = _global_dot(dE_centred, dE_centred) / n
    sigma_dE_dt = _global_dot(dt_centred, dE_centred) / n

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
    if mpi_is_distributed():  # pragma: no cover
        return int(MPI.COMM_WORLD.allreduce(local_count, op=MPI.SUM))
    return local_count
