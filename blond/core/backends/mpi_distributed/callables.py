# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions to interface with MPI distributed arrays."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np


try:
    from mpi4py import MPI
except Exception as exc:
    warnings.warn(str(exc), ImportWarning, stacklevel=1)
    MPI = None

from blond.generals.distributed.helpers import mpi_is_distributed

if TYPE_CHECKING:  # pragma: no cover
    from blond.generals.distributed.distributed_array import DistributedArray

from .ipac_hacks import parallel_dot, parallel_sum


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
    local_dt_sum = float(parallel_sum(dt.array_local))
    local_dE_sum = float(parallel_sum(dE.array_local))

    # use dot(x,x) for faster calculation of sum(x**2)
    local_dt_dt_sum = float(parallel_dot(dt.array_local, dt.array_local))
    local_dE_dE_sum = float(parallel_dot(dE.array_local, dE.array_local))
    local_dt_dE_sum = float(parallel_dot(dt.array_local, dE.array_local))
    local_count = dt.local_size

    if mpi_is_distributed():
        comm = MPI.COMM_WORLD
        dt_dt_sum = comm.allreduce(local_dt_dt_sum, op=MPI.SUM)
        dE_dE_sum = comm.allreduce(local_dE_dE_sum, op=MPI.SUM)
        dt_dE_sum = comm.allreduce(local_dt_dE_sum, op=MPI.SUM)
        dt_sum = comm.allreduce(local_dt_sum, op=MPI.SUM)
        dE_sum = comm.allreduce(local_dE_sum, op=MPI.SUM)
        n = comm.allreduce(local_count, op=MPI.SUM)
    else:
        dt_dt_sum = local_dt_dt_sum
        dE_dE_sum = local_dE_dE_sum
        dt_dE_sum = local_dt_dE_sum
        dt_sum = local_dt_sum
        dE_sum = local_dE_sum
        n = local_count

    over_n = 1 / n
    sigma_dt_squared = dt_dt_sum * over_n - (dt_sum * over_n) ** 2
    sigma_dE_squared = dE_dE_sum * over_n - (dE_sum * over_n) ** 2
    sigma_dE_dt = dt_dE_sum * over_n - dt_sum * dE_sum * over_n**2

    rms = np.sqrt(sigma_dt_squared * sigma_dE_squared - sigma_dE_dt**2)
    return float(rms)
