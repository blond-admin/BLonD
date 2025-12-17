# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions to interface with MPI distributed arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from blond import backend

if TYPE_CHECKING:  # pragma: no cover
    from blond.generals.distributed.distributed_array import DistributedArray


def mpi_is_active() -> float:
    """
    Check whether MPI is active.

    Returns
    -------
    mpi_active
        True, if MPI is active.
    """
    return (
        MPI is not None
        and MPI.Is_initialized()
        and MPI.COMM_WORLD.Get_size() > 1
    )


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
    # use dot(x,x) for faster calculation of sum(x**2)
    local_dt_dt_sum = float(backend.dot(dt.array_local, dt.array_local))
    local_dE_dE_sum = float(backend.dot(dE.array_local, dE.array_local))
    local_dt_dE_sum = float(backend.dot(dt.array_local, dE.array_local))
    local_count = dt.local_size

    if dt.is_distributed:
        comm = dt.comm  # or dE.comm
        dt_dt_sum = comm.allreduce(local_dt_dt_sum, op=MPI.SUM)
        dE_dE_sum = comm.allreduce(local_dE_dE_sum, op=MPI.SUM)
        dt_dE_sum = comm.allreduce(local_dt_dE_sum, op=MPI.SUM)
        n = comm.allreduce(local_count, op=MPI.SUM)
    else:
        dt_dt_sum = local_dt_dt_sum
        dE_dE_sum = local_dE_dE_sum
        dt_dE_sum = local_dt_dE_sum
        n = local_count
    over_n = 1 / n
    rms = np.sqrt(
        (dt_dt_sum * over_n) * (dE_dE_sum * over_n) - (dt_dE_sum * over_n) ** 2
    )
    return float(rms)
