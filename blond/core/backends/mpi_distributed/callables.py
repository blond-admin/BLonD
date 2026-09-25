# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions to interface with MPI distributed arrays."""

from __future__ import annotations

import warnings
from math import sqrt
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

try:
    from mpi4py import MPI
except Exception as exc:
    warnings.warn(str(exc), ImportWarning, stacklevel=1)
    MPI = None

from blond.generals.distributed.helpers import mpi_is_distributed

if TYPE_CHECKING:  # pragma: no cover
    from blond.generals.distributed.distributed_array import DistributedArray

from blond.core.backends.backend import backend


class PhaseSpaceMoments(NamedTuple):
    """
    The five particle sums every phase-space statistic of a beam comes from.

    Means, RMS sizes and the RMS emittance are all functions of these sums,
    so :func:`phase_space_moments` computes them once and every statistic
    reads them from here -- no statistic needs its own pass over the
    particles.

    Attributes
    ----------
    n_macroparticles
        Number of macro-particles summed over (all processes).
    dt_sum
        Sum of ``dt`` [s].
    dE_sum
        Sum of ``dE`` [eV].
    dt_dt_sum
        Sum of ``dt**2`` [s^2].
    dE_dE_sum
        Sum of ``dE**2`` [eV^2].
    dt_dE_sum
        Sum of ``dt * dE`` [s eV].
    """

    n_macroparticles: int
    dt_sum: float
    dE_sum: float
    dt_dt_sum: float
    dE_dE_sum: float
    dt_dE_sum: float

    @property
    def mean_dt(self) -> float:
        """
        Mean of ``dt`` [s].

        Returns
        -------
        mean_dt
            ``dt_sum / n``, as :meth:`DistributedArray.mean` forms it.
        """
        return self.dt_sum / self.n_macroparticles

    @property
    def mean_dE(self) -> float:
        """
        Mean of ``dE`` [eV].

        Returns
        -------
        mean_dE
            ``dE_sum / n``, as :meth:`DistributedArray.mean` forms it.
        """
        return self.dE_sum / self.n_macroparticles

    @property
    def sigma_dt(self) -> float:
        """
        RMS size of ``dt`` [s].

        Returns
        -------
        sigma_dt
            ``sqrt(<dt^2> - <dt>^2)``, as :meth:`DistributedArray.std`
            forms it.
        """
        mean_dt = self.mean_dt
        return sqrt(self.dt_dt_sum / self.n_macroparticles - mean_dt**2)

    @property
    def sigma_dE(self) -> float:
        """
        RMS size of ``dE`` [eV].

        Returns
        -------
        sigma_dE
            ``sqrt(<dE^2> - <dE>^2)``, as :meth:`DistributedArray.std`
            forms it.
        """
        mean_dE = self.mean_dE
        return sqrt(self.dE_dE_sum / self.n_macroparticles - mean_dE**2)

    @property
    def rms_emittance(self) -> float:
        """
        The Root-Mean-Square emittance [s eV].

        Returns
        -------
        rms_emittance
            ``sqrt(det(covariance))`` of the (``dt``, ``dE``) distribution.
        """
        over_n = 1 / self.n_macroparticles
        sigma_dt_squared = (
            self.dt_dt_sum * over_n - (self.dt_sum * over_n) ** 2
        )
        sigma_dE_squared = (
            self.dE_dE_sum * over_n - (self.dE_sum * over_n) ** 2
        )
        sigma_dE_dt = (
            self.dt_dE_sum * over_n - self.dt_sum * self.dE_sum * over_n**2
        )
        rms = np.sqrt(sigma_dt_squared * sigma_dE_squared - sigma_dE_dt**2)
        return float(rms)


def phase_space_moments(
    dt: DistributedArray, dE: DistributedArray
) -> PhaseSpaceMoments:
    """
    Compute the phase-space sums of a beam in one pass.

    One fused pass of the active backend's ``phase_space_sums`` kernel over
    the local particles (threaded on the compiled backends), reduced over
    all processes when distributed.

    Parameters
    ----------
    dt
        The beam time coordinates, in [s].
    dE
        The beam energy coordinates, in [eV].

    Returns
    -------
    moments
        The five sums and the particle count.
    """
    local_sums = backend.specials.phase_space_sums(
        dt.array_local, dE.array_local
    )
    local_count = dt.local_size

    if mpi_is_distributed():
        comm = MPI.COMM_WORLD
        return PhaseSpaceMoments(
            comm.allreduce(local_count, op=MPI.SUM),
            *(comm.allreduce(value, op=MPI.SUM) for value in local_sums),
        )
    return PhaseSpaceMoments(local_count, *local_sums)


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
    return phase_space_moments(dt=dt, dE=dE).rms_emittance
