# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the fundamental beam class with methods to compute beam
statistics

:Authors: **Simon Lauber**

"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import physical_constants

from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import cupy as cp

    from ..input_parameters.ring import Ring
    from ..input_parameters.rf_parameters import RFStation

m_mu = physical_constants["muon mass"][0]


class BeamBaseClass:
    @handle_legacy_kwargs
    def __init__(
        self, ring: Ring, n_macroparticles: int, intensity: float
    ) -> None:
        self.particle = ring.particle
        self.beta: float = ring.beta[0][0]
        self.gamma: float = ring.gamma[0][0]
        self.energy: float = ring.energy[0][0]
        self.momentum: float = ring.momentum[0][0]

        self.mean_dt: float = 0.0
        self.mean_dE: float = 0.0
        self.sigma_dt: float = 0.0
        self.sigma_dE: float = 0.0
        self.intensity: float = float(intensity)
        self.n_macroparticles: int = int(n_macroparticles)
        self.ratio: float = self.intensity / self.n_macroparticles

        self.epsn_rms_l: float = 0.0
        self.n_macroparticles_eliminated = 0

        # For MPI
        self._mpi_n_total_macroparticles_lost: int = 0
        self._mpi_n_total_macroparticles: int = n_macroparticles
        self._mpi_is_splitted: bool = False
        self._mpi_sumsq_dt: NDArray | float = 0.0
        self._mpi_sumsq_dE: NDArray | float = 0.0
        # For handling arrays on CPU/GPU
        self._device = "CPU"

    @property
    @abstractmethod
    def n_macroparticles_alive(self) -> int:
        """Number of macro-particles marked as alive (id ≠ 0)

        Returns
        -------
        n_macroparticles_alive : int
            Number of macro-particles marked as alive (id ≠ 0)

        """
        pass

    @property
    def n_macroparticles_not_alive(self):
        """Number of macro-particles marked as not-alive (id=0)

        Returns
        -------
        n_macroparticles_not_alive : int
            Number of macro-particles marked as not-alive (id=0)

        """
        return self.n_macroparticles - self.n_macroparticles_alive

    @abstractmethod
    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays"""
        pass

    @abstractmethod
    def statistics(self) -> None:
        r"""Update statistics of dE and dE array

        Notes
        -----
        Following attributes are updated:
        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        """
        pass

    @abstractmethod
    @handle_legacy_kwargs
    def losses_separatrix(self, ring: Ring, rf_station: RFStation) -> None:
        """Mark particles outside separatrix as not-alive (id=0)

        Parameters
        ----------
        ring : Ring
            Function arguments of is_in_separatrix(...).
        rf_station : RFStation
            Function arguments of is_in_separatrix(...).

        """

    @abstractmethod
    def losses_longitudinal_cut(self, dt_min: float, dt_max: float) -> None:
        """Mark particles outside time range as not-alive (id=0)

        Parameters
        ----------
        dt_min : float
            Lower limit (dt=dt_min is kept)
        dt_max : float
            Upper limit (dt=dt_max is kept)
        """

    @abstractmethod
    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        dE_max : float
            Upper limit (dE=dE_max is kept)
        """

    @abstractmethod
    def losses_below_energy(self, dE_min: float):
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        """

    def particle_decay(self, time: float) -> None:
        """Decreases beam intensity due to the particle decay

        Sets the ratio to a lower value if the particle can decay.
        Number of macroparticles remains unchanged.

        Parameters
        ----------
        time : float
            time in seconds, which is used
            to determine the fraction of the
            particle decay
        """
        self.ratio *= np.exp(
            -time * self.particle.decay_rate / self.gamma
        )  # todo bugfix should act on number of particles?

    @abstractmethod
    def dE_mean(self, ignore_id_0: bool = False):
        """Calculate mean of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        pass

    @abstractmethod
    def dE_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        pass

    @abstractmethod
    def dt_mean(self, ignore_id_0: bool = False):
        """Calculate mean of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        pass

    @abstractmethod
    def dt_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        pass

    @abstractmethod
    def dt_min(self):  # todo ignore lost particles?
        pass

    @abstractmethod
    def dE_min(self):  # todo ignore lost particles?
        pass

    @abstractmethod
    def dt_max(self):  # todo ignore lost particles?
        pass

    @abstractmethod
    def dE_max(self):  # todo ignore lost particles?
        pass
