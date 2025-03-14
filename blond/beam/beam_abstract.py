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

    from ..input_parameters.ring import Ring
    from ..input_parameters.rf_parameters import RFStation
    from ..utils.types import SolverTypes

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
            Class containing the general properties of the synchrotron
        rf_station : RFStation
            Class containing all the RF parameters for all the RF systems
            in one ring segment
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
        """Minimum of all 'dt'"""
        pass

    @abstractmethod
    def dE_min(self):  # todo ignore lost particles?
        """Minimum of all 'dE'"""
        pass

    @abstractmethod
    def dt_max(self):  # todo ignore lost particles?
        """Maximum of all 'dt'"""
        pass

    @abstractmethod
    def dE_max(self):  # todo ignore lost particles?
        """Maximum of all 'dE'"""
        pass

    @abstractmethod
    def kick(
        self, rf_station: RFStation, acceleration_kicks: NDArray, turn_i: int
    ):
        r"""Function updating the dE array

        Function updating the particle energy due to the RF kick in a given
        RF station. The kicks are summed over the different harmonic RF systems
        in the station. The cavity phase can be shifted by the user via
        phi_offset. The main RF (harmonic[0]) has by definition phase = 0 at
        time = 0 below transition. The phases of all other RF systems are
        defined w.r.t.\ to the main RF. The increment in energy is given by the
        discrete equation of motion:

        .. math::
            \Delta E^{n+1} = \Delta E^n + \sum_{k=0}^{n_{\mathsf{rf}}-1}{e V_k^n \\sin{\\left(\omega_{\mathsf{rf,k}}^n \\Delta t^n + \phi_{\mathsf{rf,k}}^n \\right)}} - (E_s^{n+1} - E_s^n)

        """

        pass

    @abstractmethod
    def drift(self, rf_station: RFStation, solver: SolverTypes, turn_i: int):
        r"""Function updating the dt array

        Function updating the particle arrival time to the RF station
        (drift). If only the zeroth order slippage factor is given, 'simple'
        and 'exact' solvers are available. The 'simple' solver is somewhat
        faster. Otherwise, the solver is automatically 'exact' and calculates
        the frequency slippage up to second order. The corresponding equations
        are (nb: the n indices correspond to the turn number):

        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1} \\left[ \\left(1+\\sum_{i=0}^{2}{\\alpha_i\\left(\\delta^{n+1}\\right)^{i+1}}\\right)   \\frac{1+\\left(\\Delta E/E_s\\right)^{n+1}}{1+\\delta^{n+1}}    - 1\\right] \quad \\text{(exact)}

        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1} \\left(\\frac{1}{1 - \\eta(\\delta^{n+1})\\delta^{n+1}} - 1\\right) \quad \\text{(legacy)}

        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1}\\eta_0\\delta^{n+1} \quad \\text{(simple)}

        The relative momentum needs to be calculated from the relative energy
        and is obtained as follows:

        .. math::
            \\delta = \\sqrt{1+\\beta_s^{-2}\\left[\\left(\\frac{\\Delta E}{E_s}\\right)^2 + 2\\frac{\\Delta E}{E_s}\\right]} - 1 \quad \\text{(exact)}

        .. math::
            \\delta = \\frac{\\Delta E}{\\beta_s^2 E_s} \quad \\text{(simple, legacy)}

        """
        pass

    def linear_interp_kick(
        self,
        voltage: NDArray,
        bin_centers: NDArray,
        charge: float,
        acceleration_kick: float,
    ):
        pass


    def kickdrift_considering_periodicity(
        self,
        acceleration_kicks: NDArray,
        rf_station: RFStation,
        solver: SolverTypes,
        turn_i: int,
    ):
        pass

    def slice_beam(self, profile, cut_left, cut_right):  # todo rewrite using bmath
        """Computes a histogram of the dt coordinates"""
        pass