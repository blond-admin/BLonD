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

:Authors: **Danilo Quartullo**, **Helga Timko**, **ALexandre Lasheen**

"""

from __future__ import annotations

import itertools as itl
import warnings
from typing import TYPE_CHECKING, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e, m_p, physical_constants

from .beam_abstract import BeamBaseClass
from ..trackers.utilities import is_in_separatrix
from ..utils import bmath as bm
from ..utils import exceptions as blond_exceptions
from ..utils.bmath_extras import mean_and_std
from ..utils.custom_warnings import PerformanceWarning
from ..utils.legacy_support import handle_legacy_kwargs

try:
    import cupy as cp
except ModuleNotFoundError:
    pass


if TYPE_CHECKING:
    from cupy.typing import CupyArray
    from numpy.typing import NumpyArray

    from ..input_parameters.rf_parameters import RFStation
    from ..input_parameters.ring import Ring
    from ..utils.types import DeviceType, SolverTypes

m_mu = physical_constants['muon mass'][0]


class Particle:
    r"""Class containing basic parameters, e.g. mass, of the particles to be tracked.

    The following particles are already implemented: proton, electron, positron

    Parameters
    ----------
    user_mass : float
        Energy equivalent of particle rest mass in eV
    user_charge : float
        Particle charge in units of the elementary charge
    user_decay_rate : float
        Particle decay rate in units of 1/s 

    Attributes
    ----------
    mass : float
        Energy equivalent of particle rest mass in eV.
    charge : float
        Particle charge in units of the elementary charge.
    decay_rate : float
        Inverse of the particle decay time (decay time in s)
    radius_cl : float
        Classical particle radius in :math:`m`.
    c_gamma : float
        Sand's radiation constant :math:`C_\gamma` in :math:`m / eV^3`.
    c_q : float
        Quantum radiation constant :math:`C_q` in :math:`m`.

    Examples
    --------
    >>> from blond.beam.beam import Proton
    >>> particle = Proton()

    Usually, a `Particle` is used to construct a :class:`~blond.input_parameters.ring.Ring` object,
    e.g.

    >>> Ring(circumference, momentum_compaction, sync_momentum, Proton())

    """

    def __init__(self, user_mass: float, user_charge: float, user_decay_rate: float = 0) -> None:

        if user_mass > 0.:
            self.mass = float(user_mass)
            self.charge = float(user_charge)
        else:
            # MassError
            raise RuntimeError('ERROR: Particle mass not recognized!')

        if user_decay_rate >= 0.:
            self.decay_rate = float(user_decay_rate)

        else:
            # MassError
            raise RuntimeError('ERROR: Invalid particle decay rate!')

        # classical particle radius [m]
        self.radius_cl = 0.25 / (np.pi * epsilon_0) * \
                         e ** 2 * self.charge ** 2 / (self.mass * e)

        # Sand's radiation constant [ m / eV^3]
        self.c_gamma = 4 * np.pi / 3 * self.radius_cl / self.mass ** 3

        # Quantum radiation constant [m]
        self.c_q = (55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (self.mass * e))


class Proton(Particle):
    """ Implements a proton `Particle`.
    """

    def __init__(self) -> None:
        Particle.__init__(self, m_p * c ** 2 / e, 1)


class Electron(Particle):
    """ Implements an electron `Particle`.
    """

    def __init__(self) -> None:
        Particle.__init__(self, m_e * c ** 2 / e, -1)


class Positron(Particle):
    """ Implements a positron `Particle`.
    """

    def __init__(self) -> None:
        Particle.__init__(self, m_e * c ** 2 / e, 1)


class MuPlus(Particle):
    """ Implements a muon+ `Particle`.
    """

    def __init__(self):
        Particle.__init__(self, m_mu * c ** 2 / e, 1, float(1 / 2.1969811e-6))


class MuMinus(Particle):
    """ Implements a muon- `Particle`.
    """

    def __init__(self):
        Particle.__init__(self, m_mu * c ** 2 / e, -1, float(1 / 2.1969811e-6))


class Beam(BeamBaseClass):
    r"""Class containing the beam properties.

    This class containes the beam coordinates (dt, dE) and the beam properties.

    The beam coordinate 'dt' is defined as the particle arrival time to the RF
    station w.r.t. the reference time that is the sum of turns. The beam
    coordiate 'dE' is defined as the particle energy offset w.r.t. the
    energy of the synchronous particle.

    The class creates a beam with zero dt and dE, see distributions to match
    a beam with respect to the RF and intensity effects.

    Parameters
    ----------
    ring : Ring
        Used to import different quantities such as the mass and the energy.
    n_macroparticles : int
        total number of macroparticles.
    intensity : float
        total intensity of the beam (in number of charge).

    Attributes
    ----------
    beta : float
        relativistic velocity factor [].
    gamma : float
        relativistic mass factor [].
    energy : float
        energy of the synchronous particle [eV].
    momentum : float
        momentum of the synchronous particle [eV].
    dt : numpy_array, float
        beam arrival times with respect to synchronous time [s].
    dE : numpy_array, float
        beam energy offset with respect to the synchronous particle [eV].
    mean_dt : float
        average beam arrival time [s].
    mean_dE : float
        average beam energy offset [eV].
    sigma_dt : float
        standard deviation of beam arrival time [s].
    sigma_dE : float
        standard deviation of beam energy offset [eV].
    intensity : float
        total intensity of the beam in number of charges [].
    n_macroparticles : int
        total number of macroparticles in the beam [].
    ratio : float
        ratio intensity per macroparticle [].
    id : numpy_array, int
        unique macro-particle ID number; zero if particle is 'lost'.
    n_macroparticles_eliminated : int
        Number of macroparticles that were removed
        by Beam.eliminate_lost_particles()

    See Also
    ---------
    distributions.matched_from_line_density:
        match a beam with a given bunch profile.
    distributions.matched_from_distribution_function:
        match a beam with a given distribution function in phase space.

    Examples
    --------
    >>> from input_parameters.ring import Ring
    >>> from beam.beam import Beam
    >>>
    >>> n_turns = 10
    >>> C = 100
    >>> eta = 0.03
    >>> momentum = 26e9
    >>> ring = Ring(n_turns, C, eta, momentum, 'proton') # todo this seems to be not working anymore
    >>> n_macroparticle = 1e6
    >>> intensity = 1e11
    >>>
    >>> my_beam = Beam(ring, n_macroparticle, intensity)
    """



    @handle_legacy_kwargs
    def __init__(self,
                 ring: Ring,
                 n_macroparticles: int,
                 intensity: float,
                 dt: Optional[NumpyArray | CupyArray] = None,
                 dE: Optional[NumpyArray | CupyArray] = None,
                 weights: Optional[NumpyArray | CupyArray]=None
                 ) -> None:

        super().__init__(
            ring=ring,
            n_macroparticles=n_macroparticles,
            intensity=intensity
        )
        self._ring = ring
        if dt is None:
            self.dt: NumpyArray | CupyArray  = bm.zeros([int(n_macroparticles)],
                                                        dtype=bm.precision.real_t)
        else:
            assert n_macroparticles == len(dt)
            self.dt: NumpyArray | CupyArray  = bm.ascontiguousarray(dt,
                                                                    dtype=bm.precision.real_t)

        if dE is None:
            self.dE: NumpyArray | CupyArray  = bm.zeros([int(n_macroparticles)],
                                               dtype=bm.precision.real_t)
        else:
            assert n_macroparticles == len(dE)
            self.dE: NumpyArray | CupyArray  = bm.ascontiguousarray(dE,
                                                                    dtype=bm.precision.real_t)

        if weights is None:
            weights: NumpyArray | CupyArray | None  = None
        else:
            assert n_macroparticles == len(weights)

            # machine limits for integer types
            machine_max = np.iinfo(np.int32).max
            # check upper limit
            msg = (f"`weights` should be < {machine_max},"
                   f" but {bm.max(weights)=}")
            assert bm.max(weights) < machine_max, msg
            # check lower limit
            msg = (f"`weights` should be > 0,"
                   f" but {bm.min(weights)=}")
            if bm.min(weights) == 0:
                warn(msg, PerformanceWarning, stacklevel=2)

            sum_weights = bm.sum(weights)
            if sum_weights >= machine_max:
                msg = (f"Overflow possible with `weights`, because the"
                       f" maximum allowed integer in one histogram bin is {machine_max},"
                       f" but could reach {sum_weights}!")
                warn(msg, UserWarning, stacklevel=2)
            weights: NumpyArray | CupyArray  = bm.ascontiguousarray(
                weights,
                dtype=np.int32)
        self.weights = weights

        self.id: NumpyArray | CupyArray = bm.arange(
            1,
            self.n_macroparticles + 1,
            dtype=int
        )


    @property
    def Particle(self):
        from warnings import warn
        warn("Particle is deprecated, use particle", DeprecationWarning)
        return self.particle

    @Particle.setter
    def Particle(self, val):
        from warnings import warn
        warn("Particle is deprecated, use particle", DeprecationWarning)
        self.particle = val

    @property
    def n_total_macroparticles_lost(self):
        warnings.warn("Use '_mpi_n_total_macroparticles_lost' instead !", DeprecationWarning)
        return self._mpi_n_total_macroparticles_lost

    @n_total_macroparticles_lost.setter
    def n_total_macroparticles_lost(self, val):
        self._mpi_n_total_macroparticles_lost = val

    @property
    def n_total_macroparticles(self):
        warnings.warn("Use '_mpi_n_total_macroparticles' instead !", DeprecationWarning)
        return self._mpi_n_total_macroparticles

    @n_total_macroparticles.setter
    def n_total_macroparticles(self, val):
        self._mpi_n_total_macroparticles = val

    @property
    def is_splitted(self):
        warnings.warn("Use '_mpi_is_splitted' instead !", DeprecationWarning)
        return self._mpi_is_splitted

    @is_splitted.setter
    def is_splitted(self, val):
        self._mpi_is_splitted = val

    @property
    def _sumsq_dt(self):
        warnings.warn("Use '_mpi_sumsq_dt' instead !", DeprecationWarning)
        return self._mpi_sumsq_dt

    @_sumsq_dt.setter
    def _sumsq_dt(self, val):
        self._mpi_sumsq_dt = val

    @property
    def _sumsq_dE(self):
        warnings.warn("Use '_mpi_sumsq_dE' instead !", DeprecationWarning)
        return self._mpi_sumsq_dE

    @_sumsq_dE.setter
    def _sumsq_dE(self, val):
        self._mpi_sumsq_dE = val

    @property
    def n_macroparticles_lost(self) -> int:
        """Number of macro-particles marked as not alive

        Returns
        -------
        n_macroparticles_lost : int
            number of macroparticles where 'id' is 'lost' (i.e. 0).

        """
        warnings.warn("Use 'n_macroparticles_not_alive' instead of 'n_macroparticles_lost' for readability",
                      DeprecationWarning)

        return self.n_macroparticles - self.n_macroparticles_alive

    @property
    def n_macroparticles_alive(self) -> int:
        """Number of macro-particles marked as alive (id ≠ 0)

        Returns
        -------
        n_macroparticles_alive : int
            number of macroparticles not lost.

        """

        return bm.count_nonzero(self.id)

    @property
    def n_macroparticles_not_alive(self):
        '''Number of macro-particles marked as not-alive

        Returns
        -------
        n_macroparticles_not_alive : int
            number of macroparticles marked as lost.

        '''

        return self.n_macroparticles - self.n_macroparticles_alive

    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays
        """

        select_alive = self.id != 0
        if bm.sum(select_alive) > 0:
            self.n_macroparticles_eliminated += bm.sum(~select_alive)
            self.dt = bm.ascontiguousarray(
                self.dt[select_alive], dtype=bm.precision.real_t)
            self.dE = bm.ascontiguousarray(
                self.dE[select_alive], dtype=bm.precision.real_t)
            if self.weights is not None:
                self.weights = bm.ascontiguousarray(
                    self.weights[select_alive], dtype=np.int32)
            self.n_macroparticles = len(self.dt)
            self.id = bm.arange(1, self.n_macroparticles + 1, dtype=int)
        else:
            # AllParticlesLost
            raise RuntimeError("ERROR in Beams: all particles lost and" +
                               " eliminated!")

    def statistics(self) -> None:
        r"""
        Calculation of the mean and standard deviation of beam coordinates,
        as well as beam emittance using different definitions.
        Take no arguments, statistics stored in

        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        """

        # Statistics only for particles that are not flagged as lost
        if self.weights is not None:
            itemindex = self.id > 0 # this could be as well used to set weights to 0..
            self.mean_dt, self.sigma_dt = mean_and_std(
                self.dt[itemindex],
                weights=self.weights[itemindex]
            )
            self.mean_dE, self.sigma_dE = mean_and_std(
                self.dE[itemindex],
                weights=self.weights[itemindex]
            )
        else:
            self.mean_dt = self.dt_mean(ignore_id_0=True)
            self.sigma_dt = self.dt_std(ignore_id_0=True)
            self.mean_dE = self.dE_mean(ignore_id_0=True)
            self.sigma_dE = self.dE_std(ignore_id_0=True)
        itemindex = self.id > 0 # this could be as well used to set weights to 0..
        self._mpi_sumsq_dt = bm.dot(self.dt[itemindex], self.dt[itemindex])
        # self.min_dt = bm.min(self.dt[itemindex])
        # self.max_dt = bm.max(self.dt[itemindex])
        self._mpi_sumsq_dE = bm.dot(self.dE[itemindex], self.dE[itemindex])
        # TODO _mpi_sumsq_dE must be handled

        # self.min_dE = bm.min(self.dE[itemindex])
        # self.max_dE = bm.max(self.dE[itemindex])

        # R.m.s. emittance in Gaussian approximation
        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs

    @handle_legacy_kwargs
    def losses_separatrix(self, ring: Ring, rf_station: RFStation) -> None:
        """Beam losses based on separatrix.

        Set to 0 all the particle's id not in the separatrix anymore.

        Parameters
        ----------
        ring : Ring
            Used to call the function is_in_separatrix.
        rf_station : RFStation
            Used to call the function is_in_separatrix.
        """

        lost_index = is_in_separatrix(ring, rf_station, self,
                                      self.dt, self.dE) == False

        self.id[lost_index] = 0

    def losses_longitudinal_cut(self, dt_min: float, dt_max: float) -> None:
        """Beam losses based on longitudinal cuts.

        Set to 0 all the particle's id with dt not in the interval
        (dt_min, dt_max).

        Parameters
        ----------
        dt_min : float
            minimum dt.
        dt_max : float
            maximum dt.
        """

        lost_index = (self.dt < dt_min) | (self.dt > dt_max)
        self.id[lost_index] = 0

    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        """Beam losses based on energy cuts, e.g. on collimators.

        Set to 0 all the particle's id with dE not in the interval (dE_min, dE_max).

        Parameters
        ----------
        dE_min : float
            minimum dE.
        dE_max : float
            maximum dE.
        """

        lost_index = (self.dE < dE_min) | (self.dE > dE_max)
        self.id[lost_index] = 0

    def losses_below_energy(self, dE_min: float):
        """Beam losses based on lower energy cut.

        Set to 0 all the particle's id with dE below dE_min.

        Parameters
        ----------
        dE_min : float
            minimum dE.
        """

        lost_index = (self.dE < dE_min)
        self.id[lost_index] = 0

    def particle_decay(self, time: float) -> None:
        """Decreases beam intensity due to the particle decay

        Sets the ratio to a lower value if the particle can decay. Number of macroparticles remains unchanged.

        Parameters
        ----------
        time : float
            time in seconds, which is used to determine the fraction of the
            particle decay
        """
        self.ratio *= np.exp(-time * self.particle.decay_rate / self.gamma)

    def add_particles(self, new_particles: NumpyArray | list[list[float]]) -> None:
        """
        Method to add array of new particles to beam object
        New particles are given id numbers sequential from last id of this beam

        Parameters
        ----------
        new_particles : array-like
            (2, n) array of (dt, dE) for new particles
        """

        try:
            new_dt = new_particles[0]
            new_dE = new_particles[1]
            if len(new_particles) == 3:
                new_weights = new_particles[2]
                assert len(new_weights) == len(new_dt)
            else:
                len_dt = len(new_dt)
                new_weights = None
            if len(new_dt) != len(new_dE):
                raise blond_exceptions.ParticleAdditionError(
                    "new_particles must have equal number of time and energy coordinates")
        except TypeError:
            raise blond_exceptions.ParticleAdditionError(
                "new_particles shape must be (2, n)"
            )

        n_new = len(new_dt)

        self.id = bm.concatenate((self.id, bm.arange(self.n_macroparticles + 1,
                                                     self.n_macroparticles + n_new + 1,
                                                     dtype=int)
                                  ))
        self.n_macroparticles += n_new

        self.dt = bm.concatenate((self.dt, new_dt))
        self.dE = bm.concatenate((self.dE, new_dE))
        if new_weights is not None:
            assert self.weights is not None
            self.weights = bm.concatenate((self.weights, new_weights))
            assert bm.sum(self.weights) == self.n_macroparticles

    def add_beam(self, other_beam: Beam) -> None:
        """
        Method to add the particles from another beam to this beam
        New particles are given id numbers sequential from last id of this beam
        Particles with id == 0 keep id == 0 and are included in addition

        Parameters
        ----------
        other_beam : blond beam object
        """

        if not isinstance(other_beam, type(self)):
            raise TypeError("add_beam method requires a beam object as input")

        self.dt = bm.concatenate((self.dt, other_beam.dt))
        self.dE = bm.concatenate((self.dE, other_beam.dE))
        if other_beam.weights is not None:
            assert self.weights is not None
            self.weights = bm.concatenate((self.weights, other_beam.weights))

        counter = itl.count(self.n_macroparticles + 1)
        newids = bm.zeros(other_beam.n_macroparticles)

        for i in range(other_beam.n_macroparticles):
            if other_beam.id[i]:
                newids[i] = next(counter)
            else:
                next(counter)

        self.id = bm.concatenate((self.id, newids))
        self.n_macroparticles += other_beam.n_macroparticles
        if self.weights is not None:
            assert bm.sum(self.weights) == self.n_macroparticles

    def __iadd__(self, other: Beam | NumpyArray | list[list[float]]) -> Beam:
        """
        Initialisation of in place addition calls add_beam(other) if other
        is a blond beam object, calls add_particles(other) otherwise

        Parameters
        ----------
        other : blond beam object or (2, n) array
        """

        if isinstance(other, type(self)):
            self.add_beam(other)
            return self
        else:
            self.add_particles(other)  # might raise exception on wrong type
            return self

    def split(self, random: bool = False, fast: bool = False):
        """
        MPI ONLY ROUTINE: Splits the beam equally among the workers for
        MPI processing.

        Parameters
        ----------
        random : boolean
            Shuffle the beam before splitting, to be used with the
            approximation methods.
        fast : boolean
            If true, it assumes that every worker has already a copy of the
            beam so only the particle ids are distributed.
            If false, all the coordinates are distributed by the master to all
            the workers.
        """

        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER
        if WORKER.is_master and random:
            bm.random.shuffle(self.id)
            if not fast:
                self.dt = self.dt[self.id - 1]
                self.dE = self.dE[self.id - 1]
                if self.weights is not None:
                    self.weights = self.weights[self.id - 1]

        self.id = WORKER.scatter(self.id)
        if fast:
            self.dt = bm.ascontiguousarray(self.dt[self.id - 1])
            self.dE = bm.ascontiguousarray(self.dE[self.id - 1])
            if self.weights is not None:
                self.weights = bm.ascontiguousarray(self.weights[self.id - 1])
        else:
            self.dt = WORKER.scatter(self.dt)
            self.dE = WORKER.scatter(self.dE)
            if self.weights is not None:
                self.weights = WORKER.scatter(self.weights)

        assert (len(self.dt) == len(self.dE) and len(self.dt) == len(self.id))

        self.n_macroparticles = len(self.dt)
        self._mpi_is_splitted = True

    def gather(self, all_gather: bool = False):
        """
        MPI ONLY ROUTINE: Gather the beam coordinates to the master or all workers.

        Parameters
        ----------
        all_gather : boolean
            If true, every worker will get a copy of the whole beam coordinates.
            If false, only the master will gather the coordinates.
        """
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')
        from ..utils.mpi_config import WORKER

        if all_gather:
            self.dt = WORKER.allgather(self.dt)
            self.dE = WORKER.allgather(self.dE)
            if self.weights is not None:
                self.weights = WORKER.allgather(self.weights)
            self.id = WORKER.allgather(self.id)
            self._mpi_is_splitted = False
        else:
            self.dt = WORKER.gather(self.dt)
            self.dE = WORKER.gather(self.dE)
            if self.weights is not None:
                self.weights = WORKER.gather(self.weights)
            self.id = WORKER.gather(self.id)
            if WORKER.is_master:
                self._mpi_is_splitted = False

        self.n_macroparticles = len(self.dt)

    def gather_statistics(self, all_gather: bool = False):
        """
        MPI ONLY ROUTINE: Gather beam statistics.

        Parameters
        ----------
        all_gather : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        """
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER
        if all_gather:

            self.mean_dt = WORKER.allreduce(
                np.array([self.mean_dt]), operator='mean')[0]

            self.mean_dE = WORKER.allreduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self._mpi_n_total_macroparticles_lost = WORKER.allreduce(
                np.array([self.n_macroparticles_not_alive]), operator='sum')[0]

            # self.__mpi_n_total_macroparticles_alive = WORKER.allreduce(
            # np.array([self.n_macroparticles_alive]), operator='sum')[0]

            self.sigma_dt = WORKER.allreduce(
                np.array([self._mpi_sumsq_dt]), operator='sum')[0]
            self.sigma_dt = np.sqrt(
                self.sigma_dt / (self._mpi_n_total_macroparticles -
                                 self._mpi_n_total_macroparticles_lost)
                - self.mean_dt ** 2)

            self.sigma_dE = WORKER.allreduce(
                np.array([self._mpi_sumsq_dE]), operator='sum')[0]
            self.sigma_dE = np.sqrt(
                self.sigma_dE / (self._mpi_n_total_macroparticles -
                                 self._mpi_n_total_macroparticles_lost)
                - self.mean_dE ** 2)

        else:
            self.mean_dt = WORKER.reduce(
                np.array([self.mean_dt]), operator='mean')[0]

            self.mean_dE = WORKER.reduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self._mpi_n_total_macroparticles_lost = WORKER.reduce(
                np.array([self.n_macroparticles_not_alive]), operator='sum')[0]

            self.sigma_dt = WORKER.reduce(
                np.array([self._mpi_sumsq_dt]), operator='sum')[0]
            self.sigma_dt = np.sqrt(
                self.sigma_dt / (self._mpi_n_total_macroparticles -
                                 self._mpi_n_total_macroparticles_lost)
                - self.mean_dt ** 2)

            self.sigma_dE = WORKER.reduce(
                np.array([self._mpi_sumsq_dE]), operator='sum')[0]
            self.sigma_dE = np.sqrt(
                self.sigma_dE / (self._mpi_n_total_macroparticles -
                                 self._mpi_n_total_macroparticles_lost)
                - self.mean_dE ** 2)

    def gather_losses(self, all_gather: bool = False):
        """
        MPI ONLY ROUTINE: Gather beam losses.

        Parameters
        ----------
        all_gather : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        """
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER

        if all_gather:
            temp = WORKER.allgather(np.array([self.n_macroparticles_lost]))
            self._mpi_n_total_macroparticles_lost = np.sum(temp)
        else:
            temp = WORKER.gather(np.array([self.n_macroparticles_lost]))
            self._mpi_n_total_macroparticles_lost = np.sum(temp)

    def to_gpu(self, recursive=True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return

        self.dE = cp.array(self.dE)
        self.dt = cp.array(self.dt)
        if self.weights is not None:
            self.weights = cp.array(self.weights)
        self.id = cp.array(self.id)

        self._device: DeviceType = 'GPU'

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':  # todo hasattr useless?
            return

        self.dE = cp.asnumpy(self.dE)
        self.dt = cp.asnumpy(self.dt)
        if self.weights is not None:
            self.weights = cp.asnumpy(self.weights)
        self.id = cp.asnumpy(self.id)

        # to make sure it will not be called again
        self._device: DeviceType = 'CPU'

    def dE_mean(self, ignore_id_0: bool = False):
        """Calculate mean of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            mask = self.id > 0
            if self.weights is not None:
                return bm.average(self.dE[mask], weights=self.weights[mask])
            else:
                return bm.mean(self.dE[mask])
        else:
            if self.weights is not None:
                return bm.average(self.dE, weights=self.weights)
            else:
                return bm.mean(self.dE)

    def dE_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            mask = self.id > 0
            if self.weights is not None:
                return mean_and_std(self.dE[mask], self.weights[mask])[1]
            else:
                return bm.std(self.dE[mask])
        else:
            if self.weights is not None:
                return mean_and_std(self.dE, self.weights)[1]
            else:
                return bm.std(self.dE)

    def dt_mean(self, ignore_id_0: bool = False):
        """Calculate mean of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            mask = self.id > 0
            if self.weights is not None:
                return bm.average(self.dt[mask], weights=self.weights[mask])
            else:
                return bm.mean(self.dt[mask])
        else:
            if self.weights is not None:
                return bm.average(self.dt,weights=self.weights)
            else:
                return bm.mean(self.dt)

    def dt_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            mask = self.id > 0
            if self.weights is not None:
                return mean_and_std(self.dt[mask], weights=self.weights[mask])[1]
            else:
                return bm.std(self.dt[mask])
        else:
            if self.weights is not None:
                return mean_and_std(self.dt, weights=self.weights)[1]
            else:
                return bm.std(self.dt)

    def dt_min(self, ignore_id_0: bool = False):
        if ignore_id_0:
            mask = self.id > 0
            return self.dt[mask].min()
        else:
            return self.dt.min()

    def dE_min(self, ignore_id_0: bool = False):
        if ignore_id_0:
            mask = self.id > 0
            return self.dE[mask].min()
        else:
            return self.dE.min()

    def dt_max(self, ignore_id_0: bool = False):
        if ignore_id_0:
            mask = self.id > 0
            return self.dt[mask].max()
        else:
            return self.dt.max()

    def dE_max(self, ignore_id_0: bool = False):
        if ignore_id_0:
            mask = self.id > 0
            return self.dE[mask].max()
        else:
            return self.dE.max()

    def slice_beam(self, profile: NumpyArray | CupyArray,
                   cut_left: float, cut_right: float
                   ):
        bm.slice_beam(
            dt=self.dt,
            profile=profile,
            cut_left=cut_left,
            cut_right=cut_right,
            weights=self.weights
        )
        if bm.in_mpi():
            from ..utils.mpi_config import WORKER

            if WORKER.workers == 1:
                return

            if self._mpi_is_splitted:
                if isinstance(profile, np.ndarray):
                    profile_tmp = profile.view() # guarantee numpy array
                else: # assume is cupy array
                    profile_tmp = cp.asnumpy(profile) # guarantee numpy array
                WORKER.allreduce(profile_tmp) # collect from all workers
                if isinstance(profile, np.ndarray):
                    # write reduce result back
                    # to memory of profile
                    profile[:] = profile_tmp[:]
                else: # assume is cupy array
                    # write reduce
                    # result back to memory of profile
                    profile[:] = cp.array(profile_tmp[:])


    def kick(
        self, rf_station: RFStation, acceleration_kicks: NumpyArray, turn_i: int
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

        bm.kick(
            dt=self.dt,
            dE=self.dE,
            voltage=rf_station.voltage[:, turn_i],
            omega_rf=rf_station.omega_rf[:, turn_i],
            phi_rf=rf_station.phi_rf[:, turn_i],
            charge=rf_station.particle.charge,
            n_rf=rf_station.n_rf,
            acceleration_kick=acceleration_kicks[turn_i]
        )

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
        and is obtained as follow:

        .. math::
            \\delta = \\sqrt{1+\\beta_s^{-2}\\left[\\left(\\frac{\\Delta E}{E_s}\\right)^2 + 2\\frac{\\Delta E}{E_s}\\right]} - 1 \quad \\text{(exact)}

        .. math::
            \\delta = \\frac{\\Delta E}{\\beta_s^2 E_s} \quad \\text{(simple, legacy)}

        """
        bm.drift(
            dE=self.dE,
            dt=self.dt,
            solver=solver,
            t_rev=rf_station.t_rev[turn_i],
            length_ratio=rf_station.length_ratio,
            alpha_order=rf_station.alpha_order,
            eta_0=rf_station.eta_0[turn_i],
            eta_1=rf_station.eta_1[turn_i],
            eta_2=rf_station.eta_2[turn_i],
            alpha_0=rf_station.alpha_0[turn_i],
            alpha_1=rf_station.alpha_1[turn_i],
            alpha_2=rf_station.alpha_2[turn_i],
            beta=rf_station.beta[turn_i],
            energy=rf_station.energy[turn_i])

    def linear_interp_kick(
        self,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ):
        bm.linear_interp_kick(dt=self.dt, dE=self.dE,
                              voltage=voltage,
                              bin_centers=bin_centers,
                              charge=charge,
                              acceleration_kick=acceleration_kick)

    def kickdrift_considering_periodicity(
        self,
        acceleration_kicks: NumpyArray,
        rf_station: RFStation,
        solver: SolverTypes,
        turn_i: int,
    ):
        # Distinguish the particles inside the frame from the particles on
        # the right-hand side of the frame.
        indices_right_outside = \
            bm.where(self.dt > rf_station.t_rev[turn_i + 1])[0]
        indices_inside_frame = \
            bm.where(self.dt < rf_station.t_rev[turn_i + 1])[0]

        if len(indices_right_outside) > 0:
            # Change reference of all the particles on the right of the
            # current frame; these particles skip one kick and drift
            self.dt[indices_right_outside] -= \
                rf_station.t_rev[turn_i + 1]
            # Synchronize the bunch with the particles that are on the
            # RHS of the current frame applying kick and drift to the
            # bunch
            # After that all the particles are in the new updated frame
            insiders_dt = bm.ascontiguousarray(
                self.dt[indices_inside_frame])
            insiders_dE = bm.ascontiguousarray(
                self.dE[indices_inside_frame])
            #kick(insiders_dt, insiders_dE, turn)
            index = turn_i
            bm.kick(insiders_dt, insiders_dE, rf_station.voltage[:, index],
                    rf_station.omega_rf[:, index],
                    rf_station.phi_rf[:, index],
                    rf_station.particle.charge, rf_station.n_rf,
                    acceleration_kicks[index])
            #drift(insiders_dt, insiders_dE, turn + 1)
            self.dt[indices_inside_frame] = insiders_dt
            self.dE[indices_inside_frame] = insiders_dE
            # Check all the particles on the left of the just updated
            # frame and apply a second kick and drift to them with the
            # previous wave after having changed reference.
            indices_left_outside = bm.where(self.dt < 0)[0]
        else:
            #kick(self.dt, self.dE, turn)
            self.kick(rf_station=rf_station,
                      acceleration_kicks=acceleration_kicks, turn_i=turn_i)

            #drift(self.dt, self.dE, turn + 1)
            self.drift(rf_station=rf_station,solver=solver,turn_i=turn_i+1)
            # Check all the particles on the left of the just updated
            # frame and apply a second kick and drift to them with the
            # previous wave after having changed reference.
            indices_left_outside = bm.where(self.dt < 0)[0]
        if len(indices_left_outside) > 0:
            left_outsiders_dt = bm.ascontiguousarray(
                self.dt[indices_left_outside])
            left_outsiders_dE = bm.ascontiguousarray(
                self.dE[indices_left_outside])
            left_outsiders_dt += rf_station.t_rev[turn_i + 1]
            #kick(left_outsiders_dt, left_outsiders_dE, turn)
            index = turn_i
            bm.kick(left_outsiders_dt, left_outsiders_dE, rf_station.voltage[:, index],
                    rf_station.omega_rf[:, index],
                    rf_station.phi_rf[:, index],
                    rf_station.particle.charge, rf_station.n_rf,
                    acceleration_kicks[index])
            # drift(left_outsiders_dt, left_outsiders_dE, turn + 1)
            index = turn_i + 1
            bm.drift(left_outsiders_dt, left_outsiders_dE, solver,
                     rf_station.t_rev[index],
                     rf_station.length_ratio, rf_station.alpha_order,
                     rf_station.eta_0[index], rf_station.eta_1[index],
                     rf_station.eta_2[index],
                     rf_station.alpha_0[index],
                     rf_station.alpha_1[index],
                     rf_station.alpha_2[index],
                     rf_station.beta[index], rf_station.energy[index])
            self.dt[indices_left_outside] = left_outsiders_dt
            self.dE[indices_left_outside] = left_outsiders_dE

    def get_new_beam_with_weights(self,
                                  bins: int | Tuple[int, int],
                                  range: Optional[Tuple[Tuple[float,float], Tuple[float,float]]]=None
    ) -> Beam:
        """Generate beam with weights based on a 2D histogram

        Parameters
        ----------
        bins
            The number of bins for the two dimensions
        range
            The boundaries [[xmin, xmax], [ymin, ymax]]
            """

        H, dt_edges, dE_edges = bm.histogram2d(self.dt,
                                               self.dE,
                                               bins=bins,
                                               range=range,
                                               weights=self.weights,
                                               density=False
                                               )
        dt_centers = (dt_edges[:-1] + dt_edges[1:]) / 2
        dE_centers = (dE_edges[:-1] + dE_edges[1:]) / 2

        dt, dE = bm.meshgrid(dt_centers, dE_centers, indexing='ij')
        assert dE.shape == H.shape
        assert dt.shape == H.shape
        dt = dt.flatten()
        dE = dE.flatten()
        weights = H.flatten()
        select = weights > 0
        dt, dE, weights = dt[select], dE[select], weights[select]

        new_beam = Beam(ring=self._ring,
                        n_macroparticles=len(weights),
                        intensity=self.intensity,
                        dt=dt,
                        dE=dE,
                        weights=weights
                        )
        return new_beam