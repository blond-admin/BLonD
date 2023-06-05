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

from __future__ import division

import itertools as itl

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e, m_p

from ..trackers.utilities import is_in_separatrix
from ..utils import bmath as bm
from ..utils import exceptions as blExcept


class Particle:
    r"""Class containing basic parameters, e.g. mass, of the particles to be tracked.

    The following particles are already implemented: proton, electron, positron

    Parameters
    ----------
    user_mass : float
        Energy equivalent of particle rest mass in eV
    user_charge : float
        Particle charge in units of the elementary charge

    Attributes
    ----------
    mass : float
        Energy equivalent of particle rest mass in eV.
    charge : float
        Particle charge in units of the elementary charge.
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

    def __init__(self, user_mass, user_charge):

        if user_mass > 0.:
            self.mass = float(user_mass)
            self.charge = float(user_charge)
        else:
            # MassError
            raise RuntimeError('ERROR: Particle mass not recognized!')

        # classical particle radius [m]
        self.radius_cl = 0.25 / (np.pi * epsilon_0) * \
            e**2 * self.charge**2 / (self.mass * e)

        # Sand's radiation constant [ m / eV^3]
        self.c_gamma = 4 * np.pi / 3 * self.radius_cl / self.mass**3

        # Quantum radiation constant [m]
        self.c_q = (55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (self.mass * e))


class Proton(Particle):
    """ Implements a proton `Particle`.
    """

    def __init__(self):

        Particle.__init__(self, m_p * c**2 / e, 1)


class Electron(Particle):
    """ Implements an electron `Particle`.
    """

    def __init__(self):
        Particle.__init__(self, m_e * c**2 / e, -1)


class Positron(Particle):
    """ Implements a positron `Particle`.
    """

    def __init__(self):

        Particle.__init__(self, m_e * c**2 / e, 1)


class Beam:
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
    Ring : Ring
        Used to import different quantities such as the mass and the energy.
    n_macroparticles : int
        total number of macroparticles.
    intensity : float
        total intensity of the beam (in number of charge).

    Attributes
    ----------
    mass : float
        mass of the particle [eV].
    charge : int
        integer charge of the particle [e].
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
    >>> ring = Ring(n_turns, C, eta, momentum, 'proton')
    >>> n_macroparticle = 1e6
    >>> intensity = 1e11
    >>>
    >>> my_beam = Beam(ring, n_macroparticle, intensity)
    """

    def __init__(self, Ring, n_macroparticles, intensity):

        self.Particle = Ring.Particle
        self.beta = Ring.beta[0][0]
        self.gamma = Ring.gamma[0][0]
        self.energy = Ring.energy[0][0]
        self.momentum = Ring.momentum[0][0]
        self.dt = np.zeros([int(n_macroparticles)], dtype=bm.precision.real_t)
        self.dE = np.zeros([int(n_macroparticles)], dtype=bm.precision.real_t)
        self.mean_dt = 0.
        self.mean_dE = 0.
        self.sigma_dt = 0.
        self.sigma_dE = 0.
        self.intensity = float(intensity)
        self.n_macroparticles = int(n_macroparticles)
        self.ratio = self.intensity / self.n_macroparticles
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)
        self.epsn_rms_l = 0.
        # For MPI
        self.n_total_macroparticles_lost = 0
        self.n_total_macroparticles = n_macroparticles
        self.is_splitted = False
        self._sumsq_dt = 0.
        self._sumsq_dE = 0.
        # For GPU
        self._device = 'CPU'

    @property
    def n_macroparticles_lost(self):
        '''Number of lost macro-particles, defined as @property.

        Returns
        -------
        n_macroparticles_lost : int
            number of macroparticles lost.

        '''
        return self.n_macroparticles - self.n_macroparticles_alive

    @property
    def n_macroparticles_alive(self):
        '''Number of transmitted macro-particles, defined as @property.

        Returns
        -------
        n_macroparticles_alive : int
            number of macroparticles not lost.

        '''

        return bm.count_nonzero(self.id)

    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays
        """

        indexalive = np.where(self.id != 0)[0]
        if len(indexalive) > 0:
            self.dt = np.ascontiguousarray(
                self.dt[indexalive], dtype=bm.precision.real_t)
            self.dE = np.ascontiguousarray(
                self.dE[indexalive], dtype=bm.precision.real_t)
            self.n_macroparticles = len(self.dt)
            self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)
        else:
            # AllParticlesLost
            raise RuntimeError("ERROR in Beams: all particles lost and" +
                               " eliminated!")

    def statistics(self):
        r'''
        Calculation of the mean and standard deviation of beam coordinates,
        as well as beam emittance using different definitions.
        Take no arguments, statistics stored in

        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        '''

        # Statistics only for particles that are not flagged as lost
        itemindex = bm.nonzero(self.id)[0]
        self.mean_dt = bm.mean(self.dt[itemindex])
        self.sigma_dt = bm.std(self.dt[itemindex])
        self._sumsq_dt = bm.dot(self.dt[itemindex], self.dt[itemindex])
        # self.min_dt = bm.min(self.dt[itemindex])
        # self.max_dt = bm.max(self.dt[itemindex])

        self.mean_dE = bm.mean(self.dE[itemindex])
        self.sigma_dE = bm.std(self.dE[itemindex])
        self._sumsq_dE = bm.dot(self.dE[itemindex], self.dE[itemindex])

        # self.min_dE = bm.min(self.dE[itemindex])
        # self.max_dE = bm.max(self.dE[itemindex])

        # R.m.s. emittance in Gaussian approximation
        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs

    def losses_separatrix(self, Ring, RFStation):
        '''Beam losses based on separatrix.

        Set to 0 all the particle's id not in the separatrix anymore.

        Parameters
        ----------
        Ring : Ring
            Used to call the function is_in_separatrix.
        RFStation : RFStation
            Used to call the function is_in_separatrix.
        '''

        lost_index = is_in_separatrix(Ring, RFStation, self,
                                      self.dt, self.dE) == False

        self.id[lost_index] = 0

    def losses_longitudinal_cut(self, dt_min, dt_max):
        '''Beam losses based on longitudinal cuts.

        Set to 0 all the particle's id with dt not in the interval
        (dt_min, dt_max).

        Parameters
        ----------
        dt_min : float
            minimum dt.
        dt_max : float
            maximum dt.
        '''

        lost_index = (self.dt < dt_min) | (self.dt > dt_max)
        self.id[lost_index] = 0

    def losses_energy_cut(self, dE_min, dE_max):
        '''Beam losses based on energy cuts, e.g. on collimators.

        Set to 0 all the particle's id with dE not in the interval (dE_min, dE_max).

        Parameters
        ----------
        dE_min : float
            minimum dE.
        dE_max : float
            maximum dE.
        '''

        lost_index = (self.dE < dE_min) | (self.dE > dE_max)
        self.id[lost_index] = 0

    def losses_below_energy(self, dE_min):
        '''Beam losses based on lower energy cut.

        Set to 0 all the particle's id with dE below dE_min.

        Parameters
        ----------
        dE_min : float
            minimum dE.
        '''

        lost_index = (self.dE < dE_min)
        self.id[lost_index] = 0

    def add_particles(self, new_particles):
        '''
        Method to add array of new particles to beam object
        New particles are given id numbers sequential from last id of this beam

        Parameters
        ----------
        new_particles : array-like
            (2, n) array of (dt, dE) for new particles
        '''

        try:
            newdt = new_particles[0]
            newdE = new_particles[1]
            if len(newdt) != len(newdE):
                raise blExcept.ParticleAdditionError(
                    "new_particles must have equal number of time and energy coordinates")
        except TypeError:
            raise blExcept.ParticleAdditionError(
                "new_particles shape must be (2, n)")

        nNew = len(newdt)

        self.id = bm.concatenate((self.id, bm.arange(self.n_macroparticles + 1,
                                                     self.n_macroparticles
                                                     + nNew + 1, dtype=int)))
        self.n_macroparticles += nNew

        self.dt = bm.concatenate((self.dt, newdt))
        self.dE = bm.concatenate((self.dE, newdE))

    def add_beam(self, other_beam):
        '''
        Method to add the particles from another beam to this beam
        New particles are given id numbers sequential from last id of this beam
        Particles with id == 0 keep id == 0 and are included in addition

        Parameters
        ----------
        other_beam : blond beam object
        '''

        if not isinstance(other_beam, type(self)):
            raise TypeError("add_beam method requires a beam object as input")

        self.dt = bm.concatenate((self.dt, other_beam.dt))
        self.dE = bm.concatenate((self.dE, other_beam.dE))

        counter = itl.count(self.n_macroparticles + 1)
        newids = bm.zeros(other_beam.n_macroparticles)

        for i in range(other_beam.n_macroparticles):
            if other_beam.id[i]:
                newids[i] = next(counter)
            else:
                next(counter)

        self.id = bm.concatenate((self.id, newids))
        self.n_macroparticles += other_beam.n_macroparticles

    def __iadd__(self, other):
        '''
        Initialisation of in place addition calls add_beam(other) if other
        is a blond beam object, calls add_particles(other) otherwise

        Parameters
        ----------
        other : blond beam object or (2, n) array
        '''

        if isinstance(other, type(self)):
            self.add_beam(other)
            return self

        self.add_particles(other)
        return self

    def split(self, random=False, fast=False):
        '''
        MPI ONLY ROUTINE: Splits the beam equally among the workers for
        MPI processing.

        Parameters
        ----------
        random : boolean
            Shuffle the beam before splitting, to be used with the
            approximation methonds.
        fast : boolean
            If true, it assumes that every worker has already a copy of the
            beam so only the particle ids are distributed.
            If false, all the coordinates are distributed by the master to all
            the workers.
        '''

        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER
        if WORKER.is_master and random:
            bm.random.shuffle(self.id)
            if not fast:
                self.dt = self.dt[self.id - 1]
                self.dE = self.dE[self.id - 1]

        self.id = WORKER.scatter(self.id)
        if fast:
            self.dt = bm.ascontiguousarray(self.dt[self.id - 1])
            self.dE = bm.ascontiguousarray(self.dE[self.id - 1])
        else:
            self.dt = WORKER.scatter(self.dt)
            self.dE = WORKER.scatter(self.dE)

        assert (len(self.dt) == len(self.dE) and len(self.dt) == len(self.id))

        self.n_macroparticles = len(self.dt)
        self.is_splitted = True

    def gather(self, all_gather=False):
        '''
        MPI ONLY ROUTINE: Gather the beam coordinates to the master or all workers.

        Parameters
        ----------
        all_gather : boolean
            If true, every worker will get a copy of the whole beam coordinates.
            If false, only the master will gather the coordinates.
        '''
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')
        from ..utils.mpi_config import WORKER

        if all_gather:
            self.dt = WORKER.allgather(self.dt)
            self.dE = WORKER.allgather(self.dE)
            self.id = WORKER.allgather(self.id)
            self.is_splitted = False
        else:
            self.dt = WORKER.gather(self.dt)
            self.dE = WORKER.gather(self.dE)
            self.id = WORKER.gather(self.id)
            if WORKER.is_master:
                self.is_splitted = False

        self.n_macroparticles = len(self.dt)

    def gather_statistics(self, all_gather=False):
        '''
        MPI ONLY ROUTINE: Gather beam statistics.

        Parameters
        ----------
        all_gather : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        '''
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER
        if all_gather:

            self.mean_dt = WORKER.allreduce(
                np.array([self.mean_dt]), operator='mean')[0]

            self.mean_dE = WORKER.allreduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self.n_total_macroparticles_lost = WORKER.allreduce(
                np.array([self.n_macroparticles_lost]), operator='sum')[0]

            # self.n_total_macroparticles_alive = WORKER.allreduce(
            # np.array([self.n_macroparticles_alive]), operator='sum')[0]

            self.sigma_dt = WORKER.allreduce(
                np.array([self._sumsq_dt]), operator='sum')[0]
            self.sigma_dt = np.sqrt(
                self.sigma_dt / (self.n_total_macroparticles -
                                 self.n_total_macroparticles_lost)
                - self.mean_dt**2)

            self.sigma_dE = WORKER.allreduce(
                np.array([self._sumsq_dE]), operator='sum')[0]
            self.sigma_dE = np.sqrt(
                self.sigma_dE / (self.n_total_macroparticles -
                                 self.n_total_macroparticles_lost)
                - self.mean_dE**2)

        else:
            self.mean_dt = WORKER.reduce(
                np.array([self.mean_dt]), operator='mean')[0]

            self.mean_dE = WORKER.reduce(
                np.array([self.mean_dE]), operator='mean')[0]

            self.n_total_macroparticles_lost = WORKER.reduce(
                np.array([self.n_macroparticles_lost]), operator='sum')[0]

            self.sigma_dt = WORKER.reduce(
                np.array([self._sumsq_dt]), operator='sum')[0]
            self.sigma_dt = np.sqrt(
                self.sigma_dt / (self.n_total_macroparticles -
                                 self.n_total_macroparticles_lost)
                - self.mean_dt**2)

            self.sigma_dE = WORKER.reduce(
                np.array([self._sumsq_dE]), operator='sum')[0]
            self.sigma_dE = np.sqrt(
                self.sigma_dE / (self.n_total_macroparticles -
                                 self.n_total_macroparticles_lost)
                - self.mean_dE**2)

    def gather_losses(self, all_gather=False):
        '''
        MPI ONLY ROUTINE: Gather beam losses.

        Parameters
        ----------
        all_gather : boolean
            if true, all workers will gather the beam stats.
            If false, only the master will get the beam stats.
        '''
        if not bm.in_mpi():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import WORKER

        if all_gather:
            temp = WORKER.allgather(np.array([self.n_macroparticles_lost]))
            self.n_total_macroparticles_lost = np.sum(temp)
        else:
            temp = WORKER.gather(np.array([self.n_macroparticles_lost]))
            self.n_total_macroparticles_lost = np.sum(temp)

    def to_gpu(self, recursive=True):
        '''
        Transfer all necessary arrays to the GPU
        '''
        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return

        import cupy as cp
        self.dE = cp.array(self.dE)
        self.dt = cp.array(self.dt)
        self.id = cp.array(self.id)

        self._device = 'GPU'

    def to_cpu(self, recursive=True):
        '''
        Transfer all necessary arrays back to the CPU
        '''
        # Check if to_cpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':
            return

        import cupy as cp
        self.dE = cp.asnumpy(self.dE)
        self.dt = cp.asnumpy(self.dt)
        self.id = cp.asnumpy(self.id)

        # to make sure it will not be called again
        self._device = 'CPU'
