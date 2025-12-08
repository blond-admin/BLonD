# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Pre-defined particle types, such as `proton`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import float32, float64
from scipy.constants import (  # type: ignore[import-untyped]
    c,
    e,
    epsilon_0,
    hbar,
    m_e,
    m_p,
    physical_constants,
)

m_mu = physical_constants["muon mass"][0]

if TYPE_CHECKING:
    from typing import Self


class ParticleType:
    """
    Represents a particle type with physical constants.

    Parameters
    ----------
    mass : float
        Rest mass energy of the particle, in [eV].
    charge : float
        Number of electric charges of the particle, in [].
    user_decay_rate : float, optional
        Optional user-specified decay rate. Default is 0.0.
    """

    def __init__(
        self, mass: float, charge: float, user_decay_rate: float = 0.0
    ):
        self._mass = float(mass)
        self._charge = float(charge)
        self._user_decay_rate = float(user_decay_rate)

        self._mass_inv = 1 / mass

        # classical particle radius [m]
        radius_cl = 0.25 / (np.pi * epsilon_0) * e**2 * charge**2 / (mass * e)
        self._classical_particle_radius = radius_cl

        # Sand's radiation constant [m / eV^3]
        c_gamma = 4 * np.pi / 3 * self._classical_particle_radius / mass**3
        self._sands_radiation_constant = c_gamma

        # Quantum radiation constant [m]
        c_q = 55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (mass * e)
        self._quantum_radiation_constant = c_q

    def __eq__(self, other: Self) -> bool:
        """
        Equality comparison of the paticle.

        Compares with another ParticleType object to ensure they have
        the same value.  The values of mass, charge, decay rate and
        particle radius are compared.

        Args:
            other: The ParticleType instance to compare to.

        Returns
        -------
            bool: True if both ParticleTypes are the same.
        """
        other_tuple = (
            other._mass,
            other._charge,
            other._user_decay_rate,
            other._classical_particle_radius,
        )
        self_tuple = (
            self._mass,
            self._charge,
            self._user_decay_rate,
            self._classical_particle_radius,
        )

        return other_tuple == self_tuple

    def __hash__(self):
        """
        Compute the hash of the particle.

        Compares the hash value of the particle.  Uses the hash of a
        tuple of (mass, charge, decay rate, particle radius).

        Returns
        -------
            hash: The computed hash value
        """
        return hash(
            (
                self._mass,
                self._charge,
                self._user_decay_rate,
                self._classical_particle_radius,
            )
        )

    @property
    def mass(self) -> float:
        """
        Rest mass energy of the particle, in [eV].

        Returns
        -------
        mass
            Rest mass energy of the particle, in [eV].
        """
        return self._mass

    @property
    def charge(self) -> float:
        """
        Number of electrons of the particle, unitless.

        Returns
        -------
        charge
            Number of electrons of the particle, unitless.
        """
        return self._charge

    @property
    def user_decay_rate(self) -> float:
        """
        Optional user-specified decay rate. Default is 0.0.

        Returns
        -------
        user_decay_rate
            Optional user-specified decay rate. Default is 0.0.
        """
        return self._user_decay_rate

    @property
    def mass_inv(self) -> float:
        """
        Inverse of the mass (1/mass), in [1/eV].

        Returns
        -------
        mass_inv
            Inverse of the mass (1/mass), in [1/eV].
        """
        return self._mass_inv

    @property
    def classical_particle_radius(self) -> float:
        """
        Classical particle radius [m].

        Returns
        -------
        classical_particle_radius
            Classical particle radius [m].
        """
        return self._classical_particle_radius

    @property
    def sands_radiation_constant(self) -> float:
        """
        Return Sand's radiation constant [ m / eV^3].

        Returns
        -------
        sands_radiation_constant
            Sand's radiation constant [ m / eV^3].
        """
        return self._sands_radiation_constant

    @property
    def quantum_radiation_constant(self) -> float32 | float64:
        """
        Quantum radiation constant [m].

        Returns
        -------
        quantum_radiation_constant
            Quantum radiation constant [m].
        """
        return self._quantum_radiation_constant


proton: ParticleType = ParticleType(
    mass=m_p * c**2 / e,
    charge=1,
)

uranium_29: ParticleType = ParticleType(
    mass=238 * m_p * c**2 / e,  # approximate mass-energy in eV
    charge=29,
)

lead_82: ParticleType = ParticleType(
    mass=207.93 * m_p * c**2 / e,  # approximate mass-energy in eV
    charge=82,
)

electron: ParticleType = ParticleType(
    mass=m_e * c**2 / e,
    charge=-1,
)

positron: ParticleType = ParticleType(
    mass=m_e * c**2 / e,
    charge=1,
)

_muon_decay_rate = float(1 / 2.1969811e-6)

mu_plus: ParticleType = ParticleType(
    mass=m_mu * c**2 / e,
    charge=1,
    user_decay_rate=_muon_decay_rate,
)


mu_minus: ParticleType = ParticleType(
    mass=m_mu * c**2 / e,
    charge=-1,
    user_decay_rate=_muon_decay_rate,
)
