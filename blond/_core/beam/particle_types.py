from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import float32, float64
from scipy.constants import c, e, epsilon_0, hbar, m_e, m_p, physical_constants

from ..backends.backend import backend

m_mu = physical_constants["muon mass"][0]


class ParticleType:
    """
    Represents a particle type with physical constants.

    Parameters
    ----------
    mass : float
        Rest mass energy of the particle, in [eV].
    charge : float
        Number of electric charges of the particle, in []
    user_decay_rate : float, optional
        Optional user-specified decay rate. Default is 0.0.
    """

    def __init__(
        self, mass: float, charge: float, user_decay_rate: float = 0.0
    ):
        self._mass = mass
        self._charge = charge
        self._user_decay_rate = user_decay_rate

        self._mass_inv = 1 / mass

        # classical particle radius [m]
        radius_cl = 0.25 / (np.pi * epsilon_0) * e**2 * charge**2 / (mass * e)
        self._classical_particle_radius = backend.float(radius_cl)

        # Sand's radiation constant [m / eV^3]
        c_gamma = 4 * np.pi / 3 * self._classical_particle_radius / mass**3
        self._sands_radiation_constant = backend.float(c_gamma)

        # Quantum radiation constant [m]
        c_q = 55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (mass * e)
        self._quantum_radiation_constant = backend.float(c_q)

    @property
    def mass(self) -> float:
        """Rest mass energy of the particle, in [eV]."""
        return self._mass

    @property
    def charge(self) -> float:
        """Number of electrons of the particle, in []"""
        return self._charge

    @property
    def user_decay_rate(self) -> float:
        """Optional user-specified decay rate. Default is 0.0."""
        return self._user_decay_rate

    @property
    def mass_inv(self) -> float:
        """Inverse of the mass (1/mass), in [1/eV]."""
        return self._mass_inv

    @property
    def classical_particle_radius(self) -> float32 | float64:
        """Classical particle radius [m]"""
        return self._classical_particle_radius

    @property
    def sands_radiation_constant(self) -> float32 | float64:
        """Sand's radiation constant [ m / eV^3]"""
        return self._sands_radiation_constant

    @property
    def quantum_radiation_constant(self) -> float32 | float64:
        """Quantum radiation constant [m]"""
        return self._quantum_radiation_constant


proton: ParticleType = ParticleType(
    mass=m_p * c**2 / e,
    charge=1,
)

uranium_29: ParticleType = ParticleType(
    mass=238 * m_p * c**2 / e,  # approximate mass-energy in eV
    charge=29,
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
