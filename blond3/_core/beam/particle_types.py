from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e, m_p, physical_constants

m_mu = physical_constants["muon mass"][0]


@dataclass(frozen=True)
class ParticleType:
    mass: float
    charge: float
    user_decay_rate: float = 0.0
    mass_inv: float = field(init=False)
    _classical_particle_radius: float = field(init=False)
    _sands_radiation_constant: float = field(init=False)
    _quantum_radiation_constant: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "mass_inv", 1 / self.mass)

        # classical particle radius [m]
        radius_cl = 0.25 / (np.pi * epsilon_0) * e**2 * self.charge**2 / (self.mass * e)
        object.__setattr__(self, "_classical_particle_radius", radius_cl)

        # Sand's radiation constant [ m / eV^3]
        c_gamma = 4 * np.pi / 3 * self._classical_particle_radius / self.mass**3
        object.__setattr__(self, "_sands_radiation_constant", c_gamma)

        # Quantum radiation constant [m]
        c_q = 55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (self.mass * e)
        object.__setattr__(self, "_quantum_radiation_constant", c_q)

    # The properties below are just used to allow displaying the info strings
    # in IDEs like PyCharm

    @property
    def classical_particle_radius(self):
        """Classical particle radius [m]"""
        return self._classical_particle_radius

    @property
    def sands_radiation_constant(self):
        """Sand's radiation constant [ m / eV^3]"""
        return self._sands_radiation_constant

    @property
    def quantum_radiation_constant(self):
        """Quantum radiation constant [m]"""
        return self._quantum_radiation_constant


proton = ParticleType(
    mass=m_p * c**2 / e,
    charge=1,
)

uranium_29 = ParticleType(
    mass=238 * m_p * c**2 / e,  # approximate mass-energy in eV
    charge=29,
)

electron = ParticleType(
    mass=m_e * c**2 / e,
    charge=-1,
)

positron = ParticleType(
    mass=m_e * c**2 / e,
    charge=1,
)

_muon_decay_rate = float(1 / 2.1969811e-6)

mu_plus = ParticleType(
    mass=m_mu * c**2 / e,
    charge=1,
    user_decay_rate=_muon_decay_rate,
)


mu_minus = ParticleType(
    mass=m_mu * c**2 / e,
    charge=-1,
    user_decay_rate=_muon_decay_rate,
)
