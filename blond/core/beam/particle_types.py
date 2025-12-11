# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Pre-defined particle types, such as `proton`."""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
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
    pass


@dc.dataclass(frozen=True, eq=True)
class ParticleType:
    """
    Represents a particle type with physical constants.

    Parameters
    ----------
    mass
        Rest mass energy of the particle, in [eV].
    charge
        Number of electric charges of the particle, in [].
    user_decay_rate
        Optional user-specified decay rate. Default is 0.0.
    mass_inv
        Inverse of the mass (1/mass), in [1/eV].
    classical_particle_radius
        Classical particle radius [m].
    sands_radiation_constant
        Sand's radiation constant [ m / eV^3].
    quantum_radiation_constant
        Quantum radiation constant [m].

    Attributes
    ----------
    mass
        Rest mass energy of the particle, in [eV].
    charge
        Number of electric charges of the particle, in [].
    user_decay_rate
        Optional user-specified decay rate. Default is 0.0.
    mass_inv
        Inverse of the mass (1/mass), in [1/eV].
    classical_particle_radius
        Classical particle radius [m].
    sands_radiation_constant
        Sand's radiation constant [ m / eV^3].
    quantum_radiation_constant
        Quantum radiation constant [m].
    """

    mass: float
    charge: int
    user_decay_rate: float = 0

    mass_inv: float = dc.field(init=False)
    classical_particle_radius: float = dc.field(init=False)
    sands_radiation_constant: float = dc.field(init=False)
    quantum_radiation_constant: float = dc.field(init=False)

    def __post_init__(self):
        """Complete the setup of the particle definition."""
        object.__setattr__(self, "mass_inv", 1 / self.mass)

        # classical particle radius [m]
        radius_cl = (
            0.25
            / (np.pi * epsilon_0)
            * e**2
            * self.charge**2
            / (self.mass * e)
        )
        object.__setattr__(self, "classical_particle_radius", radius_cl)

        # Sand's radiation constant [m / eV^3]
        c_gamma = 4 * np.pi / 3 * self.classical_particle_radius / self.mass**3
        object.__setattr__(self, "sands_radiation_constant", c_gamma)

        # Quantum radiation constant [m]
        c_q = 55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (self.mass * e)
        object.__setattr__(self, "quantum_radiation_constant", c_q)


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
