from __future__ import annotations

from dataclasses import dataclass

from scipy.constants import m_p, c, e


@dataclass(frozen=True)
class ParticleType:
    mass: float
    charge: float


proton = ParticleType(mass=m_p * c**2 / e, charge=1)
