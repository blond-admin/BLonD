from __future__ import annotations

from dataclasses import dataclass, field

from scipy.constants import m_p, c, e


@dataclass(frozen=True)
class ParticleType:
    mass: float
    charge: float
    mass_inv: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'mass_inv', 1 / self.mass)


proton = ParticleType(mass=m_p * c**2 / e, charge=1)
