from __future__ import annotations

from typing import Optional as LateInit, TYPE_CHECKING

from numpy.typing import NDArray as NumpyArray

from .base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from ..beam.particle_types import ParticleType


class Beam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ):
        super().__init__(
            n_particles=n_particles,
            n_macroparticles=n_macroparticles,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
        )


class WeightenedBeam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
    ):
        super().__init__(n_particles, n_macroparticles, particle_type)
        self._weights: LateInit[NumpyArray] = None

    @staticmethod
    def from_beam(beam: Beam):
        pass
