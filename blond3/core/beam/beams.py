from __future__ import annotations

from typing import Optional as LateInit, TYPE_CHECKING

from .base import BeamBaseClass
from ..backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

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

    def setup_beam(self, dt: NumpyArray | CupyArray, dE: NumpyArray | CupyArray, flags: NumpyArray | CupyArray):
        self._dE = dE.astype(backend.float)
        self._dt = dt.astype(backend.float)
        self._flags = flags.astype(backend.int)


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
