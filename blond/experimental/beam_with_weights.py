from __future__ import annotations

from cupy.typing import NDArray as CupyArray
from numpy._typing import NDArray as NumpyArray

from blond import Beam
from blond._core.backends.backend import backend
from blond._core.beam.particle_types import ParticleType


class WeightenedBeam(Beam):
    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
    ) -> None:
        raise NotImplementedError  # todo
        super().__init__(intensity, particle_type)
        self._weights: NumpyArray | None = None

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray | None = None,
        weights: NumpyArray | CupyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
    ) -> None:
        """Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        flags
            Macro-particle flags
        reference_time
            Time of the reference frame (global time), in [s]
        reference_total_energy
            Time of the reference frame (global total energy), in [eV]
        """
        assert weights is not None
        assert len(dt) == len(weights)
        super().setup_beam(dt=dt, dE=dE, flags=flags)
        self._weights = weights.astype(backend.int)

    @staticmethod
    def from_beam(beam: Beam):
        pass
