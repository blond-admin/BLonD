from __future__ import annotations

from typing import Optional as LateInit, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .base import BeamBaseClass, BeamFlags
from ..backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ..beam.particle_types import ParticleType

    from ... import Simulation


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

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: Optional[NumpyArray | CupyArray] = None,
    ):
        assert len(dt) == len(dE)
        n_particles = len(dt)
        if flags is None:
            flags = BeamFlags.ACTIVE.value * np.ones(n_particles, dtype=backend.int)
        else:
            assert flags.max() <= BeamFlags.ACTIVE.value

        self._dE = dE.astype(backend.float)
        self._dt = dt.astype(backend.float)
        self._flags = flags.astype(backend.int)


    def on_run_simulation(self, simulation: Simulation, n_turns: int,
                          turn_i_init: int) -> None:
        pass

    @property
    def dt_min(self) -> backend.float:
        pass

    @property
    def dt_max(self) -> backend.float:
        pass

    @property
    def dE_min(self) -> backend.float:
        pass

    @property
    def common_array_size(self) -> int:
        return len(self._dt)

    def invalidate_cache_dE(self) -> None:
        pass

    def invalidate_cache_dt(self) -> None:
        pass

    def invalidate_cache(self) -> None:
        pass

    def plot_hist2d(self, **kwargs):
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs.keys():
            kwargs["bins"] = 256
        plt.hist2d(self._dt, self._dE, **kwargs)

class WeightenedBeam(Beam):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
    ):
        super().__init__(n_particles, n_macroparticles, particle_type)
        self._weights: LateInit[NumpyArray] = None

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: Optional[NumpyArray | CupyArray] = None,
        weights: NumpyArray | CupyArray = None,
    ):
        assert weights is not None
        assert len(dt) == len(weights)
        super().setup_beam(dt=dt, dE=dE, flags=flags)
        self._weights = weights.astype(backend.int)

    @staticmethod
    def from_beam(beam: Beam):
        pass
