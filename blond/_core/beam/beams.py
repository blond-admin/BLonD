from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ..._generals.cupy.no_cupy_import import is_cupy_array
from ..backends.backend import backend
from .base import BeamBaseClass, BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from ... import Simulation
    from ..beam.particle_types import ParticleType


class Beam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ) -> None:
        """Base class to host particle coordinates and timing information

        Parameters
        ----------
        n_particles
            Actual/real number of particles
            a.k.a. beam intensity
        particle_type
            Type of particles, e.g. protons
        is_counter_rotating
            If this is a normal or counter-rotating beam
        """
        super().__init__(
            n_particles=n_particles,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: Optional[NumpyArray | CupyArray] = None,
        reference_time: Optional[float] = None,
        reference_total_energy: Optional[float] = None,
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
        assert len(dt) == len(dE), f"{len(dt)} != {len(dE)}"
        n_particles = len(dt)
        if flags is None:
            flags = backend.int(BeamFlags.ACTIVE.value) * backend.ones(
                n_particles, dtype=backend.int
            )
        else:
            assert flags.max() <= BeamFlags.ACTIVE.value
            assert len(dt) == len(flags)

        self._dE: NumpyArray | CupyArray = backend.array(
            dE, dtype=backend.float
        )
        self._dt: NumpyArray | CupyArray = backend.array(
            dt, dtype=backend.float
        )
        self._flags: NumpyArray | CupyArray = flags.astype(backend.int)
        if reference_time:
            self.reference_time = backend.float(reference_time)
        if reference_total_energy:
            self.reference_total_energy = reference_total_energy
        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @cached_property
    def ratio(self) -> float:
        """Ratio of the intensity vs. the sum of weights"""
        # As there are no weights, lets assume all weights are 1,
        # The sum over all macro-particles with weight 1
        # is thus `common_array_size`.
        return self.n_particles / self.common_array_size

    @cached_property
    def dt_min(self) -> backend.float:
        """Minimum dt coordinate, in [s]"""

        return self._dt.min()

    @cached_property
    def dt_max(self) -> backend.float:
        """Maximum dt coordinate, in [s]"""

        return self._dt.max()

    @cached_property
    def dE_min(self) -> backend.float:
        """Minimum dE coordinate, in [eV]"""

        return self._dE.min()

    @cached_property
    def dE_max(self) -> backend.float:
        """Maximum dE coordinate, in [eV]"""

        return self._dE.max()

    @cached_property
    def common_array_size(self) -> int:
        """Size of the beam, considering distributed beams"""

        return len(self._dt)

    def plot_hist2d(self, **kwargs) -> None:
        """Plot 2D histogram of beam coordinates"""
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs.keys():
            kwargs["bins"] = 256
        if is_cupy_array(self._dt):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE
            dt: CupyArray = self._dt
            plt.hist2d(dt.get(), dE.get(), **kwargs)
        else:
            plt.hist2d(self._dt, self._dE, **kwargs)


class ProbeBeam(Beam):
    def __init__(
        self,
        particle_type: ParticleType,
        dt: Optional[NumpyArray] = None,
        dE: Optional[NumpyArray] = None,
        reference_time: Optional[float] = None,
        reference_total_energy: Optional[float] = None,
    ) -> None:
        """
        Test Bunch without intensity effects

        Parameters
        ----------
        particle_type
            Type of particles, e.g. protons
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        """
        super().__init__(
            n_particles=0,
            particle_type=particle_type,
        )
        if dt is not None:
            dE = np.zeros_like(dt)
        elif dE is not None:
            dt = np.zeros_like(dE)
        elif (dE is None) and (dt is None):
            raise ValueError("dE or dt must be given!")
        else:
            raise RuntimeError(f"{dE=} {dt=}")

        self.setup_beam(
            dt=dt,
            dE=dE,
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
        )
