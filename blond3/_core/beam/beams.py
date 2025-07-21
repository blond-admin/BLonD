from __future__ import annotations

from functools import cached_property
from typing import Optional as LateInit, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .base import BeamBaseClass, BeamFlags
from ..backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Type
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ..beam.particle_types import ParticleType

    from ... import Simulation


class Beam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ):
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
    ):
        """Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
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
            flags = BeamFlags.ACTIVE.value * np.ones(n_particles, dtype=backend.int)
        else:
            assert flags.max() <= BeamFlags.ACTIVE.value
            assert len(dt) == len(flags)

        self._dE = dE.astype(backend.float)
        self._dt = dt.astype(backend.float)
        self._flags = flags.astype(backend.int)
        if reference_time:
            self.reference_time = reference_time
        if reference_total_energy:
            self.reference_total_energy = reference_total_energy
        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
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

    def plot_hist2d(self, **kwargs):
        """Plot 2D histogram of beam coordinates"""
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs.keys():
            kwargs["bins"] = 256
        plt.hist2d(self._dt, self._dE, **kwargs)


class ProbeBeam(Beam):
    def __init__(
        self,
        particle_type: ParticleType,
        dt: Optional[NumpyArray] = None,
        dE: Optional[NumpyArray] = None,
        reference_time: Optional[float] = None,
        reference_total_energy: Optional[float] = None,
    ):
        """
        Test Bunch without intensity effects

        Parameters
        ----------
        particle_type
            Type of particles, e.g. protons
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
        """
        super().__init__(
            n_particles=0,
            particle_type=particle_type,
        )
        if dt is not None:
            dE = np.zeros_like(dt)
        if dE is not None:
            dt = np.zeros_like(dE)
        if (dE is None) and (dt is None):
            raise ValueError("dE or dt must be given!")
        self.setup_beam(
            dt=dt,
            dE=dE,
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
        )


class WeightenedBeam(Beam):
    def __init__(
        self,
        n_particles: int | float,
        particle_type: ParticleType,
    ):
        raise NotImplementedError  # todo
        super().__init__(n_particles, particle_type)
        self._weights: LateInit[NumpyArray] = None

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: Optional[NumpyArray | CupyArray] = None,
        weights: NumpyArray | CupyArray = None,
        reference_time: Optional[float] = None,
        reference_total_energy: Optional[float] = None,
    ):
        """Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
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
