from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import speed_of_light as c0

from ..base import Preparable, HasPropertyCache
from ..helpers import int_from_float_with_warning
from ..._core.backends.backend import backend
from ..._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from .particle_types import ParticleType
    from ..simulation.simulation import Simulation

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


class BeamFlags(int, Enum):
    LOST = 0
    ACTIVE = 1


class BeamBaseClass(Preparable, HasPropertyCache, ABC):
    def __init__(
        self,
        n_particles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        """Base class to make beam classes

        Parameters
        ----------
        n_particles
            Actual/real number of particles
            a.k.a. beam intensity
        particle_type
            Type of particles, e.g. protons
        is_counter_rotating
            If this is a normal or counter-rotating beam
        is_distributed
            Developer option to allow distributed computing
        """
        super().__init__()

        self.n_particles = int_from_float_with_warning(
            n_particles, warning_stacklevel=2
        )
        self._is_distributed = is_distributed
        self._particle_type = particle_type
        self._is_counter_rotating = is_counter_rotating

        self._dE = None  # should be initialized later using `setup_beam`
        self._dt = None  # should be initialized later using `setup_beam`
        self._flags = None  # should be initialized later using `setup_beam`

        self.reference_time = 0  # todo cached properties
        self._reference_total_energy = 0  # todo cached properties

    @property
    def particle_type(self) -> ParticleType:
        """Type of particles, e.g. protons"""
        return self._particle_type

    @property
    def reference_total_energy(self) -> float:
        """Total beam energy [eV]"""
        return self._reference_total_energy

    @reference_total_energy.setter
    def reference_total_energy(self, reference_total_energy) -> float:
        # in eV and eV/c²
        self._reference_total_energy = reference_total_energy

    @property
    def reference_gamma(self) -> float:
        """Beam reference gamma a.k.a. Lorentz factor []"""
        # in eV and eV/c²
        val = self.reference_total_energy * self.particle_type.mass_inv
        return val

    @property
    def reference_beta(self) -> float:
        """Beam reference fraction of speed of light (v/c0) []"""

        gamma = self.reference_gamma
        val = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        return val

    @property
    def reference_velocity(self) -> float:
        """Beam reference speed [m/s]"""
        return self.reference_beta * c0

    @abstractmethod
    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
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
        """
        pass

    @property  # as readonly attributes
    def is_distributed(self):
        """Developer option to allow distributed computing"""
        return self._is_distributed

    @property  # as readonly attributes
    def is_counter_rotating(self):
        """If this is a normal or counter-rotating beam"""
        return self._is_counter_rotating

    @requires(["EnergyCycleBase"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager"""
        super().on_init_simulation(simulation=simulation)
        self.reference_total_energy = simulation.energy_cycle.total_energy_init

    @requires(["EnergyCycleBase"])
    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """

        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        msg = (
            "Beam was not initialized. Did you forget to call "
            "simulation.on_prepare_beam(...)?"
        )
        assert self._dt is not None, msg
        assert self._dE is not None, msg
        assert self._flags is not None, msg
        index = turn_i_init - 1
        if index < 0:
            new_reference_total_energy = simulation.energy_cycle.total_energy_init
        else:
            new_reference_total_energy = backend.float(
                simulation.energy_cycle.get_target_total_energy(
                    turn_i=index,
                    section_i=0,
                    reference_time=self.reference_time,
                )
            )
        if self.reference_total_energy != new_reference_total_energy:
            msg = (
                f"`Bunch` was prepared for "
                f"total_energy = {self.reference_total_energy} eV, "
                f"but simulation at {turn_i_init=} is "
                f"{simulation.energy_cycle.total_energy_init} eV"
                f"overwriting energy according to simulation."
            )
            warnings.warn(msg)
        self.reference_total_energy = new_reference_total_energy

    @abstractmethod
    def plot_hist2d(self):
        """Plot 2D histogram of beam coordinates"""
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dt_min(self) -> backend.float:
        """Minimum dt coordinate in [s]"""

        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dt_max(self) -> backend.float:
        """Maximum dt coordinate in [s]"""

        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dE_min(self) -> backend.float:
        """Minimum dE coordinate in [eV]"""
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dE_max(self) -> backend.float:
        """Maximum dE coordinate in [eV]"""
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def common_array_size(self) -> int:
        """Size of the beam, considering distributed beams"""
        pass

    cached_props = (
        "dE_min",
        "dE_max",
        "dt_min",
        "dt_max",
        "common_array_size",
    )

    def invalidate_cache_dE(self) -> None:
        """Reset cache of `cached_property` attributes"""
        super()._invalidate_cache(
            (
                "dE_min",
                "dE_max",
            )
        )

    def invalidate_cache_dt(self) -> None:
        """Reset cache of `cached_property` attributes"""
        super()._invalidate_cache(
            (
                "dt_min",
                "dt_max",
            )
        )

    def invalidate_cache(self) -> None:
        """Reset cache of `cached_property` attributes"""
        self._invalidate_cache(BeamBaseClass.cached_props)

    def n_macroparticles_partial(self):
        """Size of the beam, ignoring that beam might be distributed

        Note
        ----
        Depends on `is_distributed`

        """
        return len(self._dE)

    def read_partial_dt(self):
        """Returns dt-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        return self._dt

    def write_partial_dt(self):
        """Returns dt-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`

        """
        self.invalidate_cache_dt()
        return self._dt

    def read_partial_dE(self):
        """Returns dE-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        return self._dE

    def write_partial_dE(self):
        """Returns dE-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        self.invalidate_cache_dE()
        return self._dE

    def write_partial_flags(self):
        """Returns flags-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        self.invalidate_cache_dt()
        self.invalidate_cache_dE()
        return self._flags
