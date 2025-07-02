from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

from ..base import Preparable, HasPropertyCache
from ..helpers import int_from_float_with_warning
from ..._core.backends.backend import backend

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
        super().__init__()

        self._n_particles__init = int_from_float_with_warning(
            n_particles, warning_stacklevel=2
        )
        self._is_distributed = is_distributed
        self.particle_type = particle_type
        self._dE = None
        self._dt = None
        self._flags = None
        self._is_counter_rotating = is_counter_rotating

    @abstractmethod
    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
    ):
        pass

    @property  # as readonly attributes
    def is_distributed(self):
        return self._is_distributed

    @property  # as readonly attributes
    def is_counter_rotating(self):
        return self._is_counter_rotating

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        msg = (
            "Beam was not initialized. Did you forget to call "
            "simulation.on_prepare_beam(...)?"
        )
        assert self._dt is not None, msg
        assert self._dE is not None, msg
        assert self._flags is not None, msg

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    @abstractmethod
    def plot_hist2d(self):
        pass

    @cached_property

    @abstractmethod  # as readonly attributes
    def dt_min(self) -> backend.float:
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dt_max(self) -> backend.float:
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dE_min(self) -> backend.float:
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def dE_max(self) -> backend.float:
        pass

    @cached_property
    @abstractmethod  # as readonly attributes
    def common_array_size(self) -> int:
        pass

    cached_props = (
        "dE_min",
        "dE_max",
        "dt_min",
        "dt_max",
        "common_array_size",
    )

    def invalidate_cache_dE(self) -> None:
        super()._invalidate_cache(
            (
                "dE_min",
                "dE_max",
            )
        )

    def invalidate_cache_dt(self) -> None:
        super()._invalidate_cache((
                "dt_min",
                "dt_max",
            ))

    def invalidate_cache(self) -> None:
        self._invalidate_cache(BeamBaseClass.cached_props)

    def n_macroparticles_partial(self):
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
        Depends on"""
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
