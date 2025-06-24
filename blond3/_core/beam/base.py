from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from ..base import Preparable
from ..helpers import int_from_float_with_warning
from ..._core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from .particle_types import ParticleType
    from ..simulation.simulation import Simulation

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArrayprint(


class BeamFlags(int, Enum):
    LOST = 0
    ACTIVE = 1


class BeamBaseClass(Preparable, ABC):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        super().__init__()

        self._n_particles__init = int_from_float_with_warning(
            n_particles, warning_stacklevel=2
        )
        self._n_macroparticles__init = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
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

    @abstractmethod
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    @abstractmethod
    def plot_hist2d(self):
        pass

    @property
    @abstractmethod  # as readonly attributes
    def dt_min(self) -> backend.float:
        pass

    @property
    @abstractmethod  # as readonly attributes
    def dt_max(self) -> backend.float:
        pass

    @property
    @abstractmethod  # as readonly attributes
    def dE_min(self) -> backend.float:
        pass

    @property
    @abstractmethod  # as readonly attributes
    def dE_min(self) -> backend.float:
        pass

    @property
    @abstractmethod  # as readonly attributes
    def common_array_size(self) -> int:
        pass

    @abstractmethod
    def invalidate_cache_dE(self) -> None:
        pass

    @abstractmethod
    def invalidate_cache_dt(self) -> None:
        pass

    @abstractmethod
    def invalidate_cache(self) -> None:
        self.invalidate_cache_dE()
        self.invalidate_cache_dt()

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
