from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional as LateInit, Tuple, Optional
from typing import TYPE_CHECKING

from ..profiles import ProfileBaseClass
from ..._core.backends.backend import backend
from ..._core.base import BeamPhysicsRelevant
from ..._core.beam.base import BeamBaseClass
from ..._core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class WakeFieldSolver:
    @abstractmethod
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        pass

    @abstractmethod
    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        pass


class WakeFieldSource(ABC):
    pass


class TimeDomain(ABC):
    pass


class FreqDomain(ABC):
    @abstractmethod
    def get_freq_y(self, freq_x: NumpyArray) -> NumpyArray:
        pass


class AnalyticWakeFieldSource(WakeFieldSource):
    pass


class DiscreteWakeFieldSource(WakeFieldSource):
    pass


class Impedance(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0, profile: LateInit[ProfileBaseClass] = None):
        super().__init__(section_index=section_index)
        self._profile = profile

    @property  # as readonly attributes
    def profile(self):
        return self._profile

    @abstractmethod
    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, group=self.section_index
            )
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.section_index=}, but can only handle one. Set the attribute "
                f"`your_impedance.profile` in advance or remove the second "
                f"profile from this group."
            )
            self._profile = profiles[0]
        else:
            pass


class WakeField(Impedance):
    def __init__(
        self,
        sources: Tuple[WakeFieldSource, ...],
        solver: Optional[WakeFieldSolver],
        section_index: int = 0,
        profile: LateInit[ProfileBaseClass] = None,
    ):
        super().__init__(section_index=section_index, profile=profile)

        self.solver = solver
        self.sources = sources

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        assert len(self.sources) > 0, "Provide for at least one `WakeFieldSource`"
        self.solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=self
        )

    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        return self.solver.calc_induced_voltage()

    def track(self, beam: BeamBaseClass) -> None:
        induced_voltage = self.calc_induced_voltage()
        backend.kick_induced(
            beam.read_partial_dt(), beam.read_partial_dE(), induced_voltage
        )
