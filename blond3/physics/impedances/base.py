from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional as LateInit, Tuple, Optional
from typing import TYPE_CHECKING

from blond3.core.backend import backend
from ..profiles import ProfileBaseClass
from ...core.base import Preparable, BeamPhysicsRelevant
from ...core.beam.base import BeamBaseClass
from ...core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class WakeFieldSolver:

    @abstractmethod
    def on_wakefield_on_init_simulation(self, simulation: Simulation, parent_wakefield: WakeField) -> None:
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
    def __init__(self, group: int = 0, profile: LateInit[ProfileBaseClass] = None):
        super().__init__(group=group)
        self._profile = profile

    @abstractmethod
    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, group=self.group
            )
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.group=}, but can only handle one. Set the attribute "
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
        group: int = 0,
        profile: LateInit[ProfileBaseClass] = None,
    ):
        super().__init__(group=group, profile=profile)

        self.solver = solver
        self.sources = sources

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation, **kwargs)
        assert len(self.sources) > 0, "Provide for at least one `WakeFieldSource`"
        self.solver.on_wakefield_on_init_simulation(simulation=simulation, parent_wakefield=self)

    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        return self.solver.calc_induced_voltage()

    def track(self, beam: BeamBaseClass) -> None:
        induced_voltage = self._calc_induced_voltage()
        backend.kick_induced(
            beam.read_partial_dt(), beam.read_partial_dE(), induced_voltage
        )

    @property
    def profile(self):
        return self._profile
