from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..._core.backends.backend import backend
from ..._core.base import BeamPhysicsRelevant
from ..._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit, Tuple, Optional

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ..profiles import ProfileBaseClass
    from ..._core.simulation.simulation import Simulation
    from ..._core.beam.base import BeamBaseClass


class WakeFieldSolver:
    @abstractmethod
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        pass

    @abstractmethod
    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
        pass


class WakeFieldSource(ABC):
    def __init__(self, is_dynamic: bool):
        self.is_dynamic = is_dynamic


class TimeDomain(ABC):
    @abstractmethod
    def get_wake_impedance(
        self, time: NumpyArray, simulation: Simulation
    ) -> NumpyArray:
        pass


class FreqDomain(ABC):
    @abstractmethod
    def get_impedance(self, freq_x: NumpyArray, simulation: Simulation) -> NumpyArray:
        pass


class AnalyticWakeFieldSource(WakeFieldSource):
    pass


class DiscreteWakeFieldSource(WakeFieldSource):
    pass


class ImpedanceBaseClass(BeamPhysicsRelevant):
    def __init__(
        self, section_index: int = 0, profile: LateInit[ProfileBaseClass] = None
    ):
        super().__init__(section_index=section_index)
        self._profile = profile

    @property  # as readonly attributes
    def profile(self) -> ProfileBaseClass:
        return self._profile

    @abstractmethod
    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    @requires(
        [
            "BeamPhysicsRelevantElements",  # for .section_index,
        ]
    )
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        from ..profiles import ProfileBaseClass  # prevent cyclic import

        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, section_i=self.section_index
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


class WakeField(ImpedanceBaseClass):
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

    @requires(["EnergyCycleBase", "BeamBaseClass"])  # because
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        assert len(self.sources) > 0, "Provide for at least one `WakeFieldSource`"
        self.solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=self
        )

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
        self.induced_voltage = self.solver.calc_induced_voltage(beam=beam)
        return self.induced_voltage

    def track(self, beam: BeamBaseClass) -> None:
        induced_voltage = self.calc_induced_voltage(beam=beam)
        backend.specials.kick_induced_voltage(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=induced_voltage,
            bin_centers=self.profile.hist_x,  # base for induced voltage
            charge=beam.particle_type.charge,
            acceleration_kick=0.0,  # TODO was this ever required??
        )
