from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from unittest.mock import Mock

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, HasPropertyCache, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit, Tuple

    from typing import Iterable
    from numpy.typing import NDArray as NumpyArray

    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass


class DriftBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        share_of_circumference: float,
        section_index: int = 0,
    ):
        super().__init__(section_index=section_index)
        self._share_of_circumference: backend.float = backend.float(
            share_of_circumference
        )

    @property  # as readonly attributes
    def share_of_circumference(self) -> backend.float:
        return self._share_of_circumference

    def track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class DriftSimple(DriftBaseClass, Schedulable, HasPropertyCache):
    def __init__(
        self,
        share_of_circumference: float = 1.0,
        section_index: int = 0,
    ):
        super().__init__(
            share_of_circumference=share_of_circumference,
            section_index=section_index,
        )

        self._transition_gamma: float | None = None
        self._momentum_compaction_factor: float | None = None
        self.length: float | None = None

        self._simulation: LateInit[Simulation] = None

    @property  # read only, set by `transition_gamma`
    def momentum_compaction_factor(self):
        return self._momentum_compaction_factor

    @property
    def transition_gamma(self):
        return self._transition_gamma

    @transition_gamma.setter
    def transition_gamma(self, transition_gamma):
        transition_gamma = backend.float(transition_gamma)
        self._momentum_compaction_factor = 1 / (transition_gamma * transition_gamma)
        self._transition_gamma = transition_gamma

    @staticmethod
    def headless(
        transition_gamma: float | Iterable | Tuple[NumpyArray, NumpyArray],
        circumference: float,
        share_of_circumference: float,
        section_index: int = 0,
    ):
        from .._core.base import DynamicParameter

        d = DriftSimple(
            share_of_circumference=share_of_circumference,
            section_index=section_index,
        )
        d.transition_gamma = backend.float(transition_gamma)
        from .._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.circumference = backend.float(circumference)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(simulation=simulation, turn_i_init=0, n_turns=1)
        return d

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        self.length = backend.float(
            self.share_of_circumference * simulation.ring.circumference
        )
        if (
            self.transition_gamma is None
        ) and "transition_gamma" not in self.schedules.keys():
            raise ValueError(
                "You need to define `transition_gamma` via `.transition_gamma=...` "
                "or `.schedule(attribute='transition_gamma', value=...)`"
            )

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        self.apply_schedules(
            turn_i=self._simulation.turn_i.value,
            reference_time=beam.reference_time,
        )
        dt = backend.float(self.length / beam.reference_velocity)
        gamma = beam.reference_gamma
        eta_0 = self.alpha_0 - (1 / (gamma * gamma))
        backend.specials.drift_simple(
            dt=beam.write_partial_dt(),
            dE=beam.read_partial_dE(),
            T=dt,
            eta_0=eta_0,
            beta=beam.reference_beta,
            energy=beam.reference_total_energy,
        )
        beam.reference_time += dt

    def eta_0(self, gamma: float) -> backend.float:
        return backend.float(self.alpha_0 - (1 / (gamma * gamma)))

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> backend.float:
        return self.momentum_compaction_factor

    def invalidate_cache(self):
        # super()._invalidate_cache(DriftSimple.cached_props)
        pass


class DriftSpecial(DriftBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)

    pass


class DriftXSuite(DriftBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)

    pass
