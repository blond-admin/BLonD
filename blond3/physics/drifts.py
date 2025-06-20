from __future__ import annotations

from abc import ABC
from functools import cached_property

from ..core.backend import backend
from ..core.base import BeamPhysicsRelevant
from ..core.beam.base import BeamBaseClass
from ..core.simulation.simulation import Simulation


class DriftBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(self, share_of_circumference: float, group: int = 0):
        super().__init__(group=group)
        self._share_of_circumference = backend.float(share_of_circumference)

    @property
    def share_of_circumference(self):
        return self._share_of_circumference

    def track(self, beam: BeamBaseClass):
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass


class DriftSimple(DriftBaseClass):
    def __init__(
        self,
        transition_gamma: float,
        share_of_circumference: float = 1.0,
        group: int = 0,
    ):
        super().__init__(share_of_circumference=share_of_circumference, group=group)
        self._transition_gamma = backend.float(transition_gamma)

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    @property
    def transition_gamma(self):
        return self._transition_gamma

    def track(self, beam: BeamBaseClass):
        backend.drift_simple(
            beam.write_partial_dt(), beam.read_partial_dE(), self._transition_gamma
        )

    @cached_property
    def momentum_compaction_factor(self):
        return 1 / self._transition_gamma**2

    def invalidate_cache(self):
        self.__dict__.pop("momentum_compaction_factor", None)


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
