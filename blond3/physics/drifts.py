from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Optional as LateInit, TYPE_CHECKING

import numpy as np

from .cavities import CavityBaseClass
from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant
from .._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable
    from numpy.typing import NDArray as NumpyArray

    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from .. import EnergyCycle


class DriftBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(self, share_of_circumference: float, section_index: int = 0):
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


class DriftSimple(DriftBaseClass):
    def __init__(
        self,
        transition_gamma: float | Iterable,
        share_of_circumference: float = 1.0,
        section_index: int = 0,
    ):
        super().__init__(
            share_of_circumference=share_of_circumference, section_index=section_index
        )
        self.__transition_gamma = transition_gamma

        self._simulation: LateInit[Simulation] = None
        self._eta_0: LateInit[NumpyArray] = None

    @requires(["EnergyCycle"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        cycle: EnergyCycle = simulation.energy_cycle
        from blond.input_parameters.ring_options import RingOptions

        ring_options = RingOptions()
        self._transition_gamma = ring_options.reshape_data(
            input_data=self.__transition_gamma,
            n_turns=cycle.n_turns,
            n_sections=simulation.ring.elements.count(CavityBaseClass),
            interp_time=cycle.cycle_time,
        )[self.section_index, :]

        self._eta_0 = np.ascontiguousarray(
            self.alpha_0 - cycle.gamma[self.section_index, :] ** (-2.0),
            dtype=backend.float,
        )
        self._simulation = simulation

    @property  # as readonly attributes # as readonly attributes
    def eta_0(self) -> NumpyArray:
        assert self._eta_0 is not None
        return self._eta_0

    @property  # as readonly attributes # as readonly attributes
    def transition_gamma(self) -> backend.float | NumpyArray:
        return self._transition_gamma

    def track(self, beam: BeamBaseClass):
        current_turn_i = self._simulation.turn_i.value

        backend.specials.drift_simple(
            dt=beam.write_partial_dt(),
            dE=beam.read_partial_dE(),
            t_rev=self._simulation.energy_cycle.t_rev[current_turn_i],
            length_ratio=self._share_of_circumference,
            eta_0=self._eta_0[current_turn_i],
            beta=self._simulation.energy_cycle.beta[self.section_index, current_turn_i],
            energy=self._simulation.energy_cycle.energy[
                self.section_index, current_turn_i
            ],
        )

    @cached_property
    def momentum_compaction_factor(self) -> backend.float | NumpyArray:
        return 1 / self._transition_gamma**2

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> backend.float | NumpyArray:
        return self.momentum_compaction_factor

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
