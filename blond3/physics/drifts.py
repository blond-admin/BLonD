from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Optional
from unittest.mock import Mock

import numpy as np

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, HasPropertyCache
from .._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit, Tuple

    from typing import Iterable
    from numpy.typing import NDArray as NumpyArray

    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from ..cycles.energy_cycle import EnergyCycleBase


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


class DriftSimple(DriftBaseClass, HasPropertyCache):
    def __init__(
        self,
        share_of_circumference: float = 1.0,
        section_index: int = 0,
    ):
        super().__init__(
            share_of_circumference=share_of_circumference, section_index=section_index
        )

        self._simulation: LateInit[Simulation] = None
        self._transition_gamma: LateInit[NumpyArray] = None

    @staticmethod
    def headless(
        transition_gamma: float | Iterable | Tuple[NumpyArray, NumpyArray],
        gamma: NumpyArray,
        share_of_circumference: float,
        section_index: int = 0,
        cycle_time: Optional[NumpyArray] = None,
    ):
        n_turns = gamma.shape[1]
        if cycle_time is None:
            cycle_time = np.empty((section_index + 1, n_turns))
        assert gamma.shape == cycle_time.shape, (
            f"Need shape (section_index " f"+ 1, n_turns), i.e. {cycle_time.shape}"
        )
        d = DriftSimple(
            share_of_circumference=share_of_circumference,
            section_index=section_index,
        )
        from .._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.energy_cycle.cycle_time = cycle_time
        simulation.energy_cycle.n_turns = n_turns
        simulation.energy_cycle.gamma = gamma
        simulation.ring.transition_gamma_init = transition_gamma
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(simulation=simulation, turn_i_init=0, n_turns=1)
        return d

    @requires(
        [
            "EnergyCycleBase",  #  for energy_cycle
            "BeamPhysicsRelevantElements",  # for .section_index,
        ]
    )
    def on_init_simulation(self, simulation: Simulation) -> None:
        energy_cycle: EnergyCycleBase = simulation.energy_cycle
        from blond.input_parameters.ring_options import RingOptions

        ring_options = RingOptions()
        self._transition_gamma = (
            ring_options.reshape_data(  # FIXME use correct reshaping
                input_data=simulation.ring.transition_gamma_init,
                n_turns=energy_cycle.n_turns - 1,
                n_sections=1,
                interp_time=energy_cycle.cycle_time[self.section_index, :],
            )[0, :]
        )
        self._simulation = simulation

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
            eta_0=self.eta_0[current_turn_i],
            beta=self._simulation.energy_cycle.beta[self.section_index, current_turn_i],
            energy=self._simulation.energy_cycle.total_energy[
                self.section_index, current_turn_i
            ],
        )

    @cached_property  # as readonly attributes # as readonly attributes
    def eta_0(self) -> NumpyArray:
        return np.ascontiguousarray(
            self.alpha_0
            - self._simulation.energy_cycle.gamma[self.section_index, :] ** (-2.0),
            dtype=backend.float,
        )

    @cached_property
    def momentum_compaction_factor(self) -> backend.float | NumpyArray:
        return 1 / self._transition_gamma**2

    cached_props = (
        "momentum_compaction_factor",
        "eta_0",
    )

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> backend.float | NumpyArray:
        return self.momentum_compaction_factor

    def invalidate_cache(self):
        super()._invalidate_cache(DriftSimple.cached_props)


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
