from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .base import ProgrammedCycle
from .._core.backends.backend import backend
from .._core.base import HasPropertyCache
from .._core.ring.helpers import requires
from ..physics.cavities import (
    CavityBaseClass,
)
from ..physics.drifts import DriftBaseClass

if TYPE_CHECKING:
    from typing import Optional as LateInit

    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation


class EnergyCycle(ProgrammedCycle, HasPropertyCache):
    from .. import Ring

    def __init__(self, synchronous_data: NumpyArray, synchronous_data_type="momentum"):
        super().__init__()
        self._synchronous_data = synchronous_data
        self._synchronous_data_type = synchronous_data_type
        from blond.input_parameters.ring import Ring as Blond2Ring

        self._ring: LateInit[Blond2Ring] = None

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        return EnergyCycle(
            synchronous_data=backend.linspace(
                start, stop, turns + 1, endpoint=endpoint, dtype=backend.float
            )
        )

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        self.invalidate_cache()

    @requires(["Ring"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        from blond.input_parameters.ring import Ring as Blond2Ring

        drifts = simulation.ring.elements.get_elements(DriftBaseClass)
        cavities = simulation.ring.elements.get_elements(CavityBaseClass)
        assert len(drifts) == len(cavities)
        self._ring = Blond2Ring(
            ring_length=[
                (e.share_of_circumference * simulation.ring.circumference)
                for e in drifts
            ],
            synchronous_data=self._synchronous_data,
            n_sections=len(cavities),
            alpha_0=np.nan,
            particle=simulation.beams[0].particle_type,
            n_turns=self.n_turns,
            synchronous_data_type=self._synchronous_data_type,
            bending_radius=simulation.ring.bending_radius,
        )
        self.invalidate_cache()

    @cached_property
    def n_turns(self):
        return len(self._synchronous_data) - 1

    @cached_property  # as readonly attributes
    def beta(self) -> NumpyArray:
        return self._ring.beta.astype(backend.float)

    @cached_property  # as readonly attributes
    def gamma(self) -> NumpyArray:
        return self._ring.gamma.astype(backend.float)

    @cached_property  # as readonly attributes
    def energy(self) -> NumpyArray:
        return self._ring.energy.astype(backend.float)

    @cached_property  # as readonly attributes
    def kin_energy(self) -> NumpyArray:
        return self._ring.kin_energy.astype(backend.float)

    @cached_property  # as readonly attributes
    def delta_E(self) -> NumpyArray:
        return self._ring.delta_E.astype(backend.float)

    @cached_property  # as readonly attributes
    def t_rev(self) -> NumpyArray:
        return self._ring.t_rev.astype(backend.float)

    @cached_property  # as readonly attributes
    def cycle_time(self) -> NumpyArray:
        return self._ring.cycle_time.astype(backend.float)

    @cached_property  # as readonly attributes
    def f_rev(self) -> NumpyArray:
        return self._ring.f_rev.astype(backend.float)

    @cached_property  # as readonly attributes
    def omega_rev(self) -> NumpyArray:
        return self._ring.omega_rev.astype(backend.float)

    props = (
        "n_turns",
        "beta",
        "gamma",
        "energy",
        "kin_energy",
        "delta_E",
        "t_rev",
        "cycle_time",
        "f_rev",
        "omega_rev",
    )

    def invalidate_cache(self):
        super().invalidate_cache(EnergyCycle.props)
