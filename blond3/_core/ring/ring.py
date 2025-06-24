from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from .beam_physics_relevant_elements import BeamPhysicsRelevantElements
from ..backends.backend import backend
from ..base import (
    BeamPhysicsRelevant,
    Preparable,
)

if TYPE_CHECKING:
    from typing import (
        Iterable,
        Optional,
    )

    from ..simulation.simulation import Simulation


class Ring(Preparable):
    _bending_radius: np.float32 | np.float64

    _circumference: np.float32 | np.float64

    def __init__(self, circumference: float, bending_radius: Optional[float] = None):
        if bending_radius is not None:
            bending_radius = circumference / (2 * np.pi)

        super().__init__()
        self._elements = BeamPhysicsRelevantElements()

        self._circumference = backend.float(circumference)
        self._bending_radius = backend.float(bending_radius)

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    @property  # as readonly attributes
    def bending_radius(self):
        return self._bending_radius

    @property  # as readonly attributes
    def elements(self) -> BeamPhysicsRelevantElements:
        return self._elements

    @property  # as readonly attributes
    def circumference(self):
        return self._circumference

    def add_element(
        self,
        element: BeamPhysicsRelevant,
        reorder: bool = False,
        deepcopy: bool = False,
    ):
        if deepcopy:
            element = copy.deepcopy(element)

        self.elements.add_element(element)

        if reorder:
            self.elements.reorder()

    def add_elements(
        self, elements: Iterable[BeamPhysicsRelevant], reorder: bool = False
    ):
        for element in elements:
            self.add_element(element=element)

        if reorder:
            self.elements.reorder()

    def on_init_simulation(self, simulation: Simulation) -> None:
        from ...physics.drifts import DriftBaseClass  # prevent cyclic import

        all_drifts = self.elements.get_elements(DriftBaseClass)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )
