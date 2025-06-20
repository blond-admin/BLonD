from __future__ import annotations

import copy
from typing import (
    Iterable,
    Tuple,
)
from typing import Optional as LateInit

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .helpers import get_init_order
from ..base import (
    BeamPhysicsRelevantElements,
    DynamicParameter,
    BeamPhysicsRelevant,
    Preparable,
)
from ..beam.base import BeamBaseClass
from ..simulation.simulation import Simulation
from blond3.core.backend import backend
from ...cycles.base import EnergyCycle
from ...physics.drifts import DriftBaseClass


class Ring(Preparable):
    def __init__(self, circumference):
        super().__init__()
        self._circumference = backend.float(circumference)
        self._elements = BeamPhysicsRelevantElements()

    @property
    def elements(self):
        return self._elements

    @property
    def circumference(self):
        return self._circumference

    @property
    def one_turn_pathlength(self):
        return

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
        all_drifts = self.elements.get_elements(DriftBaseClass)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )
        simulation.turn_i.on_change(self.update_t_rev)
