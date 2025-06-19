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
from ..base import BeamPhysicsRelevantElements, DynamicParameter, \
    BeamPhysicsRelevant, Preparable
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
        self._beams: Tuple[BeamBaseClass, ...] = ()
        self._energy_cycle: LateInit[EnergyCycle] = None
        self._t_rev = DynamicParameter(None)

    @property
    def beams(self):
        return self._beams

    @property
    def energy_cycle(self):
        return self._energy_cycle

    @property
    def elements(self):
        return self._elements

    @property
    def t_rev(self):
        return self._t_rev

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

    def magic_add(self, ring_attributes: dict | Iterable):
        from collections.abc import Iterable as Iterable_  # so isinstance works

        if isinstance(ring_attributes, dict):
            values = ring_attributes.values()
        elif isinstance(ring_attributes, Iterable_):
            values = ring_attributes
        else:
            raise ValueError(
                f"Cant handle {type(ring_attributes)=}, must be " f"`Iterable` instead"
            )
        for val in values:
            if isinstance(val, BeamBaseClass):
                self.add_beam(beam=val)
            elif isinstance(val, BeamPhysicsRelevant):
                self.add_element(element=val)
            elif isinstance(val, EnergyCycle):
                self.set_energy_cycle(val)
            else:
                pass

        # reorder elements for correct execution order
        self.elements.reorder()

    def set_energy_cycle(self, energy_cycle: NumpyArray | EnergyCycle):
        if isinstance(energy_cycle, np.ndarray):
            energy_cycle = EnergyCycle(energy_cycle)
        self._energy_cycle = energy_cycle

    def add_beam(self, beam: BeamBaseClass):
        if len(self.beams) == 0:
            assert beam.is_counter_rotating is False
            self._beams = (beam,)

        elif len(self.beams) == 1:
            assert beam.is_counter_rotating is True
            self._beams = (self.beams[0], beam)
        else:
            raise NotImplementedError("No more than two beam allowed!")

    def late_init(self, simulation: Simulation, **kwargs) -> None:
        assert self.beams != (), f"{self.beams=}"
        assert self.energy_cycle is not None, f"{self.energy_cycle}"
        self._energy_cycle.late_init(simulation=simulation)
        ordered_classes = get_init_order(self.elements.elements, "late_init.requires")
        for cls in ordered_classes:
            for element in self.elements.elements:
                if not type(element) == cls:
                    continue
                element.late_init(simulation=simulation)
        for beam in self.beams:
            beam.late_init(simulation=simulation)
        all_drifts = self.elements.get_elements(DriftBaseClass)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )
        simulation.turn_i.on_change(self.update_t_rev)

    def get_t_rev(self, turn_i):
        return self.circumference / beta_by_ekin(
            self.energy_cycle._beam_energy_by_turn[turn_i]
        )

    def update_t_rev(self, new_turn_i: int):
        self.t_rev.value = self.get_t_rev(new_turn_i)
