from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from blond3 import Simulation

    from blond3._core.base import IntensityEffect


class IntensityEffectManager:
    def __init__(self, simulation: Simulation):
        self._parent_simulation = simulation
        self._active = True
        self._frozen = True

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if value != self._active:
            self._active = value
            intensity_elements = self._parent_simulation.ring.elements.get_elements(
                IntensityEffect,
            )
            for element in intensity_elements:
                element.active = bool(value)

    @property
    def frozen(self) -> bool:
        return self._frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        if value != self._frozen:
            self._frozen = value
            intensity_elements = self._parent_simulation.ring.elements.get_elements(
                IntensityEffect,
            )
            for element in intensity_elements:
                element.frozen = bool(value)
