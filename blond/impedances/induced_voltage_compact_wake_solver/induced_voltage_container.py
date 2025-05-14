from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from .profile_container import Lockable

if TYPE_CHECKING:
    from .induced_voltage_compact_wake_solver import (
        InducedVoltageTyes,
    )


class InducedVoltageContainer(Lockable):
    """Helper class to contain several InducedVoltage objects"""

    def __init__(self):
        super().__init__()
        self._induced_voltage_objects: Tuple[InducedVoltageTyes] = tuple()

    @property
    def n_objects(self):
        return len(self._induced_voltage_objects)

    def add_induced_voltage(self, induced_voltage: InducedVoltageTyes):
        assert not self.is_locked
        self._induced_voltage_objects = (
            *self._induced_voltage_objects,
            induced_voltage,
        )

    def __len__(self):
        return len(self._induced_voltage_objects)

    def __iter__(self):
        for induced_voltage_object in self._induced_voltage_objects:
            yield induced_voltage_object
