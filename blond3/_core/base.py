from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .beam.base import BeamBaseClass
    from .simulation.simulation import Simulation
    from typing import (
        Optional,
        TypeVar,
        List,
        Callable,
        Any,
    )

    T = TypeVar("T")


class Preparable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    @abstractmethod
    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class MainLoopRelevant(Preparable):
    def __init__(self):
        super().__init__()
        self.each_turn_i = 1

    def is_active_this_turn(self, turn_i: int):
        return turn_i % self.each_turn_i == 0


class BeamPhysicsRelevant(MainLoopRelevant):
    n_instances = 0

    def __init__(self, section_index: int = 0, name: Optional[str] = None):
        super().__init__()
        self._section_index = section_index
        if name is None:
            name = (f"Unnamed-{type(self).__name__}-"
                    f"{type(self).n_instances:03d}")
        self.name = name
        type(self).n_instances += 1

    @property  # as readonly attributes
    def section_index(self) -> int:
        return self._section_index

    @abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        pass


class DynamicParameter:  # TODO add code generation for this method with type-hints
    def __init__(self, value_init):
        self._value = value_init
        self._observers: List[Callable[[Any], None]] = []

    def on_change(self, callback: Callable[[Any], None]):
        """Subscribe to changes on a specific parameter."""
        self._observers.append(callback)

    def _notify(self, value):
        """Notify all observers about a parameter change."""
        for callback in self._observers:
            callback(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val: T):
        if new_val != self._value:
            self._notify(new_val)
        self._value = new_val


class HasPropertyCache(object):
    def invalidate_cache(self, props: Tuple[str, ...]):
        for prop in props:
            self.__dict__.pop(prop, None)
