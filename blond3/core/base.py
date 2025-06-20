from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Optional,
    Tuple,
    Type,
    TypeVar,
    List,
    Callable,
    Any,
)

from .beam.base import BeamBaseClass
from .ring.helpers import get_elements
from .simulation.simulation import Simulation

T = TypeVar("T")


class Preparable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def late_init(self, simulation: Simulation, **kwargs) -> None:
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

    def __init__(self, group: int = 0, name: Optional[str] = None):
        super().__init__()
        self._group = group
        if name is None:
            name = f"Unnamed-{type(self)}-{type(self).n_instances:3d}"
        self.name = name
        type(self).n_instances += 1

    @property
    def group(self):
        return self._group

    @abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        pass


class BeamPhysicsRelevantElements(ABC):
    def __init__(self):
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

    def add_element(self, element: BeamPhysicsRelevant):
        self.elements = (*self.elements, element)

    @property
    def n_elements(self):
        return len(self.elements)

    def get_elements(
        self, class_: Type[T], group: Optional[int] = None
    ) -> Tuple[T, ...]:
        elements = get_elements(self.elements, class_)
        if group is not None:
            elements = tuple(filter(lambda x: x.group == group, elements))
        return elements

    def get_element(self, class_: Type[T], group: Optional[int] = None) -> T:
        elements = self.get_elements(class_=class_, group=group)
        assert len(elements) == 1, f"{len(elements)=}"
        return elements[0]

    def reorder(self):
        pass

    def print_order(self):
        for element in self.elements:
            filtered_dict = {
                k: v for k, v in element.__dict__.items() if not k.startswith("_")
            }
            print(element.name, type(element), filtered_dict)


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
