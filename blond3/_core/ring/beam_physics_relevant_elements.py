from __future__ import annotations

from typing import TYPE_CHECKING

from ..ring.helpers import get_elements

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Optional,
        Tuple,
        Type,
        TypeVar,
    )
    from ..base import BeamPhysicsRelevant

    T = TypeVar("T")


class BeamPhysicsRelevantElements:
    def __init__(self):
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

    def add_element(self, element: BeamPhysicsRelevant):
        self.elements = (*self.elements, element)

    @property  # as readonly attributes
    def n_elements(self):
        return len(self.elements)

    def get_elements(
        self, class_: Type[T], group: Optional[int] = None
    ) -> Tuple[T, ...]:
        elements = get_elements(self.elements, class_)
        if group is not None:
            elements = tuple(filter(lambda x: x.section_index == group, elements))
        return elements

    def get_element(self, class_: Type[T], group: Optional[int] = None) -> T:
        elements = self.get_elements(class_=class_, group=group)
        assert len(elements) == 1, f"{len(elements)=}"
        return elements[0]

    def reorder(self):
        pass

    def count(self, class_: Type[T]):
        return len(self.get_elements(class_=class_))

    def print_order(self):
        for element in self.elements:
            filtered_dict = {
                k: v for k, v in element.__dict__.items() if not k.startswith("_")
            }
            print(element.name, type(element), filtered_dict)
