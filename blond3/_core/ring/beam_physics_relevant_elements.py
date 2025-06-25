from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..ring.helpers import get_elements

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Optional,
        Tuple,
        Type,
        TypeVar,
    )
    from ..base import BeamPhysicsRelevant
    from numpy.typing import NDArray as NumpyArray

    T = TypeVar("T")


class BeamPhysicsRelevantElements:
    def __init__(self):
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

    def get_sections(self) -> Tuple[int, ...]:
        res = set()
        for e in self.elements:
            res.add(e.section_index)
        return tuple(sorted(res))

    def get_section_circumference_shares(self) -> NumpyArray:
        from ...physics.drifts import DriftBaseClass

        sections = self.get_sections()
        result = np.empty(len(sections))
        for section_i in sections:
            drifts = self.get_elements(DriftBaseClass, section_i=section_i)
            if len(drifts) > 0:
                result[section_i] = sum([d.share_of_circumference for d in drifts])
            else:
                result[section_i] = 0
        return result

    def add_element(self, element: BeamPhysicsRelevant):
        self.elements = (*self.elements, element)

    @property  # as readonly attributes
    def n_elements(self):
        return len(self.elements)

    def get_elements(
        self, class_: Type[T], section_i: Optional[int] = None
    ) -> Tuple[T, ...]:
        elements = get_elements(self.elements, class_)
        if section_i is not None:
            elements = tuple(filter(lambda x: x.section_index == section_i, elements))
        return elements

    def get_element(self, class_: Type[T], group: Optional[int] = None) -> T:
        elements = self.get_elements(class_=class_, section_i=group)
        assert len(elements) == 1, f"{len(elements)=}"
        return elements[0]

    def reorder(self):
        pass

    def count(self, class_: Type[T]):
        return len(self.get_elements(class_=class_))

    def print_order(self):
        print(self.get_order_info())

    def get_order_info(self):
        sep = 78 * "-" + "\n"
        content = ""
        content += sep
        content += "Execution order\n"
        content += "---------------\n"
        content += (
            f"{'element.name':40s} {'type':20s} "
            f"{('section_index'):13s} {'filtered_dict'}\n"
        )
        for element in self.elements:
            filtered_dict = {
                k: v
                for k, v in element.__dict__.items()
                if (not k.startswith("_")) and (k != "name")
            }
            content += (
                f"{element.name:40s} {(type(element).__name__):20s} "
                f"{str(element.section_index):13s} {filtered_dict}\n"
            )
        content += sep
        return content
