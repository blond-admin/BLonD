from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..base import Preparable
from ..ring.helpers import get_elements
from ... import Simulation
from ...physics.drifts import DriftBaseClass

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


class BeamPhysicsRelevantElements(Preparable):
    def __init__(self):
        super().__init__()
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._check_section_indexing()

    def _check_section_indexing(self):
        from ...physics.cavities import CavityBaseClass
        from ...physics.drifts import DriftBaseClass

        elem_section_indices = [e.section_index for e in self.elements]
        assert min(elem_section_indices) == 0, "section_index=0 must be set"
        assert np.all(np.diff(elem_section_indices) >= 0), (
            f"Section indices must be "
            f"increasing, but got"
            f" {elem_section_indices}"
        )
        cavities = self.get_elements(CavityBaseClass)
        cav_section_indices = [c.section_index for c in cavities]
        all_different = len(cav_section_indices) == len(set(cav_section_indices))
        if not all_different:
            raise ValueError(
                f"Each cavity must be in a different section, "
                f"but got "
                f"{[(cav.name, cav.section_index) for cav in cavities]}"
            )

        for section_index in np.sort(np.unique(elem_section_indices)):
            cavities = self.get_elements(CavityBaseClass, section_i=section_index)
            drifts = self.get_elements(DriftBaseClass, section_i=section_index)
            if len(cavities) == 0:
                raise RuntimeError(f"Missing cavity in section" f" {section_index}")
            if len(drifts) == 0:
                raise RuntimeError(f"Missing drift in section" f"" f" {section_index}")

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    def get_sections_indices(self) -> Tuple[int, ...]:
        unique_section_indices = set()
        for e in self.elements:
            unique_section_indices.add(e.section_index)
        return tuple(sorted(unique_section_indices))

    def get_section_circumference_shares(self) -> NumpyArray:
        from ...physics.drifts import DriftBaseClass

        sections = self.get_sections_indices()
        result = np.empty(len(sections))
        for section_i in sections:
            drifts = self.get_elements(DriftBaseClass, section_i=section_i)
            if len(drifts) > 0:
                result[section_i] = sum([d.share_of_circumference for d in drifts])
            else:
                result[section_i] = 0
        return result

    def add_element(self, element: BeamPhysicsRelevant):
        assert isinstance(element.section_index, int)
        self.elements = (*self.elements, element)

    @property  # as readonly attributes
    def n_sections(self):
        return len(np.unique([e.section_index for e in self.elements]))

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

    def get_element(self, class_: Type[T], section_i: Optional[int] = None) -> T:
        elements = self.get_elements(class_=class_, section_i=section_i)
        assert len(elements) == 1, f"{len(elements)=}"
        return elements[0]

    def reorder(self):
        for section_i in range(self.n_sections):
            self.reorder_section(section_i)

        self._check_section_indexing()

    def reorder_section(self, section_index: int):
        assert isinstance(section_index, int)
        from ...physics.drifts import DriftBaseClass
        from ...physics.profiles import ProfileBaseClass
        from ...physics.cavities import CavityBaseClass
        from ...physics.losses import LossesBaseClass
        from ...physics.impedances.base import ImpedanceBaseClass
        from ...physics.feedbacks.base import FeedbackBaseClass

        natural_order = (
            LossesBaseClass,
            ProfileBaseClass,
            FeedbackBaseClass,
            ImpedanceBaseClass,
            CavityBaseClass,
            DriftBaseClass,
        )
        assert self.count(CavityBaseClass, section_i=section_index) == 1, (
            f"Only one cavity per section allowed, but got "
            f"{self.count(CavityBaseClass, section_i=section_index)}"
        )
        elements_in_section = [
            e for e in self.elements if e.section_index == section_index
        ]
        elements_before_section = [
            e for e in self.elements if e.section_index < section_index
        ]
        elements_after_section = [
            e for e in self.elements if e.section_index > section_index
        ]
        # reorder elements based on natural order
        # account for the fact that elements could be instanced of two base
        # classes. In this case the first match in natural order is chosen.
        _seen = set()
        ordered_elements = []

        for cls in natural_order:
            for e in elements_in_section:
                if e not in _seen and isinstance(e, cls):
                    ordered_elements.append(e)
                    _seen.add(e)

        self.elements = tuple(
            elements_before_section + ordered_elements + elements_after_section
        )

    def count(self, class_: Type[T], section_i: Optional[int] = None):
        return len(self.get_elements(class_=class_, section_i=section_i))

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
                k: pprint(v)
                for k, v in element.__dict__.items()
                if (not k.startswith("_")) and (k != "name")
            }
            content += (
                f"{element.name:40s} {(type(element).__name__):20s} "
                f"{str(element.section_index):13s} {filtered_dict}\n"
            )
        content += sep
        return content


def pprint(v):
    if isinstance(v, np.ndarray):
        return f"array(min={v.min()}, max={v.max()}, shape={v.shape})"
    else:
        return v
