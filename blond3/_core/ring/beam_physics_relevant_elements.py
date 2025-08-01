from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..base import Preparable
from ..beam.base import BeamBaseClass
from ..ring.helpers import get_elements
from ... import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Optional,
        Tuple,
        Type,
        TypeVar,
        Any,
        List,
    )
    from ..base import BeamPhysicsRelevant
    from numpy.typing import NDArray as NumpyArray

    T = TypeVar("T")


class BeamPhysicsRelevantElements(Preparable):
    """Container object to manage all beam interactions in `Ring`"""

    def __init__(self):
        super().__init__()
        self.elements: List[BeamPhysicsRelevant] = []

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        Parameters
        ----------
        simulation
            Simulation context manager"""
        self._check_section_indexing()

    def _check_section_indexing(self):
        """Verify that indices have been set correctly"""
        from ...physics.cavities import CavityBaseClass
        from ...physics.drifts import DriftBaseClass

        elem_section_indices = [e.section_index for e in self.elements]
        assert min(elem_section_indices) == 0, "section_index=0 must be set"
        assert np.all(np.diff(elem_section_indices) >= 0), (
            f"Section indices must be increasing, but got {elem_section_indices}"
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
                raise RuntimeError(f"Missing cavity in section {section_index}")
            if len(drifts) == 0:
                raise RuntimeError(f"Missing drift in section {section_index}")

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """

        pass

    def get_sections_indices(self) -> Tuple[int, ...]:
        """Get all unique section indices"""
        unique_section_indices = set()
        for e in self.elements:
            unique_section_indices.add(e.section_index)
        return tuple(sorted(unique_section_indices))

    def get_sections_orbit_length(self) -> NumpyArray:
        """
        Get `share_of_circumference` per section

        Notes
        -----
        This is different from per-drift listing
        """
        from ...physics.drifts import DriftBaseClass

        sections = self.get_sections_indices()
        result = np.empty(len(sections))
        for section_i in sections:
            drifts = self.get_elements(DriftBaseClass, section_i=section_i)
            if len(drifts) > 0:
                result[section_i] = sum([d.orbit_length for d in drifts])
            else:
                result[section_i] = 0
        return result

    def add_element(self, element: BeamPhysicsRelevant) -> None:
        """
        Append a beam physics-relevant element to the container.

        This method appends the given element to the
        internal sequence of elements, maintaining insertion order.
        The element must have a valid integer `section_index`.

        Parameters
        ----------
        element
            An object representing a beamline component or any element
            relevant to beam physics. Must have a valid  `section_index`
            attribute of type `int`.

        Raises
        ------
        AssertionError
            If `element.section_index` is not an integer.
        """
        assert isinstance(element.section_index, int)
        insert_at = None
        for i, elem in enumerate(self.elements):
            if elem.section_index == element.section_index:
                insert_at = i
        self.elements.append(element)

    def check_section_index_compatibility(self, element:
    BeamPhysicsRelevant, insert_at: int):
        """
        Internal method to check the element is inserted in the defined 
        section.

        Parameters
        ----------
        element
            An object representing a beamline component or any element
            relevant to beam physics. Must have a valid  `section_index`
            attribute of type `int`.
        insert_at
            Single location.
        Raises
        -------
        AssertionError
            If 'element.section_index' is inconsistent with the section of
            insertion.
            If insert_at is not within [0:len(ring.elements.elements)+1]
        """
        try :
            if (insert_at != 0) and (insert_at != len(self.elements)+1):
                assert (self.elements[insert_at - 1].section_index <=
                        element.section_index <= self.elements[
                            insert_at].section_index)
            elif insert_at == 0:
                assert (element.section_index ==
                        self.elements[insert_at].section_index)
            elif insert_at == len(self.elements)+1:
                assert (self.elements[insert_at - 1].section_index <=
                        element.section_index <=
                        self.elements[insert_at - 1].section_index + 1)
            else:
                raise AssertionError(f'The element must be inserted within ['
                                 f'0:{len(self.elements)+1}] indexes. ')
        except:
            raise AssertionError('The element section index is incompatible '
                                 'with the requested location. Please allow '
                                 'overwrite for automatic handling.')
    def insert(self, element: BeamPhysicsRelevant, insert_at: int) -> None:
        """
        Insert a beam physics-relevant element to the container at the
        specified index.

        Parameters
        ----------
        element
            An object representing a beamline component or any element
            relevant to beam physics. Must have a valid  `section_index`
            attribute of type `int`.
        insert_at:
            Location of the element to be inserted.

        Raises
        ------
        AssertionError
            If `element.section_index` is not an integer.
            If 'element.section_index' is inconsistent with the section of
            insertion.
            If insert_at is not within [0:len(ring.elements.elements)+1]
        """
        assert isinstance(element.section_index, int)
        self.check_section_index_compatibility(element = element,
                                               insert_at= insert_at)
        self.elements.insert(insert_at, element)

    @property  # as readonly attributes
    def n_sections(self) -> int:
        """Number of sections that are mentioned by elements"""
        return len(np.unique([e.section_index for e in self.elements]))

    @property  # as readonly attributes
    def n_elements(self) -> int:
        """Number of elements contained in this class"""
        return len(self.elements)

    def get_elements(
        self, class_: Type[T], section_i: Optional[int] = None
    ) -> Tuple[T, ...]:
        """
        Get all elements of specified type (potentially filtered by section)

        Parameters
        ----------
        class_
            Type of class to filter for
        section_i
            Optional filter to get instances only in one section
        """
        elements = get_elements(self.elements, class_)
        if section_i is not None:
            elements = tuple(filter(lambda x: x.section_index == section_i, elements))
        return elements

    def get_element(self, class_: Type[T], section_i: Optional[int] = None) -> T:
        """
        Retrieve a single element of the specified type, optionally filtered by section.

        This method returns exactly one element of the given type. If
        `section_i` is provided, only elements in that section are
        considered.

        Notes
        -----
        An assertion error is raised if the number of matching
        elements is not exactly one.

        Parameters
        ----------
        class_
            The class type to filter elements by.
        section_i
            Optional section index to restrict the search to a specific section.

        Returns
        -------
        signle_element
            The single element of the specified type (and section, if provided).

        Raises
        ------
        AssertionError
            If the number of matching elements is not exactly one.
        """
        elements = self.get_elements(
            class_=class_,
            section_i=section_i,
        )
        assert len(elements) == 1, f"{len(elements)=}"
        return elements[0]

    def reorder(self):
        """Reorder each section by `natural_order`"""
        for section_index in range(self.n_sections):
            self.reorder_section(
                section_index=section_index,
            )

        self._check_section_indexing()

    def reorder_section(self, section_index: int):
        """
        Reorder section by `natural_order`

        Parameters
        ----------
        section_index
            Section index to restrict the ordering to a specific section.
        """

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

        self.elements = list(
            elements_before_section + ordered_elements + elements_after_section
        )

    def count(self, class_: Type[T], section_i: Optional[int] = None):
        """
        Count instances in this class that match class-type

        Parameters
        ----------
        class_
            The class type to filter elements by.
        section_i
            Optional section index to restrict the search to a specific section.

        """
        return len(self.get_elements(class_=class_, section_i=section_i))

    def print_order(self):
        """Print current execution order"""
        print(self.get_order_info())

    def get_order_info(self):
        """Generate execution order string

        Notes
        -----
        Intended for logging and printing
        """
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


def pprint(v: NumpyArray | Any):
    """Pretty print an array"""
    if isinstance(v, np.ndarray):
        return f"array(min={v.min()}, max={v.max()}, shape={v.shape})"
    else:
        return v
