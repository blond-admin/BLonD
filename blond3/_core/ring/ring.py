from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..backends.backend import backend
from ..base import (
    BeamPhysicsRelevant,
    Preparable,
    Schedulable,
)
from ..beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable, Optional
    from .beam_physics_relevant_elements import BeamPhysicsRelevantElements

    from ..simulation.simulation import Simulation


class Ring(Preparable, Schedulable):
    _bending_radius: np.float32 | np.float64
    _circumference: np.float32 | np.float64

    def __init__(
        self,
        circumference: float,
        bending_radius: Optional[float] = None,
    ) -> None:
        """Ring a.k.a. synchrotron

        Parameters
        ----------
        circumference
            Synchrotron circumference in [m]
        bending_radius
            Optional bending radius in [m]
            If not specified, ring will be assumed perfectly round.
        """
        from .beam_physics_relevant_elements import BeamPhysicsRelevantElements

        if bending_radius is None:
            bending_radius = circumference / (2 * np.pi)

        super().__init__()
        self._elements = BeamPhysicsRelevantElements()

        self._circumference = backend.float(circumference)
        self._bending_radius = backend.float(bending_radius)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager"""
        from ...physics.drifts import DriftBaseClass  # prevent cyclic import

        all_drifts = self.elements.get_elements(DriftBaseClass)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )
        assert len(self.elements.get_sections_indices()) == self.n_cavities, (
            f"{len(self.elements.get_sections_indices())=}, " f"but {self.n_cavities=}"
        )
        # todo assert some kind of order inside the sections

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

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

    @property
    def n_cavities(self):
        """Total number of cavities in this synchrotron"""
        from ...physics.cavities import CavityBaseClass

        return self.elements.count(CavityBaseClass)

    @property  # as readonly attributes
    def bending_radius(self):
        """Bending radius in [m]"""
        return self._bending_radius

    @property  # as readonly attributes
    def elements(self) -> BeamPhysicsRelevantElements:
        """Bending radius in [m]"""

        return self._elements

    @property  # as readonly attributes
    def circumference(self):
        """Synchrotron circumference in [m]"""
        return self._circumference

    @property
    def section_lengths(self):
        """Length of each section in [m]"""
        return self.circumference * self.elements.get_section_circumference_shares()

    def add_element(
        self,
        element: BeamPhysicsRelevant,
        reorder: bool = False,
        deepcopy: bool = False,
        section_index: Optional[int] = None,
    ):
        """
        Append a beam physics-relevant element to the ring.

        This method appends the given element to the
        internal sequence of elements, maintaining insertion order if
        reorder is False.
        The element must have a valid integer `section_index`.

        Parameters
        ----------
        element
            An object representing a beamline component or any element
            relevant to beam physics. Must have a valid  `section_index`
            attribute of type `int`.
        reorder
            Reorder each section by `natural_order`
        deepcopy
            Takes a copy of the given element
        section_index
            Add element to section
            (overwrites section index of element)

        Raises
        ------
        AssertionError
            If `element.section_index` is not an integer.
        """
        if deepcopy:
            element = copy.deepcopy(element)
        if section_index is not None:
            element._section_index = int(section_index)
        self.elements.add_element(element)

        if reorder:
            self.elements.reorder()

    def add_elements(
        self,
        elements: Iterable[BeamPhysicsRelevant],
        reorder: bool = False,
        deepcopy: bool = False,
        section_index: Optional[int] = None,
    ):
        """
        Append beam physics-relevant elements to the ring.

        This method appends the given elements to the
        internal sequence of elements, maintaining
        insertion order if `reorder` is False.

        Parameters
        ----------
        elements
            Objects representing beamline components or other elements
            relevant to beam physics.
        reorder
            Reorder each section by `natural_order`
        deepcopy
            Takes copies of the given elements
        section_index
            Add elements to section
            (overwrites section index of elements)

        Raises
        ------
        AssertionError
            If `element.section_index` is not an integer.
        """
        for element in elements:
            self.add_element(
                element=element,
                deepcopy=deepcopy,
                section_index=section_index,
            )

        if reorder:
            self.elements.reorder()
