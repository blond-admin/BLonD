from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..base import (
    BeamPhysicsRelevant,
    Preparable,
    Schedulable,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Iterable,
        Optional,
        Type,
    )
    from .beam_physics_relevant_elements import BeamPhysicsRelevantElements
    from ...physics.drifts import DriftBaseClass
    from ..beam.base import BeamBaseClass

    from ..simulation.simulation import Simulation


class Ring(Preparable, Schedulable):
    def __init__(
        self,
    ) -> None:
        """Ring a.k.a. synchrotron"""
        from .beam_physics_relevant_elements import BeamPhysicsRelevantElements

        super().__init__()
        self._elements = BeamPhysicsRelevantElements()

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager"""

        assert len(self.elements.get_sections_indices()) == self.n_cavities, (
            f"{len(self.elements.get_sections_indices())=}, " f"but {self.n_cavities=}"
        )
        # todo assert some kind of order inside the sections

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: Type[BeamBaseClass],
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
    def elements(self) -> BeamPhysicsRelevantElements:
        """Bending radius, in [m]"""

        return self._elements

    @property  # as readonly attributes
    def closed_orbit_length(self):
        """Length of the closed orbit, in [m]"""
        from ...physics.drifts import DriftBaseClass

        all_drifts = self.elements.get_elements(DriftBaseClass)
        orbit_length = sum([drift.orbit_length for drift in all_drifts])
        return orbit_length

    @property
    def section_lengths(self):
        """Length of each section, in [m]"""
        return self.elements.get_sections_orbit_length()

    def assert_circumference(
        self,
        circumference: float,
        atol: float = 1e-6,
    ) -> None:
        """
        Checks that the sum of all drifts is equal to the circumference

        Parameters
        ----------
        circumference
            The circumference that should be inside the simulation, in [m]
        atol
            The tolerance of the check, in [m]

        Raises
        ------
        AssertionError
            If circumference != effective_circumference

        """
        assert np.isclose(
            self.closed_orbit_length,
            circumference,
            atol=atol,
        ), f"{self.closed_orbit_length=}m, but should be {circumference}m."

    def add_drifts(
        self,
        n_drifts_per_section: int,
        n_sections: int,
        total_orbit_length: float,
        driftclass: Type[DriftBaseClass] | None = None,
        **kwargs_drift,
    )-> None:
        """
        Add several drifts to the different sections


        Parameters
        ----------
        n_drifts_per_section
            Number of drifts per section
        n_sections
            Total number of sections to populate with drifts
        total_orbit_length
            The total length, i.e. the sum of all drift lengths, in [m].
            Each drift will have the same fraction of the total length.
        driftclass
            Drift class to be used.
        kwargs_drift
            Additional parameters to initialize the `driftclass`.
            Optional, only if `driftclass` supports it.

        """
        if driftclass is None:
            from ... import DriftSimple  # prevent cyclic import

            driftclass = DriftSimple

        n_drifts = n_drifts_per_section * n_sections
        length_per_drift = total_orbit_length / n_drifts
        for section_i in range(n_sections):
            for drift_i in range(n_drifts_per_section):
                drift = driftclass(
                    orbit_length=length_per_drift,
                    section_index=section_i,
                    **kwargs_drift,
                )
                self.elements.add_element(drift)

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
