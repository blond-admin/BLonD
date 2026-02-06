# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds the `Ring` object."""

from __future__ import annotations

import copy
import logging
import warnings
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import Preparable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond.core.base import SimulationElementBase
    from blond.core.beam.base import BeamBaseClass
    from blond.core.ring.beam_physics_relevant_elements import (
        BeamPhysicsRelevantElements,
    )
    from blond.core.simulation.simulation import Simulation
    from blond.physics.drifts import DriftBaseClass

logger = logging.getLogger(__name__)


class Ring(Preparable):
    r"""
    Create a `Ring` object representing a synchrotron accelerator.

    A Ring (synchrotron) is the fundamental structure that contains all beam
    physics elements like RF stations, drifts, and other components. It maintains
    a reference circumference used for RF frequency calculations.

    Parameters
    ----------
    circumference
        The reference circumference of the synchrotron, in [m].
        This value remains constant during simulation and is used to determine
        the RF frequency program. Note: While the actual orbit length may vary
        during simulation (e.g., due to energy changes), the circumference stays
        fixed. Orbit length changes result in timing delays but don't affect
        the RF frequency program.
    check_section_indices : bool, optional
        If True, validate section indices during initialization.
        Default is True.
    synchrotron_radiation_integrals
            Synchrotron radiation integrals.
            First five synchrotron radiation integrals are required:
            'I_1' = \int, related to the momentum compaction factor,
            'I_2' = , related to the energy loss per turn,
            'I_3' = , related to the natural energy spread,
            'I_4' =  , required for the damping times,
            'I_5' =  , required for the natural horizontal emittance
            with '\rho' the bending radius of bending elements, 'D' the
            horizontal dispersion function, 'K' the focusing strength and 'H =
            \beta_x D^2 + \alpha_x D {D'} + \gamma_x {D'}^2 ' the
            H-function.

    References
    ----------
    Further information on synchrotron radiation damping and quantum excitation
    can be found in:
    - H. Wiedemann, Synchrotron Radiation, Springer, 2003
    - S.Y. Lee, Accelerator Physics, World Scientific, Third edition,
    2014 #check date
    - A. Wolski, Introduction to Beam Dynamics in High-Energy Electron Storage
    Rings, Morgan & Claypool Publishers, 2018
    """

    def __init__(
        self,
        circumference: float,
        check_section_indices: bool = True,
        synchrotron_radiation_integrals: NumpyArray | None = None,
    ) -> None:
        from blond.core.ring.beam_physics_relevant_elements import (
            BeamPhysicsRelevantElements,
        )

        super().__init__()
        self._elements = BeamPhysicsRelevantElements(
            check_section_indices=check_section_indices
        )
        assert circumference > 0, (
            f"`circumference` must be bigger 0, but is {circumference}"
        )
        self._circumference = circumference
        self._synchrotron_radiation_integrals = synchrotron_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Initialize the ring when a simulation is created.

        This method is automatically called during simulation initialization to
        validate the ring configuration. It checks that RF stations are properly
        configured and section indices are correctly ordered.

        Parameters
        ----------
        simulation
            The `Simulation` context manager that owns this ring.
        """
        if self.n_rf_stations > 1:
            assert (
                len(self.elements.get_sections_indices()) == self.n_rf_stations
            ), (
                f"{len(self.elements.get_sections_indices())=}, but {self.n_rf_stations=}"
            )
        elif self.n_rf_stations == 0:
            warnings.warn(
                "The simulation has been initialized without RF station.",
                UserWarning,
                stacklevel=1,
            )

        if np.any(
            np.diff([e.section_index for e in self.elements.elements]) < 0
        ):
            warnings.warn(
                "Section indices must be ascending, but section order:"
                f" {[e.section_index for e in self.elements.elements]=}",
                UserWarning,
                stacklevel=1,
            )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Prepare the ring when simulation execution begins.

        This method is automatically called
        when ``simulation.run_simulation()`` starts.

        Parameters
        ----------
        simulation
            The `Simulation` context manager.
        beam
            The beam object being simulated.
        n_turns
            Total number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        pass

    @property
    def circumference(self) -> float:
        """
        Get the reference circumference of the synchrotron, in [m].

        Returns
        -------
        circumference
            The reference circumference, in [m].

        Notes
        -----
        This value remains constant throughout the simulation and determines the
        RF frequency program. The actual orbit length (see `closed_orbit_length`)
        may differ during simulation, leading to timing delays without changing
        the RF frequency.
        """
        return self._circumference

    @property
    def synchrotron_radiation_integrals(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals of the ring.

        Returns
        -------
        synchrotron_radiation_integrals
            Synchrotron radiation integrals.

        References
        ----------
        Further information on synchrotron radiation damping and quantum excitation
        can be found in:
        - H. Wiedemann, Synchrotron Radiation, Springer, 2003
        - S.Y. Lee, Accelerator Physics, World Scientific, Third edition,
        2014 #check date
        - A. Wolski, Introduction to Beam Dynamics in High-Energy Electron Storage
        Rings, Morgan & Claypool Publishers, 2018
        """
        return self._synchrotron_radiation_integrals

    @property
    def momentum_compaction_factor(self) -> float:
        r"""
        Calculate the momentum compaction factor.

        This property computes the ring momentum compaction factor by
        summing the contribution of all the drifts to recover:
        \alpha_C = 1/C * \oint (frac{ds}{\rho})
        where C is the ring circumference, and \rho its bending radius.

        Returns
        -------
        momentum_compaction_factor
            Ring momentum compaction factor.
        """
        from blond.physics.drifts import DriftBaseClass

        momentum_compaction_factor_contributions = np.array(
            [
                e.momentum_compaction_factor
                for e in self.elements.get_elements(class_=DriftBaseClass)
            ]
        )

        momentum_compaction_factor = np.sum(
            momentum_compaction_factor_contributions
        )
        return momentum_compaction_factor

    @cached_property
    def average_transition_gamma(self) -> complex:
        """
        Calculate the orbit-length weighted average transition gamma.

        The transition gamma is the Lorentz factor at which particles cross from
        below to above transition energy. This property computes a weighted average
        based on the drift sections in the ring.

        Returns
        -------
        average_transition_gamma
            The weighted average transition gamma (dimensionless).

        Notes
        -----
        Currently only considers DriftSimple elements. The weighting is based on
        the orbit length of each drift section. This value is cached after first
        calculation.
        """
        from blond import DriftSimple  # prevent cyclic import

        gammas = [
            e.transition_gamma
            for e in self.elements.get_elements(DriftSimple, recursive=False)
        ]
        weights = [
            e.orbit_length
            for e in self.elements.get_elements(DriftSimple, recursive=False)
        ]
        # todo not only simple drift
        transition_gamma_average = complex(np.average(gammas, weights=weights))
        return transition_gamma_average

    def calc_average_eta_0(self, gamma: float) -> float:
        """
        Calculate the orbit-length weighted average slip factor `eta_0`.

        The slip factor `eta_0` describes the relationship between relative momentum
        deviation and relative path length change. This method computes a weighted
        average over all drift sections in the ring at a given beam energy.

        Parameters
        ----------
        gamma
            The relativistic Lorentz factor at which to evaluate
            the slip factor.

        Returns
        -------
        average_eta_0
            The weighted average slip factor (dimensionless).

        See Also
        --------
        eta_0 : Internally used for calculation.
        """
        from blond.physics.drifts import (
            DriftBaseClass,  # prevent circular import
        )

        drifts = self.elements.get_elements(DriftBaseClass, recursive=False)
        weights = [d.orbit_length for d in drifts]
        etas = [d.eta_0(gamma) for d in drifts]
        return float(
            np.average(
                etas,
                weights=weights,
            )
        )

    def is_below_transition(self, beam: BeamBaseClass) -> bool:
        """
        Check if the beam velocity is below the transition energy.

        Below transition, higher energy particles take longer to complete an orbit.
        Above transition, higher energy particles complete orbits faster. This
        distinction is crucial for longitudinal beam dynamics and RF phase settings.

        Parameters
        ----------
        beam
            The beam object containing the reference gamma value to check.

        Returns
        -------
        bool
            True if the beam is below transition (gamma < gamma_transition),
            False otherwise.

        See Also
        --------
        average_transition_gamma : This method is internally used.
        """
        return bool(self.calc_average_eta_0(gamma=beam.reference.gamma) < 0)

    @property
    def n_rf_stations(self) -> int:
        """
        Get the total number of RF stations in the ring.

        Returns
        -------
        n_rf_stations
            The count of all RF station elements currently in the ring.
        """
        from blond.physics.cavities import RFStationBaseClass

        return self.elements.count(RFStationBaseClass)

    @property  # as readonly attributes
    def elements(self) -> BeamPhysicsRelevantElements:
        """
        Get the container of all beam physics elements in the ring.

        Returns
        -------
        elements
            A container holding all elements (RF stations, drifts, monitors, etc.)
            that affect beam physics in this ring.
        """
        return self._elements

    @property  # as readonly attributes
    def closed_orbit_length(self) -> float:
        """
        Get the actual closed orbit length, in [m].

        Returns
        -------
        closed_orbit_length
            The sum of all drift orbit lengths, in [m].

        Notes
        -----
        This is calculated by summing the `orbit_length` of all drift elements and
        may differ from the reference circumference during simulation.
        """
        from blond.physics.drifts import DriftBaseClass

        all_drifts = self.elements.get_elements(
            DriftBaseClass, recursive=False
        )
        orbit_length = float(sum([drift.orbit_length for drift in all_drifts]))
        return orbit_length

    @property
    def section_lengths(self) -> NumpyArray:
        """
        Get the orbit length of each section in the ring, in [m].

        Returns
        -------
        section_lengths
            Array containing the orbit length of each section, in [m].
        """
        return self.elements.get_sections_orbit_length()

    def assert_circumference(
        self,
        atol: float = 1e-6,
    ) -> None:
        """
        Verify that the closed orbit length matches the reference circumference.

        This method checks that the sum of all drift orbit lengths equals the
        reference circumference within the specified tolerance. Use this to
        validate your ring configuration.

        Parameters
        ----------
        atol
            The absolute tolerance for the comparison, in [m].
            Default is 1e-6 m (1 micrometer).

        Raises
        ------
        AssertionError
            If the closed orbit length differs from the reference circumference
            by more than `atol`.

        Examples
        --------
        >>> from blond import Ring, DriftSimple
        >>> ring = Ring(123)
        >>> ring.add_element(DriftSimple(orbit_length=ring.circumference))
        >>> ring.assert_circumference()  # Check with default tolerance
        >>> ring.assert_circumference(atol=1e-3)  # Check with 1 mm tolerance
        """
        if self.closed_orbit_length == 0 and self.circumference > 0:
            raise ValueError(
                f"{self.closed_orbit_length=} m. "
                f"Did you forget to add drift elements?"
            )
        assert np.isclose(
            self.closed_orbit_length,
            self.circumference,
            atol=atol,
        ), (
            f"{self.closed_orbit_length=} m,"
            f" but should be {self.circumference=} m."
        )

    def add_drifts(
        self,
        n_drifts_per_section: int,
        n_sections: int,
        driftclass: type[DriftBaseClass] | None = None,
        **kwargs_drift,
    ) -> None:
        """
        Add uniformly distributed drift sections to the ring.

        This convenience method creates and adds drift elements distributed equally
        across multiple sections. Each drift will have the same length, calculated
        to fill the entire circumference.

        Parameters
        ----------
        n_drifts_per_section
            Number of drift elements to create in each section.
        n_sections
            Total number of sections in the ring.
        driftclass
            The drift class to instantiate. If None, uses `DriftSimple`.
        **kwargs_drift
            Additional keyword arguments passed to the drift constructor
            (e.g., `transition_gamma`, `bending_radius`).

        Examples
        --------
        >>> from blond import Ring
        >>> ring = Ring(...)
        >>> ring.add_drifts(n_drifts_per_section=10, n_sections=4)
        >>> # Creates 40 drifts total, 10 per section, each with length = circumference/40
        """
        if driftclass is None:
            from blond import DriftSimple  # prevent cyclic import

            driftclass = DriftSimple

        n_drifts = n_drifts_per_section * n_sections
        length_per_drift = self.circumference / n_drifts
        for section_i in range(n_sections):
            for _drift_i in range(n_drifts_per_section):
                drift = driftclass(
                    orbit_length=length_per_drift,
                    section_index=section_i,
                    **kwargs_drift,
                )
                self.elements.add_element(drift)

    def add_element(
        self,
        element: SimulationElementBase,
        reorder: bool = False,
        deepcopy: bool = False,
        section_index: int | None = None,
    ):
        """
        Add a single element to the end of the ring.

        This is the primary method for adding beam physics elements (RF stations,
        drifts, monitors, etc.) to the ring. Elements are appended in the order
        they are added unless reordering is enabled.

        Parameters
        ----------
        element
            A beam physics element (RF station, drift, monitor, etc.) to add.
            Must have a valid integer `section_index` attribute.
        reorder
            If True, automatically reorder elements within each section by their
            `natural_order` attribute after adding. Default is False.
        deepcopy
            If True, adds a deep copy of the element instead of the original
            reference. Useful when reusing element templates. Default is False.
        section_index
            If provided, overrides the element's existing section_index attribute.
            Useful for programmatically assigning sections.

        Raises
        ------
        AssertionError
            If the element's section_index is not a valid integer.

        Examples
        --------
        >>> from blond import MultiHarmonicRFStation, Ring
        >>> ring = Ring(...)
        >>> rf_station = MultiHarmonicRFStation(voltage=1e6, harmonic=400, section_index=0)
        >>> ring.add_element(rf_station)
        """
        if deepcopy:
            element = copy.deepcopy(element)
        if section_index is not None:
            element._section_index = int(section_index)
        self.elements.add_element(element, reorder=reorder)

        if reorder:
            self.elements.reorder()

    def add_elements(
        self,
        elements: Iterable[SimulationElementBase],
        reorder: bool = False,
        deepcopy: bool = False,
        section_index: int | None = None,
    ):
        """
        Add multiple elements to the ring at once.

        Convenience method for adding several beam physics elements in a single call.
        Elements are added in the order they appear in the iterable.

        Parameters
        ----------
        elements
            A collection (list, tuple, etc.) of beam physics elements to add.
            Each element must have a valid integer `section_index` attribute.
        reorder
            If True, reorder elements within each section by their `natural_order`
            after adding all elements. Default is False.
        deepcopy
            If True, adds deep copies of the elements. Default is False.
        section_index
            If provided, overrides the section_index for all elements being added.

        Raises
        ------
        AssertionError
            If any element's section_index is not a valid integer.

        Examples
        --------
        >>> from blond import Ring, SingleHarmonicRFStation
        >>> ring = Ring(...)
        >>> rf_stations = [SingleHarmonicRFStation(section_index=i, ...) for i in range(4)]
        >>> ring.add_elements(rf_stations)
        """
        for element in elements:
            logger.info(f"Adding {element}")
            self.add_element(
                element=element,
                deepcopy=deepcopy,
                section_index=section_index,
            )

        if reorder:
            self.elements.reorder()

    def insert_element(
        self,
        element: SimulationElementBase,
        insert_at: int | list[int],
        deepcopy: bool = True,
        allow_section_index_overwrite: bool = False,
    ) -> list[int]:
        """
        Insert an element at specific position(s) in the ring.

        Unlike `add_element` which appends to the end, this method inserts an
        element at precise locations within the existing element sequence. Useful
        for adding elements at specific positions.

        Parameters
        ----------
        element
            The element to insert. Must have a valid integer `section_index`.
        insert_at
            Position(s) where the element should be inserted. Can be a single
            index or list of indices. After insertion at position k,
            ring.elements.elements[k] will be the inserted element.
        deepcopy
            Whether to insert copies of the element (recommended when inserting
            at multiple locations). Default is True.
        allow_section_index_overwrite
            If True, automatically adjusts the element's section_index to match
            the section at the insertion position. Requires deepcopy=True.
            Default is False.

        Returns
        -------
        list[int]
            The final positions of the inserted element(s) in the modified ring.

        Raises
        ------
        AssertionError
            - If element.section_index is not an integer
            - If insert_at is outside valid range [0, len(elements)]
            - If element.section_index doesn't match the insertion section
              (unless allow_section_index_overwrite=True)
            - If allow_section_index_overwrite=True but deepcopy=False

        Examples
        --------
        >>> from blond import Ring, DriftSimple
        >>> ring = Ring(...)
        >>> drift = DriftSimple(section_index=0, ...)
        >>> ring.insert_element(drift, insert_at=5)
        [5]
        """
        locations_in_the_new_ring = []
        if isinstance(insert_at, int):
            insert_at = [insert_at]

        if deepcopy:
            for already_inserted, k in enumerate(insert_at):
                element = copy.deepcopy(element)
                if allow_section_index_overwrite:
                    element = self._force_section_index_compatibility(
                        element=element,
                        insert_at=k + already_inserted,
                    )
                logger.info(f"Adding {element}")

                self.elements.insert(
                    element=element,
                    insert_at=k + already_inserted,
                )
                locations_in_the_new_ring.append(k + already_inserted)
        else:
            if allow_section_index_overwrite:
                raise ValueError(
                    "Cannot overwrite the section indexes with deepcopy == False."
                )
            for already_inserted, k in enumerate(insert_at):
                logger.info(f"Adding {element}")

                self.elements.insert(
                    element=element,
                    insert_at=k + already_inserted,
                )
                locations_in_the_new_ring.append(k + already_inserted)

        return locations_in_the_new_ring

    def insert_elements(
        self,
        elements: list[SimulationElementBase],
        insert_at: int,
        deepcopy: bool = True,
        allow_section_index_overwrite: bool = False,
    ):
        """
        Insert multiple elements at a specific position in the ring.

        This method inserts a list of elements as a contiguous block at the
        specified position. The elements maintain their order from the input list.

        After insertion:
        - ring.elements.elements[insert_at] == elements[0]
        - ring.elements.elements[insert_at:insert_at+len(elements)] == elements

        Parameters
        ----------
        elements
            List of elements to insert. Each must have a valid integer
            `section_index` attribute. They will be inserted in order.
        insert_at
            The position where the first element should be inserted.
        deepcopy
            Whether to insert copies of the elements. Default is True.
        allow_section_index_overwrite
            If True, automatically adjusts each element's section_index to match
            the section at its insertion position. Requires deepcopy=True.
            Default is False.

        Raises
        ------
        AssertionError
            - If any element's section_index is not an integer
            - If element section indices don't match the insertion section
              (unless allow_section_index_overwrite=True)
            - If insert_at is outside valid range [0, len(elements)]
            - If allow_section_index_overwrite=True but deepcopy=False

        Examples
        --------
        >>> from blond import Ring, DriftSimple
        >>> ring = Ring(...)
        >>> drifts = [DriftSimple(section_index=0, ...) for _ in range(3)]
        >>> ring.insert_elements(drifts, insert_at=10)
        """
        # The elements are inserted one by one, from the last to the first
        # to preserve the input insertion order.
        elements.reverse()
        for element in elements:
            self.insert_element(
                element=element,
                insert_at=insert_at,
                deepcopy=deepcopy,
                allow_section_index_overwrite=allow_section_index_overwrite,
            )

    def _force_section_index_compatibility(
        self, element: SimulationElementBase, insert_at: int
    ) -> SimulationElementBase:
        """
        Internal method to fix element section index for insertion.

        This private method is called by insert methods when
        `allow_section_index_overwrite=True`. It adjusts the element's `section_index`
        to match the section at the insertion position.

        The section index is copied from:
        - The element at position `insert_at` (if inserting in the middle)
        - The last element in the ring (if appending at the end)

        Parameters
        ----------
        element
            The element whose section_index needs adjustment. Must have a
            `section_index` attribute.
        insert_at
            The position where the element will be inserted.

        Returns
        -------
        SimulationElementBase
            The same element with potentially modified section_index.
        """
        if insert_at > len(self.elements.elements):
            raise IndexError(
                f"The element must be inserted within ["
                f"0:{len(self.elements.elements)}] indexes."
            )

        try:
            self.elements.check_section_index_compatibility(
                element=element,
                insert_at=insert_at,
            )
        except AssertionError:
            if insert_at == len(self.elements.elements):
                element._section_index = self.elements.elements[
                    insert_at - 1
                ].section_index
            else:
                element._section_index = self.elements.elements[
                    insert_at
                ].section_index
        # Assert that the new index works now
        self.elements.check_section_index_compatibility(
            element=element,
            insert_at=insert_at,
        )
        return element
