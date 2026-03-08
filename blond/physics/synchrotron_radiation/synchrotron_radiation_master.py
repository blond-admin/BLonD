# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Collection to include synchrotron radiation and quantum excitation effects.

Notes
-----
Authors:
L. Valle
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    calculate_isomagnetic_radiation_integrals,
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.base import Schedulable
from blond.core.beam.base import BeamBaseClass
from blond.physics.synchrotron_radiation.base import (
    SynchrotronRadiationBaseClass,
)

if TYPE_CHECKING:
    from typing import (
        TypeVar,
    )

    from numpy.typing import NDArray as NumpyArray

    from blond.core.ring.ring import Ring
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import RFStationBaseClass
    from blond.physics.drifts import DriftBaseClass

    T = TypeVar("T")

# TODO allow schedulable synchrotron radiation integrals in the master (e.g.
# tapering)


class SynchrotronRadiationMaster(Schedulable):
    """
    Master class for enabling synchrotron radiation along the ring.

    This class prepares a Ring object for synchrotron radiation tracking
    with the method prepare_ring_for_synchrotron_radiation_tracking().
    This method either sets the synchrotron radiation to use (either from
    the ring, as input or computes the isomagnetic radiation integrals)
    before generating and inserting the BeamPhysicsRelevant elements in the
    ring.

    Parameters
    ----------
    track_before_element_type
        BeamPhysicsRelevant element class for which synchrotron radiation
        should be tracked.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from blond.physics.drifts import DriftBaseClass
    >>> from blond import Ring,
    >>> from blond.physics.synchrotron_radiation.synchrotron_radiation_master import SynchrotronRadiationMaster
    >>>
    >>> ring = Ring(
    ...     circumference=10,
    ...     radiation_integrals=np.array(
    ...         [
    ...             0.646747216157,
    ...             0.0005936549319,
    ...             5.6814536525e-08,
    ...             5.92870407301e-09,
    ...             1.71368060083e-11,
    ...         ]
    ...     ),
    ... )
    >>> ring.add_drifts(n_drifts_per_section=10, n_sections=4)
    >>> SRM = SynchrotronRadiationMaster(track_before_element_type=[DriftBaseClass])
    >>> SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)
    """

    def __init__(
        self,
        track_before_element_type: list[
            type[DriftBaseClass | RFStationBaseClass]
        ]
        | None = None,
        disable_quantum_excitation: bool = False,
    ):
        from blond.physics.drifts import (
            DriftBaseClass,  # prevent cyclic import
        )

        super().__init__()

        if track_before_element_type is not None:
            self.track_before_element_type = track_before_element_type
        else:
            self.track_before_element_type = [
                DriftBaseClass,
            ]

        self._simulation: Simulation | None = None
        self._disable_quantum_excitation = disable_quantum_excitation

        self._natural_energy_spread: NumpyArray | None = None
        self._energy_loss_per_turn: NumpyArray | None = None
        self._longitudinal_damping_time: NumpyArray | None = None

        self.generated_children: list[SynchrotronRadiationBaseClass] = []

    def __str__(self) -> str:
        """
        Method to print general information about the created class.

        Returns
        -------
        message
            Prints the characteristics of the initialised
            SynchrotronRadiationMaster class.
        """
        return (
            f"Synchrotron radiation master class set up for the"
            f" ring. \n Generated "
            f"{self.number_of_generated_synchrotron_radiation_classes} "
            f"synchrotron radiation elements."
        )

    @property
    def energy_loss_per_turn(self) -> NumpyArray | None:
        """
        Energy loss per turn, in [eV per turn].

        Returns
        -------
        energy_loss_per_turn
            Energy loss per turn, in [eV per turn].
        """
        return self._energy_loss_per_turn

    @property
    def longitudinal_damping_time(self) -> NumpyArray | None:
        """
        Longitudinal damping time.

        Returns
        -------
        longitudinal_damping_time
            Damping times, in [turn].
        """
        return self._longitudinal_damping_time

    @property
    def natural_energy_spread(self) -> NumpyArray | None:
        """
        Natural energy spread.

        Returns
        -------
        natural_energy_spread
            Natural energy spread, [dimensionless].
        """
        return self._natural_energy_spread

    @property
    def number_of_generated_synchrotron_radiation_classes(self) -> int:
        """
        Number of generated synchrotron radiation classes.

        Returns
        -------
        number_of_generated_synchrotron_radiation_classes
            Number of generated synchrotron_radiation_classes.
        """
        return len(self.generated_children)

    def print_synchrotron_radiation_parameters(
        self,
        ring: Ring,
        beam: BeamBaseClass,
    ) -> str:
        """
        Print the synchrotron radiation parameter of a given turn.

        This function computes and prints the synchrotron radiation
        characteristics of the (beam, ring) pair.

        Parameters
        ----------
        ring
            `Ring` context manager.
        beam
            `Beam` object.

        Returns
        -------
        message
            Prints the characteristics of radiation damping for the given
            beam and ring.
        """
        self.compute_synchrotron_radiation_parameters(ring=ring, beam=beam)

        return (
            f"Synchrotron radiation parameters for the beam energy "
            f"{beam.reference.total_energy}"
            + f"Energy lost: {self.energy_loss_per_turn} eV per turn,"
            + f"Longitudinal damping time:"
            f" {self.longitudinal_damping_time} turns,"
            + f"Natural energy spread: {self._natural_energy_spread}"
        )

    def compute_synchrotron_radiation_parameters(
        self,
        ring: Ring,
        beam: BeamBaseClass,
    ) -> None:
        """
        Calculate the synchrotron radiation parameters for a given beam energy.

        Parameters
        ----------
        ring
            `Ring` context manager.
        beam
            `Beam` object.
        """
        (
            self._energy_loss_per_turn,
            self._longitudinal_damping_time,
            self._natural_energy_spread,
        ) = gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=beam.particle_type,
            energy=beam.reference.total_energy,
            radiation_integrals=ring.radiation_integrals,
        )

    def _user_warning_set_radiation_integrals(
        self,
        radiation_integrals: NumpyArray | None = None,
        bending_radius: float | None = None,
    ) -> None:
        """
        Internal method for user warnings.

        Parameters
        ----------
        radiation_integrals
            Synchrotron radiation integrals. If None, the ring will be
            considered isomagnetic.
            In the case of an isomagnetic ring, the synchrotron radiation
            integrals will be computed from the ring bending radius. Default:
            False.
        bending_radius
            Averaged bending radius along the ring, in [m].
        """
        if radiation_integrals is not None:
            warnings.warn(
                category=UserWarning,
                message="Radiation integrals input ignored. Using the ring's.",
                stacklevel=2,
            )
        if bending_radius:
            warnings.warn(
                category=UserWarning,
                message="Bending radius input ignored. "
                "Using the ring's radiation integrals.",
                stacklevel=2,
            )

    def _radiation_integrals_internal_setter(
        self,
        ring: Ring,
        radiation_integrals: NumpyArray | None = None,
        bending_radius: float | None = None,
    ):
        """
        Internal method to calculate the radiation integrals.

        Parameters
        ----------
        ring
            `Ring` context manager.
        radiation_integrals
            Synchrotron radiation integrals. If None, the ring will be
            considered isomagnetic.
            In the case of an isomagnetic ring, the synchrotron radiation
            integrals will be computed from the ring bending radius. Default:
            False.
        bending_radius
            Averaged bending radius along the ring, in [m].

        Returns
        -------
        integrals_to_use
            Radiation integrals to use.
        """
        minimum_number_of_expected_radiation_integrals = 5
        if radiation_integrals is None:
            if isinstance(bending_radius, float | int):
                integrals_to_use = calculate_isomagnetic_radiation_integrals(
                    circumference=ring.circumference,
                    bending_radius=bending_radius,
                    momentum_compaction_factor=ring.momentum_compaction_factor,
                )
            else:
                raise ValueError(
                    "Synchrotron radiation damping "
                    "and quantum excitation require"
                    " either the bending radius for an isomagnetic ring, or the "
                    "first five synchrotron radiation "
                    "integrals."
                )
        elif isinstance(radiation_integrals, list | np.ndarray):
            try:
                integrals = np.array(radiation_integrals)
            except ValueError as ve:
                raise ValueError(
                    "Could not transform the input into an array."
                ) from ve
            if (
                len(integrals)
                >= minimum_number_of_expected_radiation_integrals
            ):
                integrals_to_use = integrals
            else:
                raise ValueError(
                    "The first five synchrotron radiation integrals are "
                    "required."
                )
        else:
            raise TypeError(
                f"Expected a list or numpy.ndarray as an input. Received"
                f" {type(radiation_integrals)}."
            )
        return integrals_to_use

    def _set_radiation_integrals(
        self,
        ring: Ring,
        radiation_integrals: NumpyArray | None = None,
        bending_radius: float | None = None,
    ) -> None:
        """
        Set the radiation integrals of the SynchrotronRadiationMaster class.

        This function sets the radiation integrals in the Ring object if
        non-existent.

        Parameters
        ----------
        ring
            `Ring` context manager.
        radiation_integrals
            Synchrotron radiation integrals. If None, the ring will be
            considered isomagnetic.
            In the case of an isomagnetic ring, the synchrotron radiation
            integrals will be computed from the ring bending radius. Default:
            False.
        bending_radius
            Averaged bending radius along the ring, in [m].
        """
        if ring.radiation_integrals is not None:
            ring.assert_radiation_integrals()
            self._user_warning_set_radiation_integrals(
                radiation_integrals=radiation_integrals,
                bending_radius=bending_radius,
            )
        else:
            integrals_to_use = self._radiation_integrals_internal_setter(
                ring=ring,
                radiation_integrals=radiation_integrals,
                bending_radius=bending_radius,
            )
            ring._radiation_integrals = integrals_to_use

    def _get_share_of_radiation_integrals_drifts(
        self,
        ring: Ring,
        drift_list: list[type[DriftBaseClass]],
    ) -> list[NumpyArray]:
        """
        Distribute the radiation integrals for drift tracker.

        Parameters
        ----------
        ring
            `Ring` context manager.
        drift_list
            DriftBaseClass element list.

        Returns
        -------
        share_of_radiation_integrals
            Share of synchrotron radiation integrals.
        """
        shares_of_radiation_integrals = []

        drift_list_ = (
            drift.radiation_integrals is not None for drift in drift_list
        )
        drifts_with_radiation_integrals = any(drift_list_)

        if drifts_with_radiation_integrals:
            use_radiation_integrals_from_drifts = all(drift_list_)
            if not use_radiation_integrals_from_drifts:
                raise ValueError(
                    "Either all drifts should have defined radiation "
                    f"integrals or none should, but got {drift_list_}."
                )
        else:
            use_radiation_integrals_from_drifts = False

        for drift in drift_list:
            if use_radiation_integrals_from_drifts:
                shares_of_radiation_integrals.append(drift.radiation_integrals)
            else:
                shares_of_radiation_integrals.append(
                    drift.orbit_length
                    / ring.circumference
                    * ring.radiation_integrals
                )
        return shares_of_radiation_integrals

    def _get_share_of_radiation_integrals_cavities(
        self,
        ring: Ring,
        cavity_list: list[type[RFStationBaseClass]],
    ) -> list[NumpyArray]:
        """
        Distribute the synchrotron radiation integrals for cavity trackers.

        Parameters
        ----------
        ring
            `Ring` context manager.
        cavity_list
            RFStationBaseClass element list.

        Returns
        -------
        share_of_radiation_integrals
            Share of synchrotron radiation integrals.
        """
        cavities_section_indexes = [e.section_index for e in cavity_list]
        shares_of_radiation_integrals = []
        for i, cavity in enumerate(cavity_list):
            if len(cavity_list) == 1:
                section_length_to_consider = ring.circumference
            elif cavity.section_index == len(ring.section_lengths) - 1:
                section_length_to_consider = ring.section_lengths[-1]
            else:
                section_length_to_consider = np.sum(
                    ring.section_lengths[
                        cavities_section_indexes[i] : cavities_section_indexes[
                            i + 1
                        ]
                    ]
                )
            shares_of_radiation_integrals.append(
                section_length_to_consider
                / ring.circumference
                * ring.radiation_integrals
            )
        return shares_of_radiation_integrals

    def _generate_radiation_trackers(
        self,
        ring: Ring,
        element_list: list[type[RFStationBaseClass | DriftBaseClass]],
    ) -> None:
        """
        Function to create and insert the SR trackers in the ring.

        This function inserts `SynchrotronRadiationBaseClass` elements in the
        ring either:
        - before the drifts if track_before_element_type is ``None``
          or `DriftBaseClass`. In that case, `_SynchrotronRadiationDrift`
         trackers will be inserted in the ring before each drift.
        - after the RF cavities if track_before_element_type is `RFStationBaseClass`.
          In that case, `_SynchrotronRadiationSection` trackers will be
          inserted in the ring.

        Parameters
        ----------
        ring
            `Ring` context manager.
        element_list
            Element list to consider.
        """
        from blond.physics.cavities import (
            RFStationBaseClass,  # prevent cyclic import
        )
        from blond.physics.drifts import (
            DriftBaseClass,  # prevent cyclic import
        )

        if all(isinstance(e, DriftBaseClass) for e in element_list):
            # _SynchrotronRadiationDrift tracker placed before the
            # drift
            shares_of_radiation_integrals = (
                self._get_share_of_radiation_integrals_drifts(
                    ring=ring,
                    drift_list=element_list,
                )
            )
            self._insert_radiation_trackers(
                ring=ring,
                element_list=element_list,
                shares_of_radiation_integrals=shares_of_radiation_integrals,
                after_element=False,  # tracker inserted before the drift
            )
        elif all(isinstance(e, RFStationBaseClass) for e in element_list):
            shares_of_radiation_integrals = (
                self._get_share_of_radiation_integrals_cavities(
                    ring=ring,
                    cavity_list=element_list,
                )
            )
            self._insert_radiation_trackers(
                ring=ring,
                element_list=element_list,
                shares_of_radiation_integrals=shares_of_radiation_integrals,
                after_element=True,  # tracker inserted after the cavity
            )
        else:
            raise TypeError(
                "Unsupported list of elements. Full lists of "
                "DriftBaseClass and RFStationBaseClass are "
                f"allowed, but {element_list} was found."
            )

    def _insert_radiation_trackers(
        self,
        ring: Ring,
        element_list: list[DriftBaseClass | RFStationBaseClass],
        shares_of_radiation_integrals: list[NumpyArray],
        after_element: bool,
    ):
        """
        Insert the radiation trackers in the ring.

        Parameters
        ----------
        ring
            `Ring` context manager.
        element_list
            `DriftBaseClass` of `RFStationBaseClass` element list.
        shares_of_radiation_integrals
            Share of synchrotron radiation integrals.
        after_element
            If enabled, the tracker will be places after the elements.
        """
        for i, element in enumerate(element_list):
            SRClass_child = _SynchrotronRadiationTracker(
                section_index=element._section_index,
                name=f"SynchrotronRadiationTracker_"
                f"{len(self.generated_children) + 1}",
                share_of_radiation_integrals=shares_of_radiation_integrals[i],
                disable_quantum_excitation=self._disable_quantum_excitation,
            )
            shift_location = int(after_element == True)
            ring.insert_element(
                element=SRClass_child,
                insert_at=ring.elements.elements.index(element)
                + shift_location,
                deepcopy=False,  # to maintain the consistency
                # between the stored array and the ring elements
            )
            self.generated_children.append(SRClass_child)

    def prepare_ring_for_synchrotron_radiation_tracking(
        self,
        ring: Ring,
        radiation_integrals: NumpyArray | None = None,
        bending_radius: float | None = None,
    ) -> None:
        """
        Function to create synchrotron radiation elements in the ring.

        This method automatically creates, inserts and initialises the
        synchrotron radiation elements in the ring.

        Parameters
        ----------
        ring
            `Ring` context manager.
        radiation_integrals
            Synchrotron radiation integrals. If None, the ring will be
            considered isomagnetic.
            In the case of an isomagnetic ring, the synchrotron radiation
            integrals will be computed from the ring bending radius. Default:
            None.
        bending_radius
            Averaged bending radius along the ring, in [m].
        """
        self._set_radiation_integrals(
            ring=ring,
            radiation_integrals=radiation_integrals,
            bending_radius=bending_radius,
        )

        if self.generated_children:
            warnings.warn(
                "Synchrotron radiation subclasses have already been "
                "generated. Command ignored",
                UserWarning,
                stacklevel=2,
            )
        else:
            element_list = []
            for element_class in self.track_before_element_type:
                element_list += ring.elements.get_elements(
                    class_=element_class
                )
            if not element_list:
                raise TypeError(
                    f"Empty element list for class "
                    f"{self.track_before_element_type}"
                )
            else:
                self._generate_radiation_trackers(
                    ring=ring, element_list=element_list
                )


class _SynchrotronRadiationTracker(SynchrotronRadiationBaseClass):
    """
    Class to track the effect on synchrotron radiation in a ring.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_radiation_integrals
        Share of synchrotron radiation integrals.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int = 0,
        share_of_radiation_integrals: NumpyArray | None = None,
        disable_quantum_excitation: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            share_of_radiation_integrals=share_of_radiation_integrals,
            disable_quantum_excitation=disable_quantum_excitation,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_tracker(self) -> float | None:
        """
        Energy lost by passing through the arc covered by the tracker.

        Returns
        -------
        energy_lost_due_to_synchrotron_radiation_drift
            Energy lost due to synchrotron radiation along the drift,
            in [eV per turn].
        """
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def share_of_radiation_integrals(self) -> NumpyArray | None:
        """
        Share of radiation integrals.

        Returns
        -------
        share_of_radiation_integrals
            Synchrotron radiation integrals of the tracker.
        """
        return self._share_of_radiation_integrals

    @property
    def radiation_integrals_tracker(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals of the arc covered by the tracker.

        Returns
        -------
        radiation_integrals_tracker
            Synchrotron radiation integrals of the tracker.
        """
        return self._share_of_radiation_integrals
