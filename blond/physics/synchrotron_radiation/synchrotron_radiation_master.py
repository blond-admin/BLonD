# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Collection to include synchrotron radiation and quantum excitation effects.

First five synchrotron radiation integrals are required as an input of the
simulated ring:
            'I_1' = \int, related to the momentum compaction factor,
            'I_2' = , related to the energy loss per turn,
            'I_3' = , related to the natural energy spread,
            'I_4' =  , required for the damping times,
            'I_5' =  , required for the natural horizontal emittance
            with '\rho' the bending radius of bending elements, 'D' the
            horizontal dispersion function, 'K' the focusing strength and 'H =
            \beta_x D^2 + \alpha_x D {D'} + \gamma_x {D'}^2 ' the
            H-function

Notes
-----
Authors:
L. Valle

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
from blond.physics.cavities import RFStationBaseClass
from blond.physics.drifts import DriftBaseClass
from blond.physics.synchrotron_radiation.synchrotron_radiation_elements import (
    SynchrotronRadiationBaseClass,
    WigglerMagnet,
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
    >>> from blond import Ring, SynchrotronRadiationMaster
    >>>
    >>> ring = Ring(
    ...     circumference=10,
    ...     synchrotron_radiation_integrals=np.array(
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
        track_before_element_type: list[type[T]] | None = None,
        disable_quantum_excitation: bool = False,
    ):
        super().__init__()

        if track_before_element_type is not None:
            self.track_before_element_type = track_before_element_type
        else:
            self.track_before_element_type = [
                DriftBaseClass,
            ]

        self._simulation: Simulation | None = None
        self._disable_quantum_excitation = disable_quantum_excitation

        self._synchrotron_radiation_integrals = None
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
            Prints the characteristics of the initialised wiggler class.
        """
        return (
            f"Synchrotron radiation master class set up for the"
            f" ring. \n Generated "
            f"{self.number_of_generated_synchrotron_radiation_classes} "
            f"synchrotron radiation elements."
        )

    @property
    def synchrotron_radiation_integrals(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals.

        Returns
        -------
        synchrotron_radiation_integrals
            Synchrotron radiation integrals.
        """
        return self._synchrotron_radiation_integrals

    @property
    def energy_loss_per_turn(self) -> NumpyArray | None:
        """
        Energy loss per turn, eV per turn.

        Returns
        -------
        energy_loss_per_turn
            Energy loss per turn.
        """
        return self._energy_loss_per_turn

    @property
    def longitudinal_damping_time(self) -> NumpyArray | None:
        """
        Damping times, in turns.

        Returns
        -------
        longitudinal_damping_time
            Damping times in turn.
        """
        return self._longitudinal_damping_time

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
        beam: BeamBaseClass,
        ring: Ring,
    ) -> None:
        """
        Print the synchrotron radiation parameter of a given turn.

        This function computes and prints the synchrotron radiation
        characteristics of the (beam, ring) pair.

        Parameters
        ----------
        beam
            `Beam` object.
        ring
            `Ring` context manager.
        """
        self.compute_synchrotron_radiation_parameters(beam=beam, ring=ring)
        print(
            f"Synchrotron radiation parameters for the beam energy "
            f"#{beam.reference.total_energy}"
        )
        print("Energy lost:", self.energy_loss_per_turn)
        print(
            "Longitudinal damping time:",
            self.longitudinal_damping_time,
        )
        print("Natural energy spread:", self._natural_energy_spread)

    def compute_synchrotron_radiation_parameters(
        self,
        beam: BeamBaseClass,
        ring: Ring,
    ) -> None:
        """
        Calculate the synchrotron radiation parameters for a given beam energy.

        Parameters
        ----------
        beam
            `Beam` object.
        ring
            `Ring` context manager.
        """
        synchrotron_radiation_shift_from_wigglers = np.zeros(
            len(self._synchrotron_radiation_integrals)
        )

        wiggler_magnet_list = ring.elements.get_elements(
            class_=WigglerMagnet,
        )
        for element in wiggler_magnet_list:
            synchrotron_radiation_shift_from_wigglers += (
                element.update_synchrotron_radiation_integrals(
                    beam_reference_energy=beam.reference.total_energy,
                    calculation_only=True,
                )
            )
        (
            self._energy_loss_per_turn,
            self._longitudinal_damping_time,
            self._natural_energy_spread,
        ) = gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=beam.particle_type,
            energy=beam.reference.total_energy,
            synchrotron_radiation_integrals=self._synchrotron_radiation_integrals
            + synchrotron_radiation_shift_from_wigglers,
        )

    def _set_synchrotron_radiation_integrals(
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
            Averaged bending radius along the ring.
        """
        minimum_number_of_expected_synchrotron_radiation_integrals = 5
        if ring.synchrotron_radiation_integrals is not None:
            self._synchrotron_radiation_integrals = (
                ring.synchrotron_radiation_integrals.copy()
            )
        else:
            if radiation_integrals is None:
                if bending_radius:
                    self._synchrotron_radiation_integrals = calculate_isomagnetic_radiation_integrals(
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
            elif type(radiation_integrals) in {np.ndarray, list}:
                try:
                    integrals = np.array(radiation_integrals)
                except ValueError as ve:
                    raise ValueError(
                        "Could not transform the input into an array"
                    ) from ve
                if (
                    integrals.__len__()
                    >= minimum_number_of_expected_synchrotron_radiation_integrals
                ):
                    self._synchrotron_radiation_integrals = integrals
                else:
                    raise ValueError(
                        "The first five synchrotron radiation integrals are requires "
                    )
            else:
                raise TypeError(
                    f"Expected a list or numpy.ndarray as an input. Received"
                    f" {type(radiation_integrals)}."
                )
            ring._radiation_integrals = self._synchrotron_radiation_integrals

    def _generate_synchrotron_radiation_trackers(
        self, ring: Ring, element_list: list[type[T]]
    ) -> None:
        """
        Function to create and insert the SR trackers in the ring.

        This function inserts SynchrotronRadiationBaseClass elements in the
        ring either:
            - before the drifts if track_before_element_type is None or
            DriftBaseClass. In that case, _SynchrotronRadiationDrift
            trackers will be inserted in the ring before each drift.
            - after the RF cavities if track_before_element_type is
            RFStationBaseClass. In that case, _SynchrotronRadiationSection
            trackers will be inserted in the ring.

        Parameters
        ----------
        ring
            `Ring` context manager.
        element_list
            Element list to consider.
        """
        i = 0
        if all(isinstance(e, DriftBaseClass) for e in element_list):
            for element in element_list:
                i += 1
                if hasattr(element, "radiation_integrals"):
                    share_of_synchrotron_radiation_integrals = (
                        element.radiation_integrals
                    )
                else:
                    share_of_synchrotron_radiation_integrals = (
                        element.orbit_length / ring.circumference
                    ) * self._synchrotron_radiation_integrals
                SRClass_child = _SynchrotronRadiationDrift(
                    section_index=element.section_index,
                    name=f"SynchrotronRadiationTracker_{i}",
                    share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
                    disable_quantum_excitation=self._disable_quantum_excitation,
                )
                # _SynchrotronRadiationDrift tracker placed before the
                # drift
                ring.insert_element(
                    element=SRClass_child,
                    insert_at=ring.elements.elements.index(element),
                    deepcopy=False,  # to maintain the consistency
                    # between the stored array and the ring elements
                )
                self.generated_children.append(SRClass_child)
        elif all(isinstance(e, RFStationBaseClass) for e in element_list):
            cavities_section_indexes = [e.section_index for e in element_list]
            for element in element_list:
                i += 1
                if hasattr(element, "radiation_integrals"):
                    share_of_synchrotron_radiation_integrals = (
                        element.radiation_integrals
                    )
                else:
                    if ring.n_rf_stations == 1:
                        section_length_to_consider = ring.circumference
                    elif (
                        cavities_section_indexes[i - 1]
                        == len(ring.section_lengths) - 1
                    ):
                        section_length_to_consider = ring.section_lengths[-1]
                    else:
                        section_length_to_consider = np.sum(
                            ring.section_lengths[
                                cavities_section_indexes[
                                    i - 1
                                ] : cavities_section_indexes[i]
                            ]
                        )
                    share_of_synchrotron_radiation_integrals = (
                        section_length_to_consider / ring.circumference
                    ) * self._synchrotron_radiation_integrals
                SRClass_child = _SynchrotronRadiationSection(
                    section_index=element.section_index,
                    name=f"SynchrotronRadiationTracker_{i}",
                    share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
                    disable_quantum_excitation=self._disable_quantum_excitation,
                )
                # _SynchrotronRadiationSection tracker placed after the
                # cavity
                ring.insert_element(
                    element=SRClass_child,
                    insert_at=ring.elements.elements.index(element) + 1,
                    deepcopy=False,  # to maintain the consistency
                    # between the stored array and the ring elements
                )
                self.generated_children.append(SRClass_child)
        else:
            raise TypeError(
                "Unsupported list of elements. Full lists of "
                "DriftBaseClass and RFStationBaseClass are "
                f"allowed, but {element_list} was found."
            )

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
            Averaged bending radius along the ring.
        """
        self._set_synchrotron_radiation_integrals(
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
                self._generate_synchrotron_radiation_trackers(
                    ring=ring, element_list=element_list
                )


class _SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
    """
    Class to track the effect on synchrotron radiation before a drift.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_synchrotron_radiation_integrals
        Share of synchrotron radiation integrals.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int = 0,
        share_of_synchrotron_radiation_integrals: NumpyArray | None = None,
        disable_quantum_excitation: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
            disable_quantum_excitation=disable_quantum_excitation,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_drift(self) -> float | None:
        """
        Energy lost by passing through the drift.

        Returns
        -------
        energy_lost_due_to_synchrotron_radiation_drift
            Energy lost due to synchrotron radiation along the drift.
        """
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def synchrotron_radiation_integrals_drift(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals of the drift.

        Returns
        -------
        synchrotron_radiation_integrals_drift
            Synchrotron radiation integrals of the drift.
        """
        return self._share_of_synchrotron_radiation_integrals


class _SynchrotronRadiationSection(SynchrotronRadiationBaseClass):
    """
    Class to track the effect on synchrotron radiation after a RF cavity.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_synchrotron_radiation_integrals
        Share of synchrotron radiation integrals.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int = 0,
        share_of_synchrotron_radiation_integrals: NumpyArray = None,
        disable_quantum_excitation: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
            disable_quantum_excitation=disable_quantum_excitation,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self) -> float | None:
        """
        Energy lost by passing through the section.

        Returns
        -------
        energy_lost_due_to_synchrotron_radiation_section
            Energy lost due to synchrotron radiation along the section.
        """
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def synchrotron_radiation_integrals_section(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals of the section.

        Returns
        -------
        synchrotron_radiation_integrals_section
            Synchrotron radiation integrals of the section.
        """
        return self._share_of_synchrotron_radiation_integrals
