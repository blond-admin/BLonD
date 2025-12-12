# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""Collection to include synchrotron radiation and quantum excitation effects.

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
Further information on synchrotron radiation damping and quantum excitation
can be found in:
- H. Wiedemann, Synchrotron Radiation, Springer, 2003
- S.Y. Lee, Accelerator Physics, World Scientific, Third edition,
2014 #check date
- A. Wolski, Introduction to Beam Dynamics in High-Energy Electron Storage
Rings, Morgan & Claypool Publishers, 2018

Author:
L. Valle
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable
from blond.cycles.magnetic_cycle import MagneticCycleBase
from blond.physics.cavities import RfStationBaseClass
from blond.physics.drifts import DriftBaseClass
from blond.physics.synchrotron_radiation.synchrotron_radiation_elements import (
    SynchrotronRadiationBaseClass,
)

if TYPE_CHECKING:
    from typing import (
        TypeVar,
    )

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.ring.ring import Ring
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import RfStationBaseClass
    from blond.physics.drifts import DriftBaseClass

    T = TypeVar("T")

# TODO allow schedulable synchrotron radiation integrals in the master (e.g.
# tapering)


class SynchrotronRadiationMaster(BeamPhysicsRelevant, Schedulable):
    """
    Master class for handling synchrotron radiation along the ring.

    This element is to be added in the ring object prior to the simulation.
    On initialisation, it inserts subclasses along the ring after the
    specified elements (either drifts or section.)
    To be described better #fixme

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name: str, optional
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    radiation_integrals: NumpyArray, None
        Synchrotron radiation integrals. If None, the ring will be
        considered isomagnetic.
        In the case of an isomagnetic ring, the synchrotron radiation
        integrals will be computed from the ring bending radius. Default:
        False.
    """

    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        radiation_integrals: NumpyArray | None = None,
        track_before_element_type: type[T] | None = None,
        get_synchrotron_radiation_info_turn_by_turn: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )

        minimum_number_of_expected_synchrotron_radiation_integrals = 5
        if radiation_integrals is None:
            self.is_isomagnetic = True
        else:
            self.is_isomagnetic = False
            if type(radiation_integrals) in {np.ndarray, list}:
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
                    self.synchrotron_radiation_integrals = integrals
                else:
                    raise ValueError(
                        "The first five synchrotron "
                        + "radiation integrals are requires "
                        + "Ignoring input."
                    )
            else:
                raise TypeError(
                    f"Expected a list or numpy.ndarray as an input. Received"
                    f" {type(radiation_integrals)}."
                )
        self.verbose = verbose

        if track_before_element_type is not None:
            self.track_before_element_type = track_before_element_type
        else:
            self.track_before_element_type = DriftBaseClass

        self._simulation: Simulation | None = None
        self._turn_i: DynamicParameter | None = 0
        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

        self.generated_children: list[SynchrotronRadiationBaseClass] = []

    def __str__(self):
        """Method to print general information about the created class."""
        is_iso = ""
        if self.is_isomagnetic:
            is_iso = "isomagnetic"
        return (
            f"Synchrotron radiation master class set up for the {is_iso}"
            f" ring. Simulation currently set for turn "
            f"{self._turn_i}. \n Generated "
            f"{self.number_of_generated_synchrotron_radiation_classes} "
            f"synchrotron radiation elements."
        )

    @cached_property  # TODO property enough?
    def energy_loss_per_turn(self) -> NumpyArray:
        """Energy loss per turn, eV per turn."""
        return self._energy_loss_per_turn

    @cached_property  # TODO property enough?
    def damping_times(self) -> NumpyArray:
        """Damping times, in turns."""
        return self._damping_times

    @property
    def number_of_generated_synchrotron_radiation_classes(self) -> int:
        """Number of generated synchrotron radiation classes."""
        return len(self.generated_children)

    # TODO : Add a function to calculate the length of the sections/ drifts
    # before the children and store it for later SR integrals update.
    # Question: How to handle modification from wigglers and other classes.

    # TODO : Update synchrotron radiation integrals after a wiggler? Detect
    #  other SynchrotronRadiationBaseClass elements before generating the
    #  children and save their location for SRI update.

    # TODO: transmit the share of SRI to the children.
    def generate_synchrotron_radiation_subclasses(
        self,
        simulation: Simulation,
    ):
        """Function to create synchrotron radiation elements in the ring.

        This method automatically creates, inserts and initialises the
        synchrotron radiation elements in the ring.

        Parameters
        ----------
        simulation
            `Simulation` context manager
        """
        if self.generated_children:
            raise Warning(
                "Synchrotron radiation subclasses have already been "
                "generated. Command ignored"
            )
        else:
            i = 0
            element_list = simulation.ring.elements.get_elements(
                class_=self.track_before_element_type
            )
            if (element_list is None) or all(
                isinstance(e, DriftBaseClass | RfStationBaseClass)
                for e in element_list
            ):
                for element in element_list:
                    i += 1
                    if hasattr(element, "radiation_integrals"):
                        share_of_synchrotron_radiation_integrals = (
                            element.radiation_integrals
                        )
                    else:
                        share_of_synchrotron_radiation_integrals = (
                            element.orbit_length
                            / simulation.ring.circumference
                        )
                    SRClass_child = _SynchrotronRadiationDrift(
                        section_index=element.section_index,
                        name=f"SynchrotronRadiationTracker_{i}",
                        share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
                    )
                    simulation.ring.insert_element(
                        element=SRClass_child,
                        insert_at=simulation.ring.elements.elements.index(
                            element
                        ),
                        deepcopy=True,
                    )
                    self.generated_children.append(SRClass_child)

            elif all(isinstance(e, int) for e in element_list):
                for section_index in element_list:
                    i += 1
                    if hasattr(simulation.ring, "section_radiation_integrals"):
                        share_of_synchrotron_radiation_integrals = (
                            simulation.ring.section_radiation_integrals[
                                section_index
                            ]
                        )
                    else:
                        share_of_synchrotron_radiation_integrals = (
                            simulation.ring.section_lengths[section_index]
                            / simulation.ring.circumference
                        )

                    SRClass_child = _SynchrotronRadiationSection(
                        section_index=section_index,
                        name=f"SynchrotronRadiationTracker_{i}",
                        share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
                    )
                    simulation.ring.add_element(
                        SRClass_child,
                        section_index=section_index,
                        reorder=True,
                    )
                    self.generated_children.append(SRClass_child)
            else:
                raise TypeError("Inhomogeneous element classes.")

        # FIXME SR tracker BEFORE Drifts and AFTER Cavity -- do I agree now?
        return print(
            f"{len(self.generated_children)} synchrotron radiation "
            f"trackers generated"
        )

    def on_init_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        self._turn_i = simulation.turn_i
        self._magnetic_cycle = simulation.magnetic_cycle
        self._ring = simulation.ring

        if self.verbose:
            self.__str__()  # TODO WHY

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass
        # TODO create observable
        # if self.get_synchrotron_radiation_info_turn_by_turn:
        #     self._energy_loss_per_turn[self._turn_i] = (
        #         calculate_energy_loss_per_turn(
        #             energy=beam.reference_total_energy,
        #             synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        #             particle_type=beam.particle_type,
        #         )
        #     )
        #     self._damping_times[self._turn_i, :] = (
        #         calculate_damping_times_in_turns(
        #             energy=beam.reference_total_energy,
        #             synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        #             particle_type=beam.particle_type,
        #         )
        #     )
        #     self._natural_energy_spread[self._turn_i] = (
        #         calculate_natural_energy_spread(
        #             energy=beam.reference_total_energy,
        #             synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        #             particle_type=beam.particle_type,
        #         )
        #     )
        # else:


class _SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
    """
    Class to track the effect on synchrotron radiation before a drift.

    Parameters
    ----------
    name: str, optional
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index: int
        Section index to group elements into sections
    share_of_synchrotron_radiation_integrals: NumpyArray
        Fractional synchrotron radiation integrals.
    """

    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        share_of_synchrotron_radiation_integrals: NumpyArray = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_drift(self):
        """Energy lost by passing through the drift."""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def share_of_synchrotron_radiation_integrals(self):
        """Synchrotron radiation integrals of the drift."""
        return self._fractional_radiation_integrals

    @property
    def synchrotron_radiation_integrals_drift(self):
        """Synchrotron radiation integrals of the drift."""
        return self._fractional_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        self._turn_i = simulation.turn_i


class _SynchrotronRadiationSection(SynchrotronRadiationBaseClass):
    """
    Class to track the effect on synchrotron radiation before a section.

    Parameters
    ----------
    name: str, optional
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index: int
        Section index to group elements into sections
    share_of_synchrotron_radiation_integrals: NumpyArray
        Fractional synchrotron radiation integrals.
    """

    # TODO : enforce a constraint on the number of
    #  SynchrotronRadiationSection per section
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        share_of_synchrotron_radiation_integrals: NumpyArray = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            share_of_synchrotron_radiation_integrals=share_of_synchrotron_radiation_integrals,
        )
        self._energy_lost_due_to_synchrotron_radiation = None

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self):
        """Energy lost by passing through the section."""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def share_of_synchrotron_radiation_integrals(self):
        """Synchrotron radiation integrals of the section."""
        return self._fractional_radiation_integrals

    @property
    def synchrotron_radiation_integrals_section(self):
        """Synchrotron radiation integrals of the section."""
        return self._fractional_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        self._turn_i = simulation.turn_i
