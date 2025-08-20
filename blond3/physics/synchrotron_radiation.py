from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.matlib import empty
from scipy.constants import e, c
from .._core.base import BeamPhysicsRelevant, Schedulable, DynamicParameter
from typing import TYPE_CHECKING

from ..acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import \
    calculate_natural_energy_spread, calculate_damping_times_in_turn, \
    calculate_energy_loss_per_turn, \
    gather_longitudinal_synchrotron_radiation_parameters
from ..cycles.magnetic_cycle import MagneticCycleBase

if TYPE_CHECKING:
    from typing import (
        Optional,
        Optional as LateInit,
    )
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from .. import Ring
    from ..physics.drifts import DriftBaseClass
    from ..physics.cavities import CavityBaseClass


class SynchrotronRadiationMaster(BeamPhysicsRelevant, Schedulable):
    """Master class for handling synchrotron radiation along the ring.
    To be described better #fixme
    """
    def __str__(self):
        is_iso = ""
        if self.is_isomagnetic:
            is_iso = "isomagnetic"
        return (
            f"Synchrotron radiation master class set up for the {is_iso}"
            f" ring {self._simulation.ring.name}. Simulation "
            f"{self._simulation.name} currently set for turn "
            f"{self._turn_i}."
        )

    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._energy_loss_per_turn = None
        self.is_isomagnetic: Optional[bool] = False  # FIXME Optional means  bool | None, but is never none
        self.get_synchrotron_radiation_info_turn_by_turn: Optional[bool] = True
        self.synchrotron_radiation_integrals: LateInit[NumpyArray | CupyArray] = None
        self._simulation: LateInit[Simulation] = None
        self._damping_times: LateInit[NumpyArray | CupyArray] = None # TODO why duplicate `_damping_times_in_seconds`, could this be property?
        self._damping_times_in_seconds: LateInit[NumpyArray | CupyArray] = None
        self._natural_energy_spread: LateInit[NumpyArray | CupyArray] = None

        self._turn_i: LateInit[DynamicParameter] = 0
        self._energy_cycle: LateInit[MagneticCycleBase] = None
        self._ring: LateInit[Ring] = None

        self.generated_children: list[SynchrotronRadiationBaseClass] = [] # TODO add typehint List[SomeClass]

    @cached_property # TODO property enough?
    def energy_loss_per_turn(self) -> NumpyArray:
        return self._energy_loss_per_turn

    @cached_property # TODO property enough?
    def damping_times(self) -> NumpyArray:
        return self._damping_times

    @cached_property # TODO property enough?
    def damping_times_in_seconds(self) -> NumpyArray:
        return self._damping_times_in_seconds

    def generate_children(
        self,
        element_list: Optional[list[DriftBaseClass | CavityBaseClass]
                               | list[int]] = None,
    ):# FIXME SRtracker BEFORE Drifts and AFTER Cavity

        if not empty(self.generated_children):
            raise Warning(
                "Synchrotron radiation subclasses have already been "
                "generated. Command ignored"
            )
        else:
            i = 0

            if element_list is not None:
                if all([isinstance(e, DriftBaseClass | CavityBaseClass) for e in
                        element_list]):
                    for element in element_list:
                        i += 1
                        SRClass_child = SynchrotronRadiationDrift(
                            section_index=element.section_index,
                            name=f"SynchrotronRadiationTracker_{i}",
                        )
                        self._simulation.ring.insert_element(
                            element=SRClass_child,
                            insert_at=
                            self._simulation.ring.elements.elements.index(
                                element),
                            deepcopy=True,
                            )
                        self.generated_children.append(SRClass_child)

                elif all([isinstance(e, int) for e in
                          element_list]):
                    for section_index in element_list:
                        i += 1
                        SRClass_child = SynchrotronRadiationSection(
                            section_index=section_index,
                            name=f"SynchrotronRadiationTracker_{i}",
                        )
                        self._simulation.ring.add_element(
                            SRClass_child, section_index=section_index,
                            reorder=True
                        )
                        self.generated_children.append(SRClass_child)
                else:
                    raise TypeError()

            else:
                element_list = self._simulation.ring.elements.get_elements(
                    DriftBaseClass)
                for element in element_list:
                    i += 1
                    SRClass_child = SynchrotronRadiationSection(
                        section_index=element.section_index,
                        name=f"SynchrotronRadiationTracker_{i}",
                    )
                    self._simulation.ring.add_element(
                        SRClass_child, section_index=element.section_index,
                        reorder=True
                    )
                    self.generated_children.append(SRClass_child)

        return print(f"{len(self.generated_children)} synchrotron radiation "
                     f"trackers generated")

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals_ring()

        self._turn_i = simulation.turn_i
        self._energy_cycle = simulation.energy_cycle
        self._ring = simulation.ring

        self.__str__() # TODO WHY

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        if self.get_synchrotron_radiation_info_turn_by_turn:
            self._energy_loss_per_turn = np.empty(n_turns)
            self._longitudinal_damping_time = np.empty(n_turns)
            self._natural_energy_spread = np.empty(n_turns)
        pass

    def track(self, beam: BeamBaseClass) -> None:
        self.update_synchrotron_radiation_integrals()
        # Updates the SRI to be implemented in all children classes

        # Get the turn-by-turn data if requested, from the synchrotron
        # radiation integrals
        if self.get_synchrotron_radiation_info_turn_by_turn:
            self._energy_loss_per_turn[self._turn_i] = calculate_energy_loss_per_turn(
                energy=beam.reference_total_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=beam.particle_type,
            )
            self._damping_times[self._turn_i, :] = calculate_damping_times_in_turn(
                energy=beam.reference_total_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                energy_loss_per_turn=self._energy_loss_per_turn[self._turn_i],
            )
            self._natural_energy_spread[self._turn_i] = calculate_natural_energy_spread(
                energy=beam.reference_total_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=beam.particle_type,
            )
        else:
            pass

    def init_synchrotron_radiation_integrals_ring(self):
        pass

    def update_synchrotron_radiation_integrals(self):
        """
        Function to update the synchrotron radiation integrals of the
        generated children, if decided to store everything in the master for faster tracking.
        :return:
        """
        pass


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
    """Base class to handle the synchrotron radiation energy loss and damping,
    and quantum excitation effect along a section of the ring.
    """

    def __str__(self):
        return f"Sychrotron radiation section element."

    def __init__(
        self,
        name: Optional[str] = None,
        section_index: Optional[int] = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._simulation: LateInit[Simulation] = None
        self._energy_lost_due_to_synchrotron_radiation = None
        self._synchrotron_radiation_integrals: LateInit[NumpyArray | CupyArray] = None
        self._damping_partition_number: float = None
        self._damping_time: NumpyArray = None
        self._natural_energy_spread: NumpyArray = None
        self._turn_i: LateInit[DynamicParameter] = 0

    def _calculate_kick(self, beam: BeamBaseClass):
        """
        Function to calculate the energy kick induced by the energy lost by
        synchrotron radiation, its damping effect and the quantum excitation.
        Function used to update the beam partial energy dE.
        :param beam: BeamBaseClass object.
        :return:
        """
        U0, tau_z, sigma0 = gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=beam.particle_type, energy=beam.reference_total_energy
        )
        self._natural_energy_spread[self._turn_i] = np.average(sigma0)
        self._energy_lost_due_to_synchrotron_radiation[self._turn_i] = np.average(U0)
        self._damping_time[self._turn_i] = np.average(tau_z)

        return (
            -
            - 2.0 / tau_z * beam.read_partial_dE()
            - 2.0
            * sigma0
            / np.sqrt(tau_z)
            * beam.reference_total_energy
            * np.random.normal(size=len(beam.n_macroparticles_partial()))
        )

    def _update_beam_energy(self, beam: BeamBaseClass):
        """
        Function to update the beam particles partial energy after passing
        through the _SynchrotronRadiationBaseClass element in the ring.
        :param beam: BeamBaseClass object
        """
        # TODO write C++ routine
        energy_change = self._calculate_kick(beam=beam)
        dE = beam.write_partial_dE()
        dE[:] += energy_change

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

        self._turn_i = simulation.turn_i
        # generate the synchrotron radiation integrals

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def track(self) -> None:
        self._turn_i = self._simulation.turn_i
        for beam in self._simulation.beams:
            self._update_beam_energy(beam)
        pass


class SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_drift(self):
        """Energy lost by passing through the drift"""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def synchrotron_radiation_integrals_drift(self):
        """Synchrotron radiation integrals of the drift"""
        return self._synchrotron_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass
        self._turn_i = simulation.turn_i
        self._synchrotron_radiation_integrals = (self._ *
                                                 self._synchrotron_radiation_integrals)

class SynchrotronRadiationSection(SynchrotronRadiationBaseClass):
    #TODO : enforce a constraint on the number of
    # SynchrotronRadiationSection per section
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self):
        """Energy lost by passing through the section"""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def synchrotron_radiation_integrals_section(self):
        """Synchrotron radiation integrals of the section"""
        return self._synchrotron_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass
        self._turn_i = simulation.turn_i
        lengths_sections = self._simulation.ring.section_lengths()
        share_synchrotron_radiation_integrals = (
            lengths_sections[self.section_index] / self._simulation.ring.circumference
        )
        self._synchrotron_radiation_integrals = (
            share_synchrotron_radiation_integrals) * self._synchrotron_radiation_integrals

class WigglerMagnet(SynchrotronRadiationBaseClass):
    """
    Synchrotron Radiation subclass to simulate the effect of one or a
    series of identical damping wigglers on the simulated beams.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        section_index: Optional[int] = None,
        wiggler_type: Optional[str] = "sinusoidal",
        number: Optional[int] = 1,
        peak_field: Optional[float] = 1.0,
        pole_length: Optional[float] = 0.095,
        number_poles: Optional[int] = 43,
    ):
        super().__init__(name=name, section_index=section_index)

        self._type = (wiggler_type,)
        self._number = (number,)
        self._peak_field = (peak_field,)
        self._pole_length = (pole_length,)
        self._number_poles = (number_poles,)

        self._simulation: LateInit[Simulation] = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: LateInit[
            NumpyArray | CupyArray
        ] = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: LateInit[
            NumpyArray | CupyArray
        ] = np.zeros((1, 5))

    @property
    def number_of_wigglers(self):
        return self._number

    @property
    def length_wiggler(self):
        if self.type == "wiggler_type":
            return self.pole_length * self.number_poles

    @property
    def number_of_poles(self):
        return self._number_poles

    @property
    def peak_magnetic_field(self):
        return self._peak_field

    @property
    def pole_length(self):
        return self._pole_length

    def __str__(self):
        return (
            f"{self.number_of_wigglers} damping wigglers of {self.peak_magnetic_field} T "
            f"and composed of {self.number_of_poles} poles of {
                self.pole_length
            } m each have been added to "
            f"the "
            f"simulation. \n"
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        self.calculate_contribution_to_synchrotron_radiation_integrals(self)
        pass

    def calculate_contribution_to_synchrotron_radiation_integrals(self):
        """
        Function to initialize the energy-free fraction of the damping
        wiggler radiation integrals.
        :return:
        """
        self._contribution_to_synchrotron_radiation_integrals_without_energy = np.array(
            [
                (
                    -self.number
                    * self.length_wiggler
                    * (e * self.peak_field) ** 2
                    * self.length_wiggler
                    / (2 * np.pi)
                ),
                1 / 2 * self.number * self.length_wiggler * (e * self.peak_field) ** 2,
                4
                / (3 * np.pi)
                * self.number
                * self.length_wiggler
                * (e * self.peak_field) ** 3,
                0,
                self.number
                * self.pole_length**2
                * self.length_wiggler
                / (15 * np.pi**3)
                * (e * self.peak_field) ** 5,
            ]
        )

    def update_synchrotron_radiation_integrals(self, beam: BeamBaseClass):
        """
        Function to update the synchrotron radiation integrals change from
        the damping wiggler
        :param beam:
        :return:
        """
        E = beam.read_partial_dE() + beam.reference_total_energy
        var = 1 / (E * e / c)
        energy_contribution_wiggler_integrals = np.array(
            [
                var ** 2,
                var ** 2,
                var ** 3,
                var ** 3,
                var ** 5,
            ]
        )
        self._contribution_to_synchrotron_radiation_integrals_with_energy = np.multiply(
            self._contribution_to_synchrotron_radiation_integrals_without_energy,
            energy_contribution_wiggler_integrals,
        )
        pass

    @abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        for beam in self._simulation.beams:
            self.update_synchrotron_radiation_integrals(beam=beam)
            self._update_beam_energy(beam)
