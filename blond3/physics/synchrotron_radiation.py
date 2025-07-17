from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from scipy.constants import e, c, m_e

from blond3 import Simulation
from .._core.base import BeamPhysicsRelevant, Schedulable, DynamicParameter
from .._core.beam.base import BeamBaseClass
from typing import TYPE_CHECKING

from ..cycles.energy_cycle import EnergyCycleBase

if TYPE_CHECKING:
    from typing import Optional, Optional as LateInit
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from .. import Ring

class SynchrotronRadiationMaster(BeamPhysicsRelevant, Schedulable):
    """Master class for handling synchrotron radiation along the ring.
    To be described better #fixme
    """
    def __str__(self):
        is_iso = ""
        if self.is_isomagnetic:
            is_iso = "isomagnetic"
        return (f"Synchrotron radiation master class set up for the " +
                is_iso + f" ring {self._simulation.ring.name}. Simulation "
                         f"{self._simulation.name} currently set for turn "
                         f"{self._turn_i}.")

    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: Optional[bool] = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._energy_loss_per_turn = None
        self.is_isomagnetic: Optional[bool] = False
        self.get_synchrotron_radiation_info_turn_by_turn: Optional[bool] = True
        self.synchrotron_radiation_integrals: LateInit[NumpyArray | CupyArray] = None
        self._simulation: LateInit[Simulation] = None
        self._damping_times: LateInit[NumpyArray | CupyArray] = None
        self._damping_times_in_seconds: LateInit[NumpyArray | CupyArray] = None
        self._natural_energy_spread: LateInit[NumpyArray | CupyArray] = None

        self._turn_i: LateInit[DynamicParameter] = 0
        self._energy_cycle: LateInit[EnergyCycleBase] = None
        self._ring: LateInit[Ring] = None

        self.generated_children: bool = False

        self.__str__()
    @cached_property
    def energy_loss_per_turn(self) -> NumpyArray:
        return self._energy_loss_per_turn
    @cached_property
    def damping_times(self) -> NumpyArray:
        return self._damping_times
    @cached_property
    def damping_times_in_seconds(self)-> NumpyArray:
        return self._damping_times_in_seconds

    def generate_children(
        self, section_list = None, element_list = None, location:Optional[str] = "after"
    ):
        """
        Function to generate and insert synchrotron radiation elements in
        the ring. By default, the elements are inserted at the end of each section.
        A section list can be provided to limit the study to the preferred
        section.
        An element list can be provided along with a location preference to
        insert the synchrotron radiation elements at the requested
        location.#fixme

        :param section_list:
        :param element_list:
        :param location:
        :return:
        """
        if self.generated_children:
            raise Warning(
                "Synchrotron radiation subclasses have already been "
                "generated. Command ignored"
            )
        else:
            from drifts import DriftBaseClass
            drifts_list = self._simulation.ring.elements.get_elements(
                DriftBaseClass)
            number_of_sections = drifts_list[-1].section_index

            s_list = self._simulation.ring.section_lengths #
                # Access section information required
            self.generated_children = True


        return
        # for element in elements_list:

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals_ring()

        self._turn_i = simulation.turn_i
        self._energy_cycle = simulation.energy_cycle
        self._ring = simulation.ring
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

    def track(self) -> None:
        self.update_synchrotron_radiation_integrals()
        # Updates the SRI to be implemented in all children classes

        # Get the turn-by-turn data if requested, from the synchrotron
        # radiation integrals
        if self.get_synchrotron_radiation_info_turn_by_turn:
            self.calculate_energy_loss_per_turn()
            self.calculate_damping_times()
            self.calculate_natural_energy_spread()
        else:
            pass

    def init_synchrotron_radiation_integrals_ring(self):
        pass





    def calculate_damping_times(self):
        damping_partition_numbers = self.calculate_partition_numbers()
        tau_x = [
            (2 * beam.reference_total_energy / damping_partition_numbers[0])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        tau_y = [
            (2 * beam.reference_total_energy / damping_partition_numbers[1])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        tau_z = [
            (2 * beam.reference_total_energy / damping_partition_numbers[2])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        return np.array([tau_x, tau_y, tau_z])

    def calculate_natural_energy_spread(self):
        pass

    def update_synchrotron_radiation_integrals(self):
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

        self._energy_lost_due_to_synchrotron_radiation_particles = None
        self._synchrotron_radiation_integrals_section: LateInit[
            NumpyArray | CupyArray
        ] = None
        self._damping_partition_number_section: float = None
        self._damping_time_section: float = None
        self._natural_energy_spread_section: float = None

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self):
        """Energy lost by passing through the section"""
        return np.average(self._energy_lost_due_to_synchrotron_radiation_particles)
    @property
    def synchrotron_radiation_integrals_section(self):
        return self._synchrotron_radiation_integrals_section

    def update_beam_energy(self, beam: BeamBaseClass):
        """
        Function to update the beam particles energy after the passage
        through the damping wiggler(s).
        :param beam:
        :return:
        """
        # Ideally: the energy losses should be computed per particle,
        # and not per beam, to closely match the effect of SR onto each
        # particle. To be checked with theory
        DI_wE = self._synchrotron_radiation_integrals_section
        particles_total_energy = beam.reference_total_energy + beam.read_partial_dE()
        energy_lost_from_synchrotron_radiation = (
            beam.particle_type.sands_radiation_constant
            * DI_wE[1]
            / (2.0 * np.pi)
            * particles_total_energy**4.0
        )
        longitudinal_damping_partition_number = 2 + DI_wE[3] / DI_wE[1]
        longitudinal_damping_time = (
            2.0
            / longitudinal_damping_partition_number
            * beam.read_partial_dE()
            / energy_lost_from_synchrotron_radiation
        )  #
        # longitudinal damping
        # time in turns #

        # sigma_dE0 = np.sqrt(beam.particle_type.c_q() * (beam._dE/m_e)**2.0 *
        #                     DI_wE[2] / (jz * DI_wE[1]))
        # # check validity
        # beam._dE += - U0_wiggler # energy lost through the damping wiggler
        #             - 2.0 / tau_z  * beam._dE
        #             - 2.0 * self.sigma_dE /
        #                       np.sqrt(self.tau_z * self.n_kicks)
        #                       * self.beam.energy * np.random.normal
        #                       (size=self.beam.n_macroparticles))

    @abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        pass

class SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: Optional[bool] = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        pass

    def calculate_synchrotron_radiation_parameters(self, beam: BeamBaseClass):
        _DI_wE = self.synchrotron_radiation_integrals_section
        energy_lost_section = (
            beam.particle_type.sands_radiation_constant
            * _DI_wE[1]
            / (2.0 * np.pi)
            * beam.read_partial_dE() ** 4.0
        )
        return energy_lost_section

    @abstractmethod
    def track(self,  beam: BeamBaseClass) -> None:
        pass

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

        self.type = (wiggler_type,)
        self.number = (number,)
        self.peak_field = (peak_field,)
        self.pole_length = (pole_length,)
        self.number_poles = (number_poles,)

        self._simulation: LateInit[Simulation] = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: LateInit[
            NumpyArray | CupyArray
        ] = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: LateInit[
            NumpyArray | CupyArray
        ] = np.zeros((1, 5))

    @property
    def length_wiggler(self):
        if self.type == "wiggler_type":
            return self.pole_length * self.number_poles

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
        energy_contribution_wiggler_integrals = np.array(
            [
                1 / (E * e / c) ** 2,
                1 / (E * e / c) ** 2,
                1 / (E * e / c) ** 3,
                1 / (E * e / c) ** 3,
                1 / (E * e / c) ** 5,
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
            self.update_beam_energy(beam)
