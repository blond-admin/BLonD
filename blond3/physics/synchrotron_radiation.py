from __future__ import annotations

from abc import ABC
from functools import cached_property

import numpy as np
from scipy.constants import e, c, m_e

from blond3 import Simulation
from blond3._core.base import BeamPhysicsRelevant, Schedulable, DynamicParameter
from blond3._core.beam.base import BeamBaseClass
from typing import TYPE_CHECKING

from cycles.energy_cycle import EnergyCycleBase

if TYPE_CHECKING:
    from typing import Optional, Optional as LateInit
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from .. import Ring


class SynchrotronRadiation(BeamPhysicsRelevant):
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
        self.is_isomagnetic = is_isomagnetic
        self.synchrotron_radiation_integrals_section = (
            self.init_synchrotron_radiation_integrals_section()
        )

        self._simulation: LateInit[Simulation] = None
        self.longitudinal_damping_time: LateInit[NumpyArray | CupyArray] = None
        self.natural_energy_spread: LateInit[NumpyArray | CupyArray] = None

    def init_synchrotron_radiation_integrals_section(self):
        """
        Function to handle the synchrotron radiation integrals from an
        input array or a bending radius input.
        For more about synchrotron radiation damping and integral
        definition, please refer to (non-exhaustive list):
        A. Wolski, CAS Advanced Accelerator Physics, 19-29 August 2013
        H. Wiedemann, Particle Accelerator Physics, Chapter Equilibrium
        Particle Distribution, p. 384, Third Edition, Springer, 2007
        """
        if self.is_isomagnetic:
            bending_radius = self._simulation.ring.bending_radius
            I1 = self._simulation.ring.circumference * self._simulation
            I2 = 2.0 * np.pi / bending_radius
            I3 = 2.0 * np.pi / bending_radius**2.0
            I4 = (
                self._simulation.ring.circumference
                * self._simulation.ring.alpha_0[0, 0]
                / bending_radius**2.0
            )
            I5 = I1 * I2  # disregarding the effect of quadrupoles
            self.jz = 2.0 + self.I4 / self.I2
            radiation_integrals = np.array([I1, I2, I3, I4, I5])

        else:
            radiation_integrals = None  # fixme
        return radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals_section()

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
        DI_wE = self.synchrotron_radiation_integrals_section
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

    def track(self) -> None:
        pass


class SynchrotronRadiationMaster(BeamPhysicsRelevant, Schedulable):
    """Master class for handling synchrotron radiation along the ring.
    To be described better #fixme
    """

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
        self._natural_energy_spread: LateInit[NumpyArray | CupyArray] = None

        self._turn_i: LateInit[DynamicParameter] = None
        self._energy_cycle: LateInit[EnergyCycleBase] = None
        self._ring: LateInit[Ring] = None

    @cached_property
    def energy_loss_per_turn(self) -> NumpyArray:
        return self._energy_loss_per_turn

    @cached_property
    def damping_times(self) -> NumpyArray:
        return self._damping_times

    @cached_property
    def damping_times_in_seconds(self):
        tau_x_s = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[0])
            / self.energy_loss_per_turn
            / beam.revolution_frequency
            for beam in self._simulation.beams
        ]
        tau_y_s = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[1])
            / self.energy_loss_per_turn
            / beam.revolution_frequency
            for beam in self._simulation.beams
        ]
        tau_z_s = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[2])
            / self.energy_loss_per_turn
            / beam.revolution_frequency
            for beam in self._simulation.beams
        ]
        return self.damping_times / self._simulation.beam.revolution_frequency

    def generate_children(
        self, element_type="drift", location: Optional[str] = "after"
    ):
        """Function to generate and insert synchrotron radiation trackers
        along the ring, either before or after the element type specified."""
        if element_type == "drift":
            from drifts import DriftBaseClass

            elements_list = self._simulation.ring.get_elements(DriftBaseClass)
        elif element_type == "cavity":
            from cavities import CavityBaseClass

            elements_list = self._simulation.ring.get_elements(CavityBaseClass)

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
            self.calculate_longitudinal_damping_time()
            self.calculate_natural_energy_spread()
        else:
            pass

    def init_synchrotron_radiation_integrals_ring(self):
        pass

    def calculate_energy_loss_per_turn(self):
        """
        Function to calculate the energy lost due to synchrotron radiation
        for all beams, within one turn
        """
        energy_loss_per_turn = [
            (
                beam.particle_type.sands_radiation_constant
                * beam.reference_total_energy
                * 4
                * self.synchrotron_radiation_integrals[1]
                / self._simulation.ring.circumference
            )
            for beam in self._simulation.beams
        ]
        return energy_loss_per_turn

    def calculate_partition_numbers(self):
        """Damping partition numbers"""
        jx = (
            1
            - self.synchrotron_radiation_integrals[3]
            / self.synchrotron_radiation_integrals[1]
        )
        jy = 1
        jz = (
            2
            + self.synchrotron_radiation_integrals[3]
            / self.synchrotron_radiation_integrals[1]
        )
        return np.array([jx, jy, jz])

    def calculate_damping_times(self):
        tau_x = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[0])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        tau_y = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[1])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        tau_z = [
            (2 * beam.reference_total_energy / self.damping_partition_numbers[2])
            / self.energy_loss_per_turn
            for beam in self._simulation.beams
        ]
        return np.array([tau_x, tau_y, tau_z])

    def calculate_natural_energy_spread(self):
        pass

    def update_synchrotron_radiation_integrals(self):
        pass


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        section_index: Optional[int] = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._energy_lost_due_to_synchrotron_radiation_particles = None
        self.synchrotron_radiation_integrals_section: LateInit[
            NumpyArray | CupyArray
        ] = None
        self._damping_partition_number_section: float = None
        self._damping_time_section: float = None
        self.natural_energy_spread_section: float = None

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self):
        """Energy lost by passing through the section"""
        return np.average(self._energy_lost_due_to_synchrotron_radiation_particles)

    def update_beam_energy(self, beam: BeamBaseClass):
        """
        Function to update the beam particles energy to include synchrotron
        energy loss, radiation damping and quantum excitation.
        The energy lost due to synchrotron radiation is calculated per
        particle.
        :param beam:
        :return:
        """

        # Ideally: the energy losses should be computed per particle,
        # and not per beam, to closely match the effect of SR onto each
        # particle. To be checked with theory #fixme

        _DI_wE = self.synchrotron_radiation_integrals_section
        Cgamma = beam.particle_type.sands_radiation_constant
        Cq = beam.particle_type.quantum_radiation_constant
        beam_energy = beam.reference_total_energy
        particles_total_energy = beam_energy + beam.read_partial_dE()
        self._energy_lost_due_to_synchrotron_radiation_particles = (
            Cgamma * _DI_wE[1] / (2.0 * np.pi) * particles_total_energy**4.0
        )

        U0_particles = self._energy_lost_due_to_synchrotron_radiation_particles
        self.damping_partition_number_section = (
            2 + _DI_wE[3] / _DI_wE[1]
        )  # longitudinal damping number

        self.natural_energy_spread_section = np.sqrt(
            Cq
            * (beam.reference_total_energy / m_e) ** 2.0
            * _DI_wE[2]
            / (self.damping_partition_number_section * self._DI_wE[1])
        )

        # check validity
        beam.write_partial_dE += (
            -U0_particles
            - self.damping_partition_number_section * U0_particles
            + 2.0
            * self.natural_energy_spread_section
            / np.sqrt(self.damping_time_section)
            * np.average(particles_total_energy)
            * np.random.normal(size=beam.n_macroparticles)
        )


class WigglerMagnet(SynchrotronRadiationBaseClass):
    """Damping wiggler magnet class"""

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

    def track(self, beam: BeamBaseClass) -> None:
        for beam in self._simulation.beams:
            self.update_synchrotron_radiation_integrals(beam=beam)
            self.update_beam_energy(beam)
