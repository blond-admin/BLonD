from functools import cached_property

import numpy as np
from scipy.constants import e, c, m_e

from blond3 import Simulation
from blond3._core.base import BeamPhysicsRelevant
from blond3._core.beam.base import BeamBaseClass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Optional as LateInit
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass


class SynchrotronRadiation(BeamPhysicsRelevant, section_i=None):
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: Optional[bool] = False,
    ):
        super().__init__(
            section_index=section_index, name=name,
        )
        self.is_isomagnetic = is_isomagnetic
        self.synchrotron_radiation_integrals = self.init_synchrotron_radiation_integrals()
        self._simulation: LateInit[Simulation] = None
        self.longitudinal_damping_time : LateInit[NumpyArray | CupyArray] = None
        self.natural_energy_spread : LateInit[NumpyArray | CupyArray] = None

    def init_synchrotron_radiation_integrals(self):
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
            self.rho = bending_radius
            I1 = (self._simulation.ring._circumference *
                  self._simulation)
            I2 = 2.0 * np.pi / self.rho
            I3 = 2.0 * np.pi / self.rho ** 2.0
            I4= (self.ring.ring_circumference *
                       self.ring.alpha_0[0, 0] / self.rho ** 2.0)
            I5 = I1 * I2 # disregarding the effect of quadrupoles
            self.jz = 2.0 + self.I4 / self.I2

        return np.array([I1, I2, I3, I4, I5])


    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals()


    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        if simulation.get_sr_info_tbt:
            self.energy_loss_per_turn = np.empty(n_turns)
            self.longitudinal_damping_time = np.empty(n_turns)
            self.natural_energy_spread = np.empty(n_turns)
        pass





    def calculate_synchrotron_radiation_parameters(self, beam: BeamBaseClass):
        _DI_wE = self._contribution_to_synchrotron_radiation_integrals_without_energy

        energy_loss_per_turn = (beam.particle_type.sands_radiation_constant) * _DI_wE[
             1]/ (2.0 * np.pi) * beam._dE ** 4.0  # in eV per turn


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
        DI_wE = self._contribution_to_synchrotron_radiation_integrals_without_energy
        particles_total_energy = beam. reference_total_energy + beam.read_partial_dE()
        energy_lost_from_synchrotron_radiation = beam.particle_type.sands_radiation_constant * DI_wE[1]/ (2.0 *np.pi) * particles_total_energy** 4.0
        longitudinal_damping_partition_number = 2 + DI_wE[3]/DI_wE[1] #
        # longitudinal damping number
        longitudinal_damping_time = (2.0 / longitudinal_damping_partition_number *
                                     beam.read_partial_dE()/energy_lost_from_synchrotron_radiation) #
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

    # def track(self, beam: BeamBaseClass) -> None:
    #     pass
    @energy_loss_per_turn.setter
    def energy_loss_per_turn(self, value):
        self._energy_loss_per_turn = value


class SynchrotronRadiationMaster(BeamPhysicsRelevant):
    """ Master class for handling synchrotron radiation along the ring.
    To be described better #fixme
    """
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
        is_isomagnetic: Optional[bool] = False,
    ):
        super().__init__(
            section_index=section_index, name=name,
        )
        self.is_isomagnetic = is_isomagnetic
        self.synchrotron_radiation_integrals = self.init_synchrotron_radiation_integrals()
        self._simulation: LateInit[Simulation] = None
        self.longitudinal_damping_time : LateInit[NumpyArray | CupyArray] = None
        self.natural_energy_spread : LateInit[NumpyArray | CupyArray] = None

    @cached_property
    def energy_loss_per_turn(self, beam: BeamBaseClass) -> float:
        """Beam reference fraction of speed of light (v/c0) []"""
        energy_loss_per_turn = (
                beam.particle_type.sands_radiation_constant *
                beam.reference_total_energy * 4
                * self.synchrotron_radiation_integrals[1] /
                self._simulation.ring.circumference)
        return energy_loss_per_turn

    @cached_property
    def damping_times(self, beam: BeamBaseClass) -> float:
        """Beam reference fraction of speed of light (v/c0) []"""
        energy_loss_per_turn = (
                beam.particle_type.sands_radiation_constant *
                beam.reference_total_energy * 4
                * self.synchrotron_radiation_integrals[1] /
                self._simulation.ring.circumference)
        return energy_loss_per_turn

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals_ring()

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        if simulation.get_sr_info_tbt:
            self.energy_loss_per_turn = np.empty(n_turns)
            self.longitudinal_damping_time = np.empty(n_turns)
            self.natural_energy_spread = np.empty(n_turns)
        pass

    def generate_children(self, element_type = 'drift', location: Optional[str] ='after'):
        """ Function to generate and insert synchrotron radiation trackers
        along the ring, either before or after the element type specified."""
        if element_type == 'drift':
            from drifts import DriftBaseClass
            elements_list = self._simulation.ring.get_elements(DriftBaseClass)
        elif element_type == 'cavity':
            from cavities import CavityBaseClass
            elements_list = self._simulation.ring.get_elements(CavityBaseClass)

        # for element in elements_list:

class SynchrotronRadiationBaseClass(BeamPhysicsRelevant):
    def __init__(
        self,
        name: Optional[str] = None,
        section_index: Optional[int] = None,
    ):
        super().__init__(name = name,
                         section_index=section_index
        )

        self.synchrotron_radiation_integrals_section: LateInit[NumpyArray | CupyArray] = None
        self.energy_lost_due_to_synchrotron_radiation_section: float = None
        self.damping_partition_number_section: float = None
        self.damping_time_section: float = None
        self.natural_energy_spread_section: float = None
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
        particles_total_energy = (beam_energy
                                  + beam.read_partial_dE())
        self.energy_lost_due_to_synchrotron_radiation_section = (Cgamma * _DI_wE[1] / (
                2.0 * np.pi) * particles_total_energy**4.0)

        self.damping_partition_number_section = 2 + _DI_wE[3]/_DI_wE[1] # longitudinal damping number
        self.damping_time_section = (2.0 / self.damping_partition_number_section
                                     * beam.read_partial_dE() /
                                     self.energy_lost_due_to_synchrotron_radiation_section)


        self.natural_energy_spread_section = np.sqrt( Cq *
                            (beam.reference_total_energy/m_e)**2.0
                                * self.I3 / (self.jz * self.I2))

        # check validity
        beam.write_partial_dE += (- U0_wiggler - 2.0 / tau_z  * beam._dE - 2.0 *
                      self.sigma_dE /
                              np.sqrt(self.tau_z * self.n_kicks)
                              * self.beam.energy * np.random.normal
                              (size=self.beam.n_macroparticles))

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
        super().__init__(name=name,
                         section_index=section_index
                         )

        self.type = wiggler_type,
        self.number = number,
        self.peak_field = peak_field,
        self.pole_length = pole_length,
        self.number_poles = number_poles,

        self._simulation: LateInit[Simulation] = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: \
            LateInit[NumpyArray | CupyArray] = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: \
            LateInit[NumpyArray | CupyArray] = np.zeros((1, 5))
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
        self._DI_woE = np.array(
            [
                (
                    -self.number
                    * self.length_wiggler
                    * (e * self.peak_field) ** 2
                    * self.length_wiggler
                    / (2 * np.pi)
                ),
                1
                / 2
                * self.number
                * self.length_wiggler
                * (e * self.peak_field) ** 2,
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
        E = beam._dE # check if alright
        energy_contribution_wiggler_integrals = np.array([
            1/ (E * e / c)**2,
            1/ (E * e / c)**2,
            1/ (E * e / c)**3,
            1/ (E * e / c)**3,
            1/ (E * e / c)**5
        ])
        self._contribution_to_synchrotron_radiation_integrals_with_energy = (
            np.multiply(self._contribution_to_synchrotron_radiation_integrals_without_energy,
                                  energy_contribution_wiggler_integrals))
        pass



    def track(self, beam: BeamBaseClass) -> None:
        for beam in self._simulation.beams:
            self.update_synchrotron_radiation_integrals(beam = beam)
            self.update_beam_energy(beam)
        pass
