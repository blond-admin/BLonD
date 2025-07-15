import numpy as np
from scipy.constants import e, c, m_e

from blond3 import Simulation, DriftSimple, DriftBaseClass
from blond3._core.base import BeamPhysicsRelevant
from blond3._core.beam.base import BeamBaseClass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Optional as LateInit
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray
    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass


class WigglerMagnet(BeamPhysicsRelevant):
    """Damping wiggler magnet class"""

    def __init__(
        self,
        wiggler_type: Optional[str] = "sinusoidal",
        number_wigglers: Optional[int] = 1,
        peak_field: Optional[float] = 1.0,
        pole_length: Optional[float] = 0.095,
        number_poles: Optional[int] = 43,
    ):
        super().__init__(

        )
        self.wiggler_type = wiggler_type,
        self.number_wigglers = number_wigglers,
        self.peak_field = peak_field,
        self.pole_length = pole_length,
        self.number_poles = number_poles,

        self._simulation: LateInit[Simulation] = None
        self._DI_woE: LateInit[NumpyArray | CupyArray] = np.zeros((1,
                                                                   5)) #fixme
        self._DI_wE: LateInit[NumpyArray | CupyArray] = np.zeros((1,
                                                                   5)) #fixme
    @property
    def length_wiggler(self):
        if self.wiggler_type == "wiggler_type":
            return self.pole_length * self.number_poles

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation


    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        self.calculate_wiggler_integrals(self)
        pass

    def calculate_wiggler_integrals(self):
        """
        Function to initialize the energy-free fraction of the damping
        wiggler radiation integrals.
        :return:
        """
        self._DI_woE = np.array(
            [
                (
                    -self.number_wigglers
                    * self.length_wiggler
                    * (e * self.peak_field) ** 2
                    * self.length_wiggler
                    / (2 * np.pi)
                ),
                1
                / 2
                * self.number_wigglers
                * self.length_wiggler
                * (e * self.peak_field) ** 2,
                4
                / (3 * np.pi)
                * self.number_wigglers
                * self.length_wiggler
                * (e * self.peak_field) ** 3,
                0,
                self.number_wigglers
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
        self._DI_wE = np.multiply(self._DI_woE,
                                  energy_contribution_wiggler_integrals)
        pass

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

        U0_wiggler = (beam.particle_type.c_gamma() * self._DI_wE[1]
                      / (2.0 * np.pi)) * beam._dE** 4.0 # in eV per turn
        jz = 2 + self._DI_wE[3]/self._DI_wE[1] # longitudinal damping number
        tau_z = 2.0 / jz * beam._dE / U0_wiggler # longitudinal damping time
        # in turns #


        sigma_dE0 = np.sqrt(beam.particle_type.c_q() * (beam._dE/m_e)**2.0
                                * self.I3 / (self.jz * self.I2))

        # check validity
        beam._dE += (- U0_wiggler - 2.0 / tau_z  * beam._dE - 2.0 * self.sigma_dE /
                              np.sqrt(self.tau_z * self.n_kicks)
                              * self.beam.energy * np.random.normal
                              (size=self.beam.n_macroparticles))

    def track(self, beam: BeamBaseClass) -> None:
        for beam in self._simulation.beams:
            self.update_synchrotron_radiation_integrals(beam = beam)
            self.update_beam_energy(beam)
        pass


class SynchrotronRadiation(BeamPhysicsRelevant):
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
        raise NotImplementedError("For Lina")
        self._simulation: LateInit[Simulation] = None

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
            self.I2 = 2.0 * np.pi / self.rho
            self.I3 = 2.0 * np.pi / self.rho ** 2.0
            self.I4 = (self.ring.ring_circumference *
                       self.ring.alpha_0[0, 0] / self.rho ** 2.0)
            self.jz = 2.0 + self.I4 / self.I2

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation
        self.init_synchrotron_radiation_integrals()


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

    def generate_children(self, ring, simulation):

        drifts = ring.get_elements(DriftBaseClass)

    def calculate_synchrotron_radiation_parameters(self):
        U0_wiggler = (beam.particle_type.c_gamma() * self._DI_wE[1]
                      / (2.0 * np.pi)) * beam._dE ** 4.0  # in eV per turn


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

        U0_wiggler = (beam.particle_type.c_gamma() * self._DI_wE[1]
                      / (2.0 * np.pi)) * beam._dE** 4.0 # in eV per turn
        jz = 2 + self._DI_wE[3]/self._DI_wE[1] # longitudinal damping number
        tau_z = 2.0 / jz * beam._dE
                      / U0_wiggler # longitudinal damping time in turns #


        sigma_dE0 = np.sqrt(beam.particle_type.c_q() * (beam._dE/E0)**2.0
                                * self.I3 / (self.jz * self.I2))
        # check validity
        beam._dE += - U0_wiggler # energy lost through the damping wiggler
                    - 2.0 / tau_z  * beam._dE
                    - 2.0 * self.sigma_dE /
                              np.sqrt(self.tau_z * self.n_kicks)
                              * self.beam.energy * np.random.normal
                              (size=self.beam.n_macroparticles))

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass
