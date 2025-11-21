from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c, e

from blond._core.base import BeamPhysicsRelevant, DynamicParameter
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)

if TYPE_CHECKING:

    from numpy.typing import NDArray as NumpyArray

    from blond._core.beam.base import BeamBaseClass
    from blond._core.simulation.simulation import Simulation


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
    """
    Base class to handle the synchrotron radiation energy loss and damping,
    and quantum excitation effect along a section of the ring.
    """

    def __str__(self):
        return "Synchrotron radiation section element."

    def __init__(
        self,
        fractional_radiation_integrals: NumpyArray,
        name: str | None = None,
        section_index: int | None = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._simulation: Simulation | None = None
        self._fractional_radiation_integrals = fractional_radiation_integrals
        self._turn_i: DynamicParameter | None = 0

    def _calculate_kick(self, beam: BeamBaseClass) -> NumpyArray:
        """
        Function to calculate the energy kick induced by the energy lost by
        synchrotron radiation, its damping effect and the quantum excitation.
        Function used to update the beam partial energy dE.

        Parameters
        ----------
        beam
             BeamBaseClass object

        Returns
        -------
            Energy kick to be applied on the energy coordinates of the beam
        """
        U0, tau_z, sigma0 = (
            gather_longitudinal_synchrotron_radiation_parameters(
                particle_type=beam.particle_type,
                energy=beam.reference_total_energy,
                synchrotron_radiation_integrals=self._fractional_radiation_integrals,
            )
        )
        self._natural_energy_spread[self._turn_i] = np.average(sigma0)
        self._energy_lost_due_to_synchrotron_radiation[self._turn_i] = (
            np.average(U0)
        )
        self._damping_time[self._turn_i] = np.average(tau_z)

        return -2.0 / tau_z * beam.read_partial_dE() - 2.0 * sigma0 / np.sqrt(
            tau_z
        ) * beam.reference_total_energy * np.random.normal(
            size=len(beam.n_macroparticles_partial())
        )

    def _update_beam_energy(self, beam: BeamBaseClass):
        """
        Function to update the beam partial energy with radiation damping
        and quantum excitation

        Parameters
        ----------
        beam
            BeamBaseClass object
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

    def track(self, beam: BeamBaseClass) -> None:
        self._turn_i = self._simulation.turn_i
        self._update_beam_energy(beam)


class SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        fraction_of_ring_circumference: float = None,
        share_of_synchrotron_radiation_integrals: NumpyArray = None,
        is_isomagnetic: bool = False,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._fraction_of_ring_circumference = fraction_of_ring_circumference
        self._share_of_synchrotron_radiation_integrals = (
            share_of_synchrotron_radiation_integrals
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_drift(self):
        """Energy lost by passing through the drift"""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def share_of_synchrotron_radiation_integrals(self):
        return self._share_of_synchrotron_radiation_integrals

    @property
    def synchrotron_radiation_integrals_drift(self):
        """Synchrotron radiation integrals of the drift"""
        return self._share_of_synchrotron_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass
        self._turn_i = simulation.turn_i


class SynchrotronRadiationSection(SynchrotronRadiationBaseClass):
    # TODO : enforce a constraint on the number of
    # SynchrotronRadiationSection per section
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        fraction_of_ring_circumference: float = None,
        share_of_synchrotron_radiation_integrals: NumpyArray = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._fraction_of_ring_circumference = fraction_of_ring_circumference
        self._share_of_synchrotron_radiation_integrals = (
            share_of_synchrotron_radiation_integrals
        )

    @property
    def energy_lost_due_to_synchrotron_radiation_section(self):
        """Energy lost by passing through the section"""
        return self._energy_lost_due_to_synchrotron_radiation

    @property
    def share_of_synchrotron_radiation_integrals(self):
        return self._share_of_synchrotron_radiation_integrals

    @property
    def synchrotron_radiation_integrals_section(self):
        """Synchrotron radiation integrals of the section"""
        return self._share_of_synchrotron_radiation_integrals

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass
        self._turn_i = simulation.turn_i
        lengths_sections = self._simulation.ring.section_lengths
        share_synchrotron_radiation_integrals = (
            lengths_sections[self.section_index]
            / self._simulation.ring.circumference
        )
        self._synchrotron_radiation_integrals = (
            share_synchrotron_radiation_integrals
        ) * self._synchrotron_radiation_integrals


class WigglerMagnet(SynchrotronRadiationBaseClass):
    """
    Synchrotron Radiation subclass to simulate the effect of one or a
    series of identical damping wigglers on the simulated beams.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int | None = None,
        wiggler_type: str | None = "sinusoidal",
        number: int | None = 1,
        peak_field: float | None = 1.0,
        pole_length: float | None = 0.095,
        number_poles: int | None = 43,
    ):
        super().__init__(name=name, section_index=section_index)

        self._type = (wiggler_type,)
        self._number = (number,)
        self._peak_field = (peak_field,)
        self._pole_length = (pole_length,)
        self._number_poles = (number_poles,)

        self._simulation: Simulation | None = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: NumpyArray | None = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: NumpyArray | None = np.zeros((1, 5))

    @property
    def number_of_wigglers(self):
        return self._number

    @property
    def length_wiggler(self):
        if self._type == "sinusoidal":
            return self.pole_length * self._number_poles
        else:
            return None

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
        self.calculate_contribution_to_synchrotron_radiation_integrals()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        self._turn_i = simulation.turn_i

    def calculate_contribution_to_synchrotron_radiation_integrals(self):
        """
        Function to initialize the energy-free fraction of the damping
        wiggler radiation integrals.
        :return:
        """
        self._contribution_to_synchrotron_radiation_integrals_without_energy = np.array(
            [
                (
                    -1
                    * self.number_of_wigglers
                    * self.length_wiggler
                    * (e * self.peak_magnetic_field) ** 2
                    * self.length_wiggler
                    / (2 * np.pi)
                ),
                1
                / 2
                * self.number_of_wigglers
                * self.length_wiggler
                * (e * self.peak_magnetic_field) ** 2,
                4
                / (3 * np.pi)
                * self.number_of_wigglers
                * self.length_wiggler
                * (e * self.peak_magnetic_field) ** 3,
                0,
                self.number_of_wigglers
                * self.pole_length**2
                * self.length_wiggler
                / (15 * np.pi**3)
                * (e * self.peak_magnetic_field) ** 5,
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
                var**2,
                var**2,
                var**3,
                var**3,
                var**5,
            ]
        )
        self._contribution_to_synchrotron_radiation_integrals_with_energy = np.multiply(
            self._contribution_to_synchrotron_radiation_integrals_without_energy,
            energy_contribution_wiggler_integrals,
        )
        pass

    def track(self, beam: BeamBaseClass) -> None:
        self._turn_i = self._simulation.turn_i
        self.update_synchrotron_radiation_integrals(beam=beam)
        self._update_beam_energy(beam)
