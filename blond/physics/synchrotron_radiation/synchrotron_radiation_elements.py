# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Synchrotron radiation ring elements.

Author:
L. Valle
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import dtype, ndarray
from numpy.random import Generator
from scipy.constants import c, e

from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.base import BeamPhysicsRelevant, DynamicParameter

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def calculation_synchrotron_radiation_and_quantum_excitation_energy_kick(
    energy_lost: float,
    beam_delta_energy_array: NumpyArray,
    random_generator: Generator,
    natural_energy_spread: float,
    longitudinal_damping_time: float,
    total_energy: float,
    _disable_quantum_excitation: bool = False,
) -> float | ndarray[tuple[int, ...], dtype[Any]] | Any:
    """
    Energy kick induced by synchrotron radiation and quantum excitation.

    Function to calculate the energy kick induced by the energy lost by
    synchrotron radiation, its damping effect and the quantum excitation.
    Class independent.

    Parameters
    ----------
    energy_lost
        Energy lost through the considered synchrotron segment.
    beam_delta_energy_array
        Beam energy array.
    random_generator
        Random generator.
    natural_energy_spread
        Natural energy spread of the considered synchrotron segment.
    longitudinal_damping_time
        Longitudinal damping time of the considered synchrotron segment.
    total_energy
        Beam total reference energy.
    _disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.

    Returns
    -------
    energy_kick
        Energy kick induced by synchrotron radiation and quantum excitation.
    """
    if _disable_quantum_excitation:
        energy_kick = (
            -energy_lost
            - 2.0 / longitudinal_damping_time * beam_delta_energy_array
        )
    else:
        energy_kick = (
            -energy_lost
            - 2.0 / longitudinal_damping_time * beam_delta_energy_array
            + 2.0
            * natural_energy_spread
            / np.sqrt(longitudinal_damping_time)
            * total_energy
            * random_generator.normal(size=len(beam_delta_energy_array))
        )
    return energy_kick


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
    """
    Base class for radiating ring elements.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_synchrotron_radiation_integrals
        Fractional synchrotron radiation integrals.
    _disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.
    seed
        Random seed parameter.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int | None = None,
        share_of_synchrotron_radiation_integrals: NumpyArray | None = None,
        _disable_quantum_excitation: bool = False,
        seed: int | None = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._simulation: Simulation | None = None
        self._turn_i: DynamicParameter | None = 0
        self._fractional_radiation_integrals = (
            share_of_synchrotron_radiation_integrals
        )

        self._disable_quantum_excitation = _disable_quantum_excitation

        self._energy_lost_due_to_synchrotron_radiation: float | None = None
        self._damping_time: float | None = None
        self._natural_energy_spread: float | None = None

        self.rng = np.random.default_rng(seed=seed)

    def _calculate_kick(
        self,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Energy kick induced by synchrotron radiation and quantum excitation.

        Function to calculate the energy kick induced by the energy lost by
        synchrotron radiation, its damping effect and the quantum excitation.
        Function used to update the beam partial energy dE.

        Parameters
        ----------
        beam
             BeamBaseClass object.

        Returns
        -------
        energy_kick
            Energy kick to be applied on the energy coordinates of the beam.
        """
        total_energy = beam.reference.total_energy
        (
            estimated_energy_lost,
            estimated_damping_time,
            estimated_natural_energy_spread,
        ) = gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=beam.particle_type,
            energy=total_energy,
            synchrotron_radiation_integrals=self._fractional_radiation_integrals,
        )
        self._energy_lost_due_to_synchrotron_radiation = estimated_energy_lost
        self._damping_time = estimated_damping_time
        self._natural_energy_spread = estimated_natural_energy_spread

        beam_dE = beam.read_partial_dE()
        random_generator = self.rng
        return calculation_synchrotron_radiation_and_quantum_excitation_energy_kick(
            energy_lost=estimated_energy_lost,
            beam_delta_energy_array=beam_dE,
            random_generator=random_generator,
            natural_energy_spread=estimated_natural_energy_spread,
            longitudinal_damping_time=estimated_damping_time,
            total_energy=total_energy,
            _disable_quantum_excitation=self._disable_quantum_excitation,
        )

    def _update_beam_energy(
        self,
        beam: BeamBaseClass,
    ):
        """
        Update the beam partial energy with radiation damping and excitation.

        Function to update the beam partial energy including the energy lost by
        synchrotron radiation, its damping effect and the quantum
        excitation. Energy kick computed from self._calculate_kick method.

        Parameters
        ----------
        beam
            BeamBaseClass object.
        """
        # TODO write C++ routine
        energy_change = self._calculate_kick(beam=beam)
        dE = beam.write_partial_dE()
        dE[:] += energy_change

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        self._turn_i = simulation.turn_i

    def on_run_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments for simulation setup.
        """
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self._turn_i = self._simulation.turn_i
        self._update_beam_energy(beam)


class WigglerMagnet(SynchrotronRadiationBaseClass):
    """
    Synchrotron Radiation subclass to include a damping wiggler in the ring.

    This class simulates the effect of one or a series of identical damping
    wigglers on the simulated beams.

    Parameters
    ----------
    name
        Name of the damping wigglers.
    section_index
        Section index.
    wiggler_type
        Type of damping wiggler. Default: 'sinusoidal'.
    number
        Number of damping wigglers.
    peak_field
        Magnetic peak field per wiggler.
    pole_length
        Pole length.
    number_poles
        Number of poles per wiggler.
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

        self._type = wiggler_type
        self._number = number
        self._peak_field = peak_field
        self._pole_length = pole_length
        self._number_poles = number_poles

        self._simulation: Simulation | None = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: (
            NumpyArray | None
        ) = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: (
            NumpyArray | None
        ) = np.zeros((1, 5))

    @property
    def number_of_wigglers(self):
        """
        Number of damping wigglers.

        Returns
        -------
        number_of_wigglers
            Number of damping wigglers.
        """
        return self._number

    @property
    def length_wiggler(self):
        """
        Length of each damping wiggler.

        Returns
        -------
        length_wiggler
            Length of each damping wiggler.
        """
        if self._type == "sinusoidal":
            return self.pole_length * self._number_poles
        else:
            return None

    @property
    def number_of_poles(self):
        """
        Number of poles per wiggler.

        Returns
        -------
        number_of_poles
            Number of poles.
        """
        return self._number_poles

    @property
    def peak_magnetic_field(self):
        """
        Peak magnetic field per wiggler.

        Returns
        -------
        peak_magnetic_field
            Magnetic peak field.
        """
        return self._peak_field

    @property
    def pole_length(self):
        """
        Pole length per wiggler.

        Returns
        -------
        pole_length
            Pole length per wiggler.
        """
        return self._pole_length

    def __str__(self):
        """
        Method to print general information about the created class.

        Returns
        -------
        message
            Prints the characteristics of the initialised wiggler class.
        """
        return (
            f"{self.number_of_wigglers} damping wigglers of {self.peak_magnetic_field} T "
            f"and composed of {self.number_of_poles} poles of {self.pole_length}"
            f" m each have been added to "
            f"the "
            f"simulation. \n"
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._calculate_contribution_to_synchrotron_radiation_integrals_without_beam_energy()

    def on_run_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments for simulation setup.
        """
        super().on_run_simulation(simulation=simulation)

    def _calculate_energy_contribution_to_synchrotron_radiation_integrals(
        self, reference_energy: float
    ):
        """
        Calculate the wiggler contribution to the radiation integrals.

        The damping wiggler enhances synchrotron radiation damping and
        changes the synchrotron radiation. This function calculates the
        synchrotron radiation integrals variation from the damping
        wiggler according the beam energy.

        Parameters
        ----------
        reference_energy
            Beam reference energy.

        Returns
        -------
        energy_contribution_wiggler_integrals
            Wiggler contribution to the synchrotron radiation integrals.
        """
        var = 1 / (reference_energy * e / c)
        energy_contribution_wiggler_integrals = np.array(
            [
                var**2,
                var**2,
                var**3,
                var**3,
                var**5,
            ]
        )
        return energy_contribution_wiggler_integrals

    def _calculate_contribution_to_synchrotron_radiation_integrals_without_beam_energy(
        self,
    ):
        """Calculate the wiggler radiation integrals without beam energy."""
        if self._type == "sinusoidal":
            self._contribution_to_synchrotron_radiation_integrals_without_energy = np.array(
                [
                    (
                        -1
                        / 2
                        * self.number_of_wigglers
                        * self.length_wiggler
                        * (e * self.peak_magnetic_field) ** 2
                        * (self.length_wiggler / (2 * np.pi)) ** 2
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
        else:
            self._contribution_to_synchrotron_radiation_integrals_without_energy = None

    def update_synchrotron_radiation_integrals(
        self,
        beam: BeamBaseClass,
        radiation_integrals: NumpyArray,
    ):
        """
        Function to update the synchrotron radiation integrals.

        The damping wiggler enhances synchrotron radiation damping and
        changes the synchrotron radiation. This function updates the
        synchrotron radiation integrals variation from the damping
        wiggler according the beam energy.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        radiation_integrals
            Synchrotron radiation integrals fron ring.
        """
        energy_contribution_wiggler_integrals = self._calculate_energy_contribution_to_synchrotron_radiation_integrals(
            reference_energy=beam.reference.total_energy
        )

        self._contribution_to_synchrotron_radiation_integrals_with_energy = np.multiply(
            self._contribution_to_synchrotron_radiation_integrals_without_energy,
            energy_contribution_wiggler_integrals,
        )
        self._fractional_radiation_integrals = self._contribution_to_synchrotron_radiation_integrals_without_energy

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # The super updates the beam energy. Any calculation should be
        # conducted beforehand.
        self.update_synchrotron_radiation_integrals(beam=beam)
        super().track(beam=beam)
