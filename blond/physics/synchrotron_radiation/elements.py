# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Synchrotron radiation ring elements.

Author:
L. Valle
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c, e

from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.base import BeamPhysicsRelevant, DynamicParameter

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
    """
    Base class for radiating ring elements.

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
        name: str | None = None,
        section_index: int | None = None,
        share_of_synchrotron_radiation_integrals: NumpyArray | None = None,
        seed: int | None = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._simulation: Simulation | None = None
        self._turn_i: DynamicParameter | None = 0
        self._fractional_radiation_integrals = (
            share_of_synchrotron_radiation_integrals
        )
        self.rng = np.random.default_rng(seed=seed)

    def _calculate_kick(
        self,
        beam: BeamBaseClass,
        seed: int | None = None,
    ) -> NumpyArray:
        """
        Energy kick induced by synchrotron radiation and quantum excitation.

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
        # TODO: does it make sense to have the contribution of these
        # parameters per base class?
        self._natural_energy_spread = sigma0
        self._energy_lost_due_to_synchrotron_radiation = U0
        self._damping_time = tau_z
        # fixme How to integrate the random generator??? Best practice?
        return -2.0 / tau_z * beam.read_partial_dE() - 2.0 * sigma0 / np.sqrt(
            tau_z
        ) * beam.reference_total_energy * self.rng.normal(
            size=beam.n_macroparticles_partial()
        )

    def _update_beam_energy(
        self,
        beam: BeamBaseClass,
        seed: int | None = None,
    ):
        """
        Update the beam partial energy with radiation damping and excitation.

        Function to update the beam partial energy including the energy lost by
        synchrotron radiation, its damping effect and the quantum
        excitation. Energy kick computed from self._calculate_kick method.

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
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        self._turn_i = simulation.turn_i

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
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
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        self._turn_i = self._simulation.turn_i
        self._update_beam_energy(beam)


class SynchrotronRadiationDrift(SynchrotronRadiationBaseClass):
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


class SynchrotronRadiationSection(SynchrotronRadiationBaseClass):
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


class WigglerMagnet(SynchrotronRadiationBaseClass):
    """
    Synchrotron Radiation subclass to include a damping wiggler in the ring.

    This class simulates the effect of one or a series of identical damping
    wigglers on the simulated beams.

    Parameters
    ----------
    name
        Name of the damping wigglers
    section_index
    wiggler_type
    number
    peak_field
    pole_length
    number_poles
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
        self._contribution_to_synchrotron_radiation_integrals_without_energy: (
            NumpyArray | None
        ) = np.zeros((1, 5))
        self._contribution_to_synchrotron_radiation_integrals_with_energy: (
            NumpyArray | None
        ) = np.zeros((1, 5))

    @property
    def number_of_wigglers(self):
        """Number of damping wigglers."""
        return self._number

    @property
    def length_wiggler(self):
        """Length of each damping wiggler."""
        if self._type == "sinusoidal":
            return self.pole_length * self._number_poles
        else:
            return None

    @property
    def number_of_poles(self):
        """Number of poles per wiggler."""
        return self._number_poles

    @property
    def peak_magnetic_field(self):
        """Peak magnetic field per wiggler."""
        return self._peak_field

    @property
    def pole_length(self):
        """Pole length per wiggler."""
        return self._pole_length

    def __str__(self):
        """Method to print general information about the created class."""
        return (
            f"{self.number_of_wigglers} damping wigglers of {self.peak_magnetic_field} T "
            f"and composed of {self.number_of_poles} poles of {
                self.pole_length
            } m each have been added to "
            f"the "
            f"simulation. \n"
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        self._simulation = simulation
        self._calculate_contribution_to_synchrotron_radiation_integrals()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
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
        self._turn_i = simulation.turn_i

    def _calculate_contribution_to_synchrotron_radiation_integrals(self):
        """Calculates the wiggler radiation integrals without beam energy."""
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
        Function to update the synchrotron radiation integrals.

        The damping wiggler enhances synchrotron radiation damping and
        changes the synchrotron radiation. This function updates the
        synchrotron radiation integrals variation from the damping
        wiggler according the beam energy.

        Parameters
        ----------
        beam
            Beam class to interact with this element
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

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        self._turn_i = self._simulation.turn_i
        self.update_synchrotron_radiation_integrals(beam=beam)
        self._update_beam_energy(beam)
