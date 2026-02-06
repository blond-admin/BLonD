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

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import dtype, ndarray
from scipy.constants import c, e

from blond.physics.synchrotron_radiation.base import (
    SynchrotronRadiationBaseClass,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class WigglerMagnet(SynchrotronRadiationBaseClass):
    """
    Synchrotron Radiation subclass to include a damping wiggler in the ring.

    This class simulates the effect of one or a series of identical damping
    wigglers on the simulated beams.

    Parameters
    ----------
    peak_field
        Magnetic peak field per wiggler.
    pole_length
        Pole length.
    number_of_poles
        Number of poles per wiggler.
    number_of_wigglers
        Number of damping wigglers.
    wiggler_type
        Type of damping wiggler. Default: 'sinusoidal'.
    name
        Name of the damping wigglers.
    section_index
        Section index.
    """

    def __init__(
        self,
        peak_field: float,
        pole_length: float,
        number_of_poles: int,
        number_of_wigglers: int = 1,
        wiggler_type: str = "sinusoidal",
        name: str = "",
        section_index: int = 0,
    ):
        super().__init__(name=name, section_index=section_index)

        self._type = wiggler_type
        self._number_of_wigglers = number_of_wigglers
        self._peak_field = peak_field
        self._pole_length = pole_length
        self._number_of_poles = number_of_poles

        self._simulation: Simulation | None = None
        self._contribution_to_synchrotron_radiation_integrals_without_energy: (
            NumpyArray | None
        ) = np.zeros(5)
        self._contribution_to_synchrotron_radiation_integrals_with_energy: (
            NumpyArray | None
        ) = np.zeros(5)

    @property
    def number_of_wigglers(self) -> int:
        """
        Number of damping wigglers.

        Returns
        -------
        number_of_wigglers
            Number of damping wigglers.
        """
        return self._number_of_wigglers

    @property
    def length_wiggler(self) -> float | None:
        """
        Length of each damping wiggler.

        Returns
        -------
        length_wiggler
            Length of each damping wiggler.
        """
        if self._type == "sinusoidal":
            return self.pole_length * self._number_of_poles
        else:
            return None

    @property
    def number_of_poles(self) -> int:
        """
        Number of poles per wiggler.

        Returns
        -------
        number_of_poles
            Number of poles.
        """
        return self._number_of_poles

    @property
    def peak_magnetic_field(self) -> float:
        """
        Peak magnetic field per wiggler.

        Returns
        -------
        peak_magnetic_field
            Magnetic peak field.
        """
        return self._peak_field

    @property
    def pole_length(self) -> float:
        """
        Pole length per wiggler.

        Returns
        -------
        pole_length
            Pole length per wiggler.
        """
        return self._pole_length

    def __str__(self) -> str:
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
    ) -> NumpyArray:
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
    ) -> NumpyArray | None:
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
        beam_reference_energy: float,
        calculation_only: bool = False,
    ) -> ndarray[tuple[Any, ...], dtype[Any]] | None:
        """
        Function to update the synchrotron radiation integrals.

        The damping wiggler enhances synchrotron radiation damping and
        changes the synchrotron radiation. This function updates the
        synchrotron radiation integrals variation from the damping
        wiggler according the beam energy.

        Parameters
        ----------
        beam_reference_energy
            Beam reference energy.
        calculation_only
            If enabled, the calculated wiggler radiation integrals will be
            outputed. The internal properties will not be updated. False by
            default.

        Returns
        -------
        wiggler_radiation_integrals
            Wiggler contribution to the synchrotron radiation integrals if
            calculation_only is True.
        """
        energy_contribution_wiggler_integrals = self._calculate_energy_contribution_to_synchrotron_radiation_integrals(
            reference_energy=beam_reference_energy
        )

        wiggler_radiation_integrals = np.multiply(
            self._contribution_to_synchrotron_radiation_integrals_without_energy,
            energy_contribution_wiggler_integrals,
        )
        if not calculation_only:
            self._contribution_to_synchrotron_radiation_integrals_with_energy = wiggler_radiation_integrals
            self._share_of_synchrotron_radiation_integrals = self._contribution_to_synchrotron_radiation_integrals_with_energy
            return None
        else:
            return wiggler_radiation_integrals

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # The super updates the beam energy. Any calculation should be
        # conducted beforehand.
        self.update_synchrotron_radiation_integrals(
            beam_reference_energy=beam.reference.total_energy
        )
        super().track(beam=beam)
