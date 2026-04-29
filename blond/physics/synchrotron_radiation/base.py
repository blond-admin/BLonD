# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Synchrotron radiation base classes.

Author:
L. Valle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray as NumpyArray

from blond import backend
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def calculation_synchrotron_radiation_and_quantum_excitation_energy_kick(
    beam_delta_energy_array: NumpyArray,
    energy_lost: float,
    longitudinal_damping_time: float,
    natural_energy_spread: float | None = None,
    total_energy: float | None = None,
    random_generator: Generator | None = None,
    disable_quantum_excitation: bool = False,
) -> float | NumpyArray:
    """
    Energy kick induced by synchrotron radiation and quantum excitation.

    Function to calculate the energy kick induced by the energy lost by
    synchrotron radiation, its damping effect and the quantum excitation.
    Class independent.

    Parameters
    ----------
    beam_delta_energy_array
        Beam energy array.
    energy_lost
        Energy lost through the considered synchrotron segment, in [eV per
        turn].
    longitudinal_damping_time
        Longitudinal damping time of the considered synchrotron segment,
        in [turn].
    natural_energy_spread
        Natural energy spread of the considered synchrotron segment,
        [dimensionless].
    total_energy
        Beam total reference energy, in [eV].
    random_generator
        Random generator.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.

    Returns
    -------
    energy_kick
        Energy kick induced by synchrotron radiation and quantum excitation.
    """
    if disable_quantum_excitation:
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
            * random_generator.standard_normal(
                size=len(beam_delta_energy_array)
            )
        )
    return backend.cast_arr_float_if_needed(energy_kick)


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, Schedulable):
    """
    Base class for radiating ring elements.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_radiation_integrals
        Share of synchrotron radiation integrals.
    disable_quantum_excitation
        Expert user only. Disables the quantum excitation kick.
    seed
        Random seed parameter.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int | None = None,
        share_of_radiation_integrals: NumpyArray | None = None,
        disable_quantum_excitation: bool = False,
        seed: int | None = None,
    ):
        super().__init__(name=name, section_index=section_index)

        self._add_intended_schedule(
            "share_of_radiation_integrals",
        )

        self._simulation: Simulation | None = None
        self._turn_i: DynamicParameter | int = 0
        self.share_of_radiation_integrals = share_of_radiation_integrals

        self._disable_quantum_excitation = disable_quantum_excitation

        self._energy_lost_due_to_synchrotron_radiation: float | None = None
        self._damping_time: float | None = None
        self._natural_energy_spread: float | None = None

        self.rng = backend.default_rng(seed=seed)
        # backend.default_rng

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
            radiation_integrals=self.share_of_radiation_integrals,
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
            disable_quantum_excitation=self._disable_quantum_excitation,
        )

    def _update_beam_energy(
        self,
        beam: BeamBaseClass,
    ) -> None:
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
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_i.value,
                reference_time=float(beam.reference.time),
            )
        self._update_beam_energy(beam)
