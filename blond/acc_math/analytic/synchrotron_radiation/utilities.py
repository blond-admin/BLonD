# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Collection of gathering functions for synchrotron radiation simulations.

Notes
-----
Authors:
L.Valle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import (
    calculate_energy_loss_per_turn,
    calculate_longitudinal_damping_time_in_turns,
    calculate_natural_energy_spread,
)
from blond.core.beam.particle_types import ParticleType, electron

if TYPE_CHECKING:
    from numpy import ndarray as NumpyArray


def gather_longitudinal_synchrotron_radiation_parameters(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType = electron,
) -> tuple[
    float | NumpyArray,
    float | NumpyArray,
    float | NumpyArray,
]:
    """
    Calculate the relevant synchrotron radiation parameters for tracking.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    energy_lost_from_synchrotron_radiation
        Energy loss per turn, in [eV per turn].
    longitudinal_damping_time
        Longitudinal damping time, in [turn].
    natural_energy_spread
        Natural energy spread, [dimensionless].
    """
    energy_lost_from_synchrotron_radiation = calculate_energy_loss_per_turn(
        particle_type=particle_type,
        energy=energy,
        radiation_integrals=radiation_integrals,
    )
    longitudinal_damping_time = calculate_longitudinal_damping_time_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
    )
    natural_energy_spread = calculate_natural_energy_spread(
        particle_type=particle_type,
        energy=energy,
        radiation_integrals=radiation_integrals,
    )
    return (
        energy_lost_from_synchrotron_radiation,
        longitudinal_damping_time,
        natural_energy_spread,
    )


def calculate_isomagnetic_radiation_integrals(
    circumference: float,
    bending_radius: float,
    momentum_compaction_factor: float,
) -> NumpyArray:
    """
    Generate the radiation integrals in the case of an isomagnetic ring.

    Warning: the fifth synchrotron radiation is set to 0 for lack of
    information; the natural horizontal emittance therefore is 0.

    Parameters
    ----------
    circumference
        Circumference, in [m].
    bending_radius
        Bending radius of all bending elements, in [m].
    momentum_compaction_factor
        Momentum compaction factor, [dimensionless].

    Returns
    -------
    isomagnetic_radiation_integrals
        Array of the first five radiation integrals of an isomagnetic
        synchrotron.
    """
    isomagnetic_radiation_integrals = np.array(
        [
            momentum_compaction_factor * circumference,
            2.0 * np.pi / bending_radius,
            2.0 * np.pi / bending_radius**2,
            momentum_compaction_factor * circumference / bending_radius**2,
            0,
        ]
    )
    return isomagnetic_radiation_integrals
