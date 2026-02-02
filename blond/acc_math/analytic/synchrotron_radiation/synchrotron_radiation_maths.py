# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Collection of functions to compute synchrotron radiation related quantities.

For relativistic charged particles, synchrotron radiation is emitted
along its direction of motion, which recoil induces small perturbation of
the betatron motion in all planes. This effect damps the beam amplitudes,
with typical damping times.
Practically, the synchrotron radiation damping times are proportional to the
inverse of 'U_0 / (2 T_0 E)', where 'U_0' is the energy loss per turn,
'T_0' the revolution period, and 'E' the beam energy. The proportionality
coefficient are the damping partition numbers:

'j_x = 1 - D'
'j_y = 1' (no vertical dispersion)
'j_z = 2 + D'

with D the damping partition.

Spontaneous emission of a quanta yields to an immediate energy change and
random small energy oscillations which tend to blow the beam sizes. Quantum
excitation and synchrotron radiation damping combined define a natural
equilibrium state of transverse and longitudinal beam emittances.

First five synchrotron radiation integrals are required in BLonD3 as an input
of the simulated ring:
            'I_1' = \int, related to the momentum compaction factor,
            'I_2' = , related to the energy loss per turn,
            'I_3' = , related to the natural energy spread,
            'I_4' =  , required for the damping times,
            'I_5' =  , required for the natural horizontal emittance
            with '\rho' the bending radius of bending elements, 'D' the
            horizontal dispersion function, 'K' the focusing strength and 'H =
            \beta_x D^2 + \alpha_x D {D'} + \gamma_x {D'}^2 ' the
            H-function

Notes
-----
Authors:
L. Valle

References
----------
Further information on synchrotron radiation damping and quantum excitation
can be found in:
- H. Wiedemann, Synchrotron Radiation, Springer, 2003
- S.Y. Lee, Accelerator Physics, World Scientific, Third edition,
2014 #check date
- A. Wolski, Introduction to Beam Dynamics in High-Energy Electron Storage
Rings, Morgan & Claypool Publishers, 2018
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from blond.core.beam.particle_types import ParticleType

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def calculate_partition_numbers(
    synchrotron_radiation_integrals: NumpyArray,
) -> NumpyArray:
    """
    Compute the damping partition numbers.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    damping_partition_numbers
        Damping partition numbers in the [horizontal, vertical, longitudinal] order.
    """
    jx = (
        1
        - synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )
    jy = 1
    jz = (
        2
        + synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )
    damping_partition_numbers = np.array([jx, jy, jz])
    return damping_partition_numbers


def calculate_horizontal_damping_partition_number(
    synchrotron_radiation_integrals: NumpyArray,
) -> float:
    """
    Compute the horizontal damping partition number.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    horizontal_damping_partition_number
        Horizontal damping partition number.
    """
    horizontal_damping_partition_number = (
        1
        - synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )
    return horizontal_damping_partition_number


def calculate_longitudinal_damping_partition_number(
    synchrotron_radiation_integrals: NumpyArray,
) -> float:
    """
    Compute the longitudinal damping partition number.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    longitudinal_damping_partition_number
        Longitudinal damping partition number.
    """
    longitudinal_damping_partition_number = (
        2
        + synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )
    return longitudinal_damping_partition_number


def calculate_damping_times_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> NumpyArray:
    """
    Calculate the damping times in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    damping_times_turn
        Damping times in turn in the [horizontal, vertical, longitudinal] order.
    """
    damping_partition_numbers = calculate_partition_numbers(
        synchrotron_radiation_integrals
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )

    damping_times_turn = np.array(
        [
            (2 * energy / damping_partition_numbers[k] / energy_loss_per_turn)
            for k in range(3)
        ]
    )
    return damping_times_turn


def calculate_horizontal_damping_time_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculate the horizontal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    horizontal_damping_time_turn
        Horizontal damping time in turns.
    """
    horizontal_damping_partition_number = (
        calculate_horizontal_damping_partition_number(
            synchrotron_radiation_integrals
        )
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )
    horizontal_damping_time_turn = (
        2 * energy / horizontal_damping_partition_number / energy_loss_per_turn
    )
    return horizontal_damping_time_turn


def calculate_longitudinal_damping_time_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculate the longitudinal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    longitudinal_damping_time_turn
        Longitudinal damping time in turns.
    """
    longitudinal_damping_partition_number = (
        calculate_longitudinal_damping_partition_number(
            synchrotron_radiation_integrals
        )
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )
    longitudinal_damping_time_turn = (
        2
        * energy
        / longitudinal_damping_partition_number
        / energy_loss_per_turn
    )
    return longitudinal_damping_time_turn


def calculate_damping_times_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> NumpyArray:
    """
    Calculate the damping times in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency, in [Hz].

    Returns
    -------
    damping_times
        Damping times in seconds in the [horizontal, vertical,
        longitudinal] order.
    """
    if isinstance(revolution_frequency, np.ndarray) and isinstance(
        energy, np.ndarray
    ):
        assert len(revolution_frequency) == len(energy)

    damping_partition_numbers = calculate_partition_numbers(
        synchrotron_radiation_integrals
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )
    damping_times = np.array(
        [
            (
                2
                * energy
                / damping_partition_numbers[k]
                / energy_loss_per_turn
                / revolution_frequency
            )
            for k in range(3)
        ]
    )
    return damping_times


def calculate_horizontal_damping_time_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculate the horizontal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency [Hz].

    Returns
    -------
    horizontal_damping_time
        Horizontal damping time in seconds.
    """
    horizontal_damping_partition_number = (
        calculate_horizontal_damping_partition_number(
            synchrotron_radiation_integrals
        )
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )
    horizontal_damping_time = (
        2
        * energy
        / horizontal_damping_partition_number
        / energy_loss_per_turn
        / revolution_frequency
    )
    return horizontal_damping_time


def calculate_longitudinal_damping_time_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculate the longitudinal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency [Hz].

    Returns
    -------
    longitudinal_damping_time
        Longitudinal damping time in seconds.
    """
    longitudinal_damping_partition_number = (
        calculate_longitudinal_damping_partition_number(
            synchrotron_radiation_integrals
        )
    )
    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        synchrotron_radiation_integrals,
        particle_type,
    )
    longitudinal_damping_time = (
        2
        * energy
        / longitudinal_damping_partition_number
        / energy_loss_per_turn
        / revolution_frequency
    )
    return longitudinal_damping_time


def calculate_energy_loss_per_turn(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the energy loss per turn from synchrotron radiation.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    energy_loss_per_turn
        Energy loss per turn, in [eV per turn].
    """
    energy_loss_per_turn = (
        particle_type.sands_radiation_constant
        * energy**4
        * synchrotron_radiation_integrals[1]
        / (2 * np.pi)
    )
    return energy_loss_per_turn


def calculate_natural_horizontal_emittance(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural horizontal emittance from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    natural_horizontal_emittance
        Natural horizontal emittance, in [m rad].
    """
    jx = calculate_horizontal_damping_partition_number(
        synchrotron_radiation_integrals,
    )
    natural_horizontal_emittance = (
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[4]
        / jx
        / synchrotron_radiation_integrals[1]
    )
    return natural_horizontal_emittance


def calculate_natural_energy_spread(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural energy spread from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    natural_energy_spread
        Natural energy spread, [dimensionless].
    """
    jz = calculate_longitudinal_damping_partition_number(
        synchrotron_radiation_integrals,
    )
    natural_energy_spread = np.sqrt(
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[2]
        / (jz * synchrotron_radiation_integrals[1])
    )
    return natural_energy_spread


def calculate_natural_bunch_length(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    angular_synchrotron_frequency: (float | NumpyArray),
    momentum_compaction_factor: float | NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural bunch length from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    synchrotron_radiation_integrals
        Synchrotron radiation integrals.
    angular_synchrotron_frequency
        Angular synchrotron frequency, in  [rad].
    momentum_compaction_factor
        Momentum compaction factor, [dimensionless].
    particle_type
        ParticleType class object.

    Returns
    -------
    natural_bunch_length
        Natural bunch length, in [m].
    """
    natural_energy_spread = calculate_natural_energy_spread(
        particle_type=particle_type,
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
    )
    natural_bunch_length = (
        momentum_compaction_factor
        * c
        / angular_synchrotron_frequency
        * natural_energy_spread
    )
    return natural_bunch_length
