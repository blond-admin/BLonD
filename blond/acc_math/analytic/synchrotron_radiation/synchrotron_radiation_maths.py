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
Further information on synchrotron radiation damping and quantum excitation
can be found in:
- H. Wiedemann, Synchrotron Radiation, Springer, 2003
- S.Y. Lee, Accelerator Physics, World Scientific, Third edition,
2014 #check date
- A. Wolski, Introduction to Beam Dynamics in High-Energy Electron Storage
Rings, Morgan & Claypool Publishers, 2018


Author:
L. Valle
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
    Computes the damping partition numbers.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals

    Returns
    -------
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
    return np.array([jx, jy, jz])


def calculate_horizontal_damping_partition_number(
    synchrotron_radiation_integrals: NumpyArray,
) -> float:
    """
    Computes the horizontal damping partition number.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals

    Returns
    -------
        Horizontal damping partition number.
    """
    return (
        1
        - synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )


def calculate_longitudinal_damping_partition_number(
    synchrotron_radiation_integrals: NumpyArray,
) -> float:
    """
    Computes the longitudinal damping partition number.

    Parameters
    ----------
    synchrotron_radiation_integrals
        Synchrotron radiation integrals

    Returns
    -------
        Longitudinal damping partition number.
    """
    return (
        2
        + synchrotron_radiation_integrals[3]
        / synchrotron_radiation_integrals[1]
    )


def calculate_damping_times_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> NumpyArray:
    """
    Calculates the damping times in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
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

    return np.array(
        [
            (2 * energy / damping_partition_numbers[k] / energy_loss_per_turn)
            for k in range(3)
        ]
    )


def calculate_horizontal_damping_time_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculates the horizontal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
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
    return (
        2 * energy / horizontal_damping_partition_number / energy_loss_per_turn
    )


def calculate_longitudinal_damping_time_in_turns(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculates the longitudinal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
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
    return (
        2
        * energy
        / longitudinal_damping_partition_number
        / energy_loss_per_turn
    )


def calculate_damping_times_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> NumpyArray:
    """
    Calculates the damping times in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object
    revolution_frequency
        Revolution frequency, in [Hz]

    Returns
    -------
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
    return np.array(
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


def calculate_horizontal_damping_time_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculates the horizontal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object
    revolution_frequency
        Revolution frequency [Hz]

    Returns
    -------
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
    return (
        2
        * energy
        / horizontal_damping_partition_number
        / energy_loss_per_turn
        / revolution_frequency
    )


def calculate_longitudinal_damping_time_in_seconds(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculates the longitudinal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object
    revolution_frequency
        Revolution frequency [Hz]

    Returns
    -------
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
    return (
        2
        * energy
        / longitudinal_damping_partition_number
        / energy_loss_per_turn
        / revolution_frequency
    )


def calculate_energy_loss_per_turn(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Computes the energy loss per turn from synchrotron radiation.

    Parameters
    ----------
    energy
        Total energy, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
        Energy loss per turn, in [eV per turn]
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
    Computes the natural horizontal emittance from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
        Natural horizontal emittance, in [m rad]
    """
    jx = calculate_horizontal_damping_partition_number(
        synchrotron_radiation_integrals,
    )
    return (
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[4]
        / jx
        / synchrotron_radiation_integrals[1]
    )


def calculate_natural_energy_spread(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Computes the natural energy spread from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    particle_type
        ParticleType class object

    Returns
    -------
        Natural energy spread, [dimensionless]
    """
    jz = calculate_longitudinal_damping_partition_number(
        synchrotron_radiation_integrals,
    )
    return np.sqrt(
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[2]
        / (jz * synchrotron_radiation_integrals[1])
    )


def calculate_natural_bunch_length(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    angular_synchrotron_frequency: (float | NumpyArray),
    momentum_compaction_factor: float | NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Computes the natural bunch length from the total energy.

    Parameters
    ----------
    energy
        Total energy, in [eV]
    synchrotron_radiation_integrals
        Synchrotron radiation integrals
    angular_synchrotron_frequency
        Angular synchrotron frequency, in  [rad]
    momentum_compaction_factor
        Momentum compaction factor, [dimensionless]
    particle_type
        ParticleType class object

    Returns
    -------
        Natural bunch length, in [m]
    """
    natural_energy_spread = calculate_natural_energy_spread(
        particle_type=particle_type,
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
    )
    return (
        momentum_compaction_factor
        * c
        / angular_synchrotron_frequency
        * natural_energy_spread
    )
