from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from ...._core.beam.particle_types import ParticleType

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
    Compute the horizontal damping partition number.

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
    Compute the longitudinal damping partition number.

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
    Calculate the damping times in turns.

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
    Calculate the horizontal damping time in turns.

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
    Calculate the longitudinal damping time in turns.

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
    Calculate the damping times in seconds.

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
    Calculate the horizontal damping time in seconds.

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
    Calculate the longitudinal damping time in seconds.

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
    Function to calculate the expected energy loss per turn due to synchrotron
    radiation

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
