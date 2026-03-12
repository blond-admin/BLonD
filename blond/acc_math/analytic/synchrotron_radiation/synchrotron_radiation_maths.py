# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Collection of functions to compute synchrotron radiation related quantities.

Authors:
L. Valle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from blond.core.beam.particle_types import ParticleType
from blond.generals.function_helpers import raise_on_uneven_array_sizes

if TYPE_CHECKING:
    from numpy import ndarray as NumpyArray


def _selective_calculate_partition_numbers(
    radiation_integrals: NumpyArray,
    x: bool = False,
    y: bool = False,
    z: bool = False,
) -> dict[str, NumpyArray]:
    """
    Helper for the calculation of the damping partition numbers.

    Parameters
    ----------
    radiation_integrals
        Synchrotron radiation integrals.
    x
        Enables calculation in the horizontal plane.
    y
        Enables calculation in the vertical plane.
    z
        Enables calculation in the longitudinal plane.

    Returns
    -------
    result
        Dictionary containing the requested damping partition numbers.
    """
    result = {}
    if x:
        result["x"] = 1 - radiation_integrals[3] / radiation_integrals[1]

    if y:
        result["y"] = 1

    if z:
        result["z"] = 2 + radiation_integrals[3] / radiation_integrals[1]

    return result


def _selective_calculate_damping_times_in_turns(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    x: bool = False,
    y: bool = False,
    z: bool = False,
) -> dict[str, NumpyArray]:
    """
    Helper for the calculation of the damping partition numbers.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    x
        Enables calculation in the horizontal plane.
    y
        Enables calculation in the vertical plane.
    z
        Enables calculation in the longitudinal plane.

    Returns
    -------
    result
        Dictionary containing the requested damping partition numbers.
    """
    result = {}

    partitions = _selective_calculate_partition_numbers(
        radiation_integrals, x=x, y=y, z=z
    )

    energy_loss_per_turn = calculate_energy_loss_per_turn(
        energy,
        radiation_integrals,
        particle_type,
    )

    if x:
        jx = partitions["x"]
        result["x"] = np.array(2 * energy / jx / energy_loss_per_turn)

    if y:
        jy = partitions["y"]
        result["y"] = np.array(2 * energy / jy / energy_loss_per_turn)

    if z:
        jz = partitions["z"]
        result["z"] = np.array(2 * energy / jz / energy_loss_per_turn)

    return result


def calculate_partition_numbers(
    radiation_integrals: NumpyArray,
) -> NumpyArray:
    """
    Compute the damping partition numbers.

    Parameters
    ----------
    radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    damping_partition_numbers
        Damping partition numbers in the [horizontal, vertical, longitudinal] order.
    """
    partitions = _selective_calculate_partition_numbers(
        radiation_integrals, x=True, y=True, z=True
    )

    jx = partitions["x"]
    jy = partitions["y"]
    jz = partitions["z"]

    damping_partition_numbers = np.array([jx, jy, jz])
    return damping_partition_numbers


def calculate_horizontal_damping_partition_number(
    radiation_integrals: NumpyArray,
) -> float:
    """
    Compute the horizontal damping partition number.

    Parameters
    ----------
    radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    horizontal_damping_partition_number
        Horizontal damping partition number.
    """
    partitions = _selective_calculate_partition_numbers(
        radiation_integrals, x=True
    )

    horizontal_damping_partition_number = partitions["x"]

    return float(horizontal_damping_partition_number)


def calculate_longitudinal_damping_partition_number(
    radiation_integrals: NumpyArray,
) -> float:
    """
    Compute the longitudinal damping partition number.

    Parameters
    ----------
    radiation_integrals
        Synchrotron radiation integrals.

    Returns
    -------
    longitudinal_damping_partition_number
        Longitudinal damping partition number.
    """
    partitions = _selective_calculate_partition_numbers(
        radiation_integrals, z=True
    )

    longitudinal_damping_partition_number = partitions["z"]

    return float(longitudinal_damping_partition_number)


def calculate_damping_times_in_turns(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> NumpyArray:
    """
    Calculate the damping times in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    damping_times_turn
        Damping times in the [horizontal, vertical, longitudinal] order, in [turn].
    """
    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        x=True,
        y=True,
        z=True,
    )

    damping_times_turn_x = damping_times_turn_dict["x"]
    damping_times_turn_y = damping_times_turn_dict["y"]
    damping_times_turn_z = damping_times_turn_dict["z"]

    damping_times = np.array(
        [damping_times_turn_x, damping_times_turn_y, damping_times_turn_z]
    )
    return damping_times


def calculate_horizontal_damping_time_in_turns(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculate the horizontal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    horizontal_damping_time_turn
        Horizontal damping time, in [turn].
    """
    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        x=True,
    )
    horizontal_damping_time_turn = damping_times_turn_dict["x"]
    return horizontal_damping_time_turn


def calculate_longitudinal_damping_time_in_turns(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Calculate the longitudinal damping time in turns.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.

    Returns
    -------
    longitudinal_damping_time_turn
        Longitudinal damping time, in [turn].
    """
    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        z=True,
    )
    longitudinal_damping_time_turn = damping_times_turn_dict["z"]
    return longitudinal_damping_time_turn


def calculate_damping_times_in_seconds(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> NumpyArray:
    """
    Calculate the damping times in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency, in [Hz].

    Returns
    -------
    damping_times
        Damping times in the [horizontal, vertical,
        longitudinal] order, in [s].

    Examples
    --------
    >>> from blond import Ring, Beam, electron
    >>> ring = Ring(
    ...     circumference=10,
    ...     radiation_integrals=np.array(
    ...         [
    ...             0.646747216157,
    ...             0.0005936549319,
    ...             5.6814536525e-08,
    ...             5.92870407301e-09,
    ...             1.71368060083e-11,
    ...         ]
    ...     ),
    ... )
    >>> beam = Beam(particle_type=electron)
    >>> [tau_x, tau_y, tau_z] = calculate_damping_times_in_seconds(
    ... energy=beam.reference.total_energy,
    ... radiation_integrals = ring.radiation_integrals,
    ... particle_type = beam.particle_type,
    ... revolution_frequency = beam.reference.velocity /ring.circumference)
    """
    raise_on_uneven_array_sizes(energy, revolution_frequency)

    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        x=True,
        y=True,
        z=True,
    )

    damping_times_seconds_x = (
        damping_times_turn_dict["x"] / revolution_frequency
    )
    damping_times_seconds_y = (
        damping_times_turn_dict["y"] / revolution_frequency
    )
    damping_times_seconds_z = (
        damping_times_turn_dict["z"] / revolution_frequency
    )

    damping_times = np.array(
        [
            damping_times_seconds_x,
            damping_times_seconds_y,
            damping_times_seconds_z,
        ]
    )
    return damping_times


def calculate_horizontal_damping_time_in_seconds(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculate the horizontal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency [Hz].

    Returns
    -------
    horizontal_damping_time
        Horizontal damping time, in [s].
    """
    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        x=True,
    )

    horizontal_damping_time = (
        damping_times_turn_dict["x"] / revolution_frequency
    )
    return horizontal_damping_time


def calculate_longitudinal_damping_time_in_seconds(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculate the longitudinal damping time in seconds.

    Parameters
    ----------
    energy
        Energy of the reference particle, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    particle_type
        ParticleType class object.
    revolution_frequency
        Revolution frequency [Hz].

    Returns
    -------
    longitudinal_damping_time
        Longitudinal damping time, in [s].
    """
    damping_times_turn_dict = _selective_calculate_damping_times_in_turns(
        energy=energy,
        radiation_integrals=radiation_integrals,
        particle_type=particle_type,
        z=True,
    )

    longitudinal_damping_time = (
        damping_times_turn_dict["z"] / revolution_frequency
    )
    return longitudinal_damping_time


def calculate_energy_loss_per_turn(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the energy loss per turn from synchrotron radiation.

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
    energy_loss_per_turn
        Energy loss per turn, in [eV per turn].
    """
    energy_loss_per_turn = (
        particle_type.sands_radiation_constant
        * energy**4
        * radiation_integrals[1]
        / (2 * np.pi)
    )
    return energy_loss_per_turn


def calculate_natural_horizontal_emittance(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural horizontal emittance from the total energy.

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
    natural_horizontal_emittance
        Natural horizontal emittance, in [m rad].
    """
    jx = calculate_horizontal_damping_partition_number(
        radiation_integrals,
    )
    natural_horizontal_emittance = (
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * radiation_integrals[4]
        / jx
        / radiation_integrals[1]
    )
    return natural_horizontal_emittance


def calculate_natural_energy_spread(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural energy spread from the total energy.

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
    natural_energy_spread
        Natural energy spread, [dimensionless].
    """
    jz = calculate_longitudinal_damping_partition_number(
        radiation_integrals,
    )
    natural_energy_spread = np.sqrt(
        particle_type.quantum_radiation_constant
        * (energy / particle_type.mass) ** 2.0
        * radiation_integrals[2]
        / (jz * radiation_integrals[1])
    )
    return natural_energy_spread


def calculate_natural_bunch_length(
    energy: float | NumpyArray,
    radiation_integrals: NumpyArray,
    angular_synchrotron_frequency: (float | NumpyArray),
    momentum_compaction_factor: float | NumpyArray,
    particle_type: ParticleType,
) -> float | NumpyArray:
    """
    Compute the natural bunch length from the total energy due to synchrotron radiation damping.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    radiation_integrals
        Synchrotron radiation integrals.
    angular_synchrotron_frequency
        Angular synchrotron frequency, in  [rad/s].
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
        radiation_integrals=radiation_integrals,
    )
    natural_bunch_length = (
        abs(momentum_compaction_factor)
        * c
        / angular_synchrotron_frequency
        * natural_energy_spread
    )
    return natural_bunch_length
