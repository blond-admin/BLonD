from __future__ import annotations
import numpy as np
from scipy.constants import c
from _core.beam.particle_types import ParticleType, electron
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def calculate_natural_horizontal_emittance(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType = electron,
):
    """
    Function to calculate the natural horizontal emittance.
    :param particle_type: ParticleType object. Electron by default.
    :param energy: float | NumpyArray. Beam energy
    :param synchrotron_radiation_integrals: synchrotron radiation integrals
    """
    jx = calculate_partition_numbers(synchrotron_radiation_integrals, which_plane="h")
    return (
        particle_type.quantum_radiation_constant()
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[4]
        / jx
        / synchrotron_radiation_integrals[1]
    )


def calculate_natural_energy_spread(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType = electron,
):
    """
    Function to calculate the natural energy spread.
    :param particle_type: ParticleType object. Electron by default.
    :param energy: float | NumpyArray. Beam energy
    :param synchrotron_radiation_integrals: synchrotron radiation integrals
    """
    jz = calculate_partition_numbers(synchrotron_radiation_integrals, which_plane="z")
    return np.sqrt(
        particle_type.quantum_radiation_constant()()
        * (energy / particle_type.mass) ** 2.0
        * synchrotron_radiation_integrals[2]
        / (jz * synchrotron_radiation_integrals[1])
    )


def calculate_natural_bunch_length(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    angular_synchrotron_frequency: (float | NumpyArray),
    momentum_compaction_factor: float | NumpyArray,
    particle_type: ParticleType = electron,
):
    """
    Function to compute the natural bunch length from the natural energy
    spread.
    :param energy: beam energy
    :param synchrotron_radiation_integrals: synchrotron radiation integrals
    :param angular_synchrotron_frequency: can be calculated using
    longitudinal_beam_dynamics.get_synchrotron_frequency(---) function
    :param momentum_compaction_factor: momentum compaction factor
    :param particle_type: ParticleType object. Electron by default.
    :return:
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


def calculate_partition_numbers(
    synchrotron_radiation_integrals: NumpyArray,
    which_plane: str = ("z" or "h" or "horizontal" or "longitudinal"),
):
    """
    Function to compute the damping partition numbers in horizontal,
    longitudinal or all planes.
    :param synchrotron_radiation_integrals: synchrotron radiation integrals
    :param which_plane: str input to request the relevant damping partition
    number. If none is provided, all three numbers are returned.
    :return:
    """
    if (which_plane == "horizontal") or (which_plane == "h"):
        return (
            1 - synchrotron_radiation_integrals[3] / synchrotron_radiation_integrals[1]
        )
    elif (which_plane == "longitudinal") or (which_plane == "z"):
        return (
            2 + synchrotron_radiation_integrals[3] / synchrotron_radiation_integrals[1]
        )
    else:
        jx = 1 - synchrotron_radiation_integrals[3] / synchrotron_radiation_integrals[1]
        jy = 1
        jz = 2 + synchrotron_radiation_integrals[3] / synchrotron_radiation_integrals[1]
        return np.array([jx, jy, jz])


def calculate_energy_loss_per_turn(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType = electron,
):
    """
    Function to calculate the expected energy loss per turn due to synchrotron
    radiation
    :param particle_type: ParticleType class object
    :param energy: energy in eV
    :param synchrotron_radiation_integrals: NumpyArray
    :return:
    """
    energy_loss_per_turn = (
        particle_type.quantum_radiation_constant()
        * energy**4
        * synchrotron_radiation_integrals[1]
        / (2 * np.pi)
    )
    return energy_loss_per_turn


def calculate_damping_times_in_second(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    energy_loss_per_turn: float,
    revolution_frequency: float,
    which_plane: str = ("z" or "h" or "horizontal" or "longitudinal"),
):
    """
    Function to calculate the transverse and longitudinal damping times in
    second.
    :param which_plane:
    :param energy: expected beam energy [eV]
    :param synchrotron_radiation_integrals:
    :param energy_loss_per_turn: in eV per turn
    :param revolution_frequency: expected revolution frequency in Hz
    :return:
    """
    if which_plane is not None:
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals, which_plane=which_plane
        )
        return (
            2
            * energy
            / damping_partition_numbers[0]
            / energy_loss_per_turn
            / revolution_frequency
        )
    else:
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals
        )
        tau_x_s = (
            (2 * energy / damping_partition_numbers[0])
            / energy_loss_per_turn
            / revolution_frequency
        )
        tau_y_s = (
            (2 * energy / damping_partition_numbers[1])
            / energy_loss_per_turn
            / revolution_frequency
        )
        tau_z_s = (
            (2 * energy / damping_partition_numbers[2])
            / energy_loss_per_turn
            / revolution_frequency
        )

        return np.array([tau_x_s, tau_y_s, tau_z_s])


def calculate_damping_times_in_turn(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    energy_loss_per_turn: float,
    which_plane: str = None,
):
    """
    Function to calculate the transverse and longitudinal damping times in
    seconds.
    :param which_plane:
    :param energy: expected beam energy [eV]
    :param synchrotron_radiation_integrals:
    :param energy_loss_per_turn: in eV per turn
    :return:
    """
    if which_plane is not None:
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals, which_plane=which_plane
        )
        return 2 * energy / damping_partition_numbers[0] / energy_loss_per_turn
    else:
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals
        )
        tau_x_s = (2 * energy / damping_partition_numbers[0]) / energy_loss_per_turn
        tau_y_s = (2 * energy / damping_partition_numbers[1]) / energy_loss_per_turn
        tau_z_s = (2 * energy / damping_partition_numbers[2]) / energy_loss_per_turn
        return np.array([tau_x_s, tau_y_s, tau_z_s])


def gather_longitudinal_synchrotron_radiation_parameters(
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
    particle_type: ParticleType = electron,
):
    """
    Function to calculate the relevant synchrotron radiation parameters for
    additional calculations or comparison.
    :param particle_type: ParticleType object. Electron by default.
    :param energy: beam energy
    :param synchrotron_radiation_integrals: synchrotron radiation integrals
    :return:
    """
    energy_lost_from_synchrotron_radiation = calculate_energy_loss_per_turn(
        particle_type=particle_type,
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
    )
    longitudinal_damping_time = calculate_damping_times_in_turn(
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
        energy_loss_per_turn=energy_lost_from_synchrotron_radiation,
        which_plane="z",
    )
    natural_energy_spread = calculate_natural_energy_spread(
        particle_type=particle_type,
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
    )
    return (
        energy_lost_from_synchrotron_radiation,
        longitudinal_damping_time,
        natural_energy_spread,
    )
