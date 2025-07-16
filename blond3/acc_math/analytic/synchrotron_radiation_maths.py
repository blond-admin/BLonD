from __future__ import annotations
import numpy as np
from _core.beam.particle_types import ParticleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

def calculate_partition_numbers(synchrotron_radiation_integrals: NumpyArray):
    """Function to calculate the damping partition numbers from a set of
    synchrotron radiation integrals"""
    jx = 1 - synchrotron_radiation_integrals[
        3]/synchrotron_radiation_integrals[1]
    jy = 1
    jz = 2 + synchrotron_radiation_integrals[
        3]/synchrotron_radiation_integrals[1]
    return np.array([jx, jy, jz])

def calculate_energy_loss_per_turn(particle_type: ParticleType, energy: float | NumpyArray,
                                   synchrotron_radiation_integrals: NumpyArray):
    """
    Function to calculate the expected energy loss per turn due to synchrotron
    radiation
    :param particle_type: ParticleType class object
    :param energy: energy in eV
    :param synchrotron_radiation_integrals: NumpyArray
    :return:
    """
    energy_loss_per_turn = (particle_type.sands_radiation_constant()* energy
                            **4 * synchrotron_radiation_integrals[1]/ (
                                    2*np.pi))
    return energy_loss_per_turn

def calculate_damping_times_in_seconds(energy: float | NumpyArray,
                               synchrotron_radiation_integrals: NumpyArray,
                               energy_loss_per_turn: float,
                               revolution_frequency: float):
    """
    Function to calculate the transverse and longitudinal damping times in
    seconds.
    :param energy: expected beam energy [eV]
    :param synchrotron_radiation_integrals:
    :param energy_loss_per_turn: in eV per turn
    :param revolution_frequency: expected revolution frequency in Hz
    :return:
    """
    damping_partition_numbers = calculate_partition_numbers(synchrotron_radiation_integrals)
    tau_x_s = (2 * energy / damping_partition_numbers[
    0]) / energy_loss_per_turn / revolution_frequency
    tau_y_s = (2 * energy / damping_partition_numbers[
    1]) / energy_loss_per_turn / revolution_frequency
    tau_z_s = (2 * energy / damping_partition_numbers[
    2]) / energy_loss_per_turn / revolution_frequency

    return np.array([tau_x_s, tau_y_s, tau_z_s])