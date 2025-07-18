from __future__ import annotations
import numpy as np
from scipy.constants import c
from _core.beam.particle_types import ParticleType
from typing import TYPE_CHECKING

from acc_math.analytic.longitudinal_beam_dynamics import \
    get_synchrotron_frequency
from examples.EX_05_Wake_impedance import harmonic_number
from performance_blond3.backends.kick import voltage

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def calculate_natural_horizontal_emittance(particle:ParticleType,
                                    energy : float | NumpyArray,
                                    synchrotron_radiation_integrals : NumpyArray
                                    ):
    jx = calculate_partition_numbers(synchrotron_radiation_integrals,
                                     which_plane='h')
    return (particle.quantum_radiation_constant() * (energy / particle.mass) **
            2.0 * synchrotron_radiation_integrals[
        4]/jx /synchrotron_radiation_integrals[1])

def calculate_natural_energy_spread(particle:ParticleType,
                                    energy : float | NumpyArray,
                                    synchrotron_radiation_integrals : NumpyArray
                                    ):
    jz = calculate_partition_numbers(synchrotron_radiation_integrals,
                                     which_plane = 'z')
    return np.sqrt(particle.quantum_radiation_constant()() * (energy / particle.mass) ** 2.0 *
            synchrotron_radiation_integrals[2] / (jz * synchrotron_radiation_integrals[1]))

def calculate_natural_bunch_length(particle:ParticleType,
                                    energy : float | NumpyArray,
                                    synchrotron_radiation_integrals :
                                    NumpyArray,
                                    angular_synchrotron_frequency:(float
                                                                   |NumpyArray),
                                   momentum_compaction_factor: float|NumpyArray):

    natural_energy_spread = calculate_natural_energy_spread(particle =
                                                            particle,
                                                            energy = energy,
                                                            synchrotron_radiation_integrals=synchrotron_radiation_integrals)
    return momentum_compaction_factor  * c / angular_synchrotron_frequency * natural_energy_spread

def calculate_partition_numbers(synchrotron_radiation_integrals: NumpyArray,
                                which_plane: str = (
                                        'z' or 'h' or
                                        'horizontal' or
                                        'longitudinal')):
    """Function to calculate the damping partition numbers from a set of
    synchrotron radiation integrals"""
    if (which_plane == 'horizontal') or (which_plane == 'h'):
        return 1 - synchrotron_radiation_integrals[3] / \
             synchrotron_radiation_integrals[1]
    elif (which_plane == 'longitudinal') or (which_plane == 'z'):
        return 2 + synchrotron_radiation_integrals[3] / \
             synchrotron_radiation_integrals[1]
    else :
        jx = 1 - synchrotron_radiation_integrals[3] / \
             synchrotron_radiation_integrals[1]
        jy = 1
        jz = 2 + synchrotron_radiation_integrals[3] / \
             synchrotron_radiation_integrals[1]
        return np.array([jx, jy, jz])

def calculate_energy_loss_per_turn(
    particle_type: ParticleType,
    energy: float | NumpyArray,
    synchrotron_radiation_integrals: NumpyArray,
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
        which_plane: str = (
                'z' or 'h' or
                'horizontal' or
                'longitudinal'),
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
            synchrotron_radiation_integrals, which_plane = which_plane
        )
        return 2 * energy / damping_partition_numbers[0]/ energy_loss_per_turn/ revolution_frequency
    else :
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
    which_plane : str = None,
):
    """
    Function to calculate the transverse and longitudinal damping times in
    seconds.
    :param which_plane:
    :param energy: expected beam energy [eV]
    :param synchrotron_radiation_integrals:
    :param energy_loss_per_turn: in eV per turn
    :param revolution_frequency: expected revolution frequency in Hz
    :return:
    """
    if which_plane is not None:
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals, which_plane = which_plane
        )
        return 2 * energy / damping_partition_numbers[0]/ energy_loss_per_turn
    else :
        damping_partition_numbers = calculate_partition_numbers(
            synchrotron_radiation_integrals
        )
        tau_x_s = (
            (2 * energy / damping_partition_numbers[0])
            / energy_loss_per_turn
        )
        tau_y_s = (
            (2 * energy / damping_partition_numbers[1])
            / energy_loss_per_turn
        )
        tau_z_s = (
            (2 * energy / damping_partition_numbers[2])
            / energy_loss_per_turn
        )
        return np.array([tau_x_s, tau_y_s, tau_z_s])

def gather_longitudinal_synchrotron_radiation_parameters(particle:
ParticleType,
                                             energy: float | NumpyArray,
                                            synchrotron_radiation_integrals:
                                            NumpyArray):
    energy_lost_from_synchrotron_radiation \
        = calculate_energy_loss_per_turn(
        particle_type=particle,
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
    )
    longitudinal_damping_time = calculate_damping_times_in_turn(
        energy=energy,
        synchrotron_radiation_integrals=synchrotron_radiation_integrals,
        energy_loss_per_turn=energy_lost_from_synchrotron_radiation,
        which_plane='z',
        )
    natural_energy_spread = calculate_natural_energy_spread(
                                            particle=particle,
                                            energy=energy,
                                            synchrotron_radiation_integrals=synchrotron_radiation_integrals
                                            )
    return (energy_lost_from_synchrotron_radiation,
            longitudinal_damping_time, natural_energy_spread)
