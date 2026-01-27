# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Collection of longitudinal beam dynamics analytical formulae.

Author:
L. Valle
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import e

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def check_inputs_consistency(arguments: list):
    """
    Check argument length consistency.

    Parameters
    ----------
    arguments
        List of inputs.
    """
    keys = inspect.signature(
        get_small_amplitude_angular_synchrotron_frequency
    ).parameters.keys()
    # Inspection of the input lengths to ensure arrays have the same lengths.
    input_lengths = {
        len(arguments.get(arg))
        if isinstance(arguments.get(arg), np.ndarray)
        else 1
        for arg in keys
    }
    if len(input_lengths.difference({1, max(input_lengths)})) != 0:
        raise ValueError(
            "Input arrays of more than one element have different lengths."
        )


def get_small_amplitude_angular_synchrotron_tune(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculation of the small amplitude synchrotron angular tune.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    voltage
        RF cavity voltage, in [V].
    harmonic_number
        Harmonic number, from the synchronous condition.
    synchronous_phase
        Phase of the synchronous particle, in [rad].
    phase_slip_factor
        Phase slip factor, [dimensionless].

    Returns
    -------
    angular_synchrotron_tune
        Angular synchrotron tune.
    """
    args = locals()
    check_inputs_consistency(arguments=args)

    angular_synchrotron_tune = np.sqrt(
        (
            harmonic_number
            * e
            * voltage
            * np.abs(phase_slip_factor * np.cos(synchronous_phase))
        )
        / (2 * np.pi * energy)
    )
    return angular_synchrotron_tune


def get_small_amplitude_angular_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculation of the small amplitude synchrotron angular frequency.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    voltage
        RF cavity voltage, in [V].
    harmonic_number
        Harmonic number, from the synchronous condition.
    synchronous_phase
        Phase of the synchronous particle, in [rad].
    phase_slip_factor
        Phase slip factor, [dimensionless].
    revolution_frequency
        Revolution frequency, in [Hz].

    Returns
    -------
    angular_synchrotron_frequency
            Angular synchrotron frequency, in [rad].
    """
    args = locals()
    check_inputs_consistency(arguments=args)

    small_amplitude_angular_synchrotron_tune = (
        get_small_amplitude_angular_synchrotron_tune(
            energy=energy,
            voltage=voltage,
            harmonic_number=harmonic_number,
            synchronous_phase=synchronous_phase,
            phase_slip_factor=phase_slip_factor,
        )
    )

    angular_synchrotron_frequency = (
        2
        * np.pi
        * revolution_frequency
        * small_amplitude_angular_synchrotron_tune
    )
    return angular_synchrotron_frequency


def get_angular_synchrotron_tune(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculation of the synchrotron angular tune.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    voltage
        RF cavity voltage, in [V].
    harmonic_number
        Harmonic number, from the synchronous condition.
    synchronous_phase
        Phase of the synchronous particle, in [rad].
    phase_slip_factor
        Phase slip factor, [dimensionless].

    Returns
    -------
    angular_synchrotron_frequency
            Angular synchrotron frequency, in [rad].
    """
    args = locals()
    check_inputs_consistency(arguments=args)

    small_amplitude_angular_synchrotron_tune = (
        get_small_amplitude_angular_synchrotron_tune(
            energy=energy,
            voltage=voltage,
            harmonic_number=harmonic_number,
            synchronous_phase=synchronous_phase,
            phase_slip_factor=phase_slip_factor,
        )
    )
    angular_synchrotron_frequency = (
        1
        / (2 * np.pi)
        * np.arccos(
            1 - 2 * (np.pi * small_amplitude_angular_synchrotron_tune) ** 2
        )
    )
    return angular_synchrotron_frequency


def get_angular_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculation of the synchrotron angular frequency.

    Parameters
    ----------
    energy
        Total energy, in [eV].
    voltage
        RF cavity voltage, in [V].
    harmonic_number
        Harmonic number, from the synchronous condition.
    synchronous_phase
        Phase of the synchronous particle, in [rad].
    phase_slip_factor
        Phase slip factor, [dimensionless].
    revolution_frequency
        Revolution frequency, in [Hz].

    Returns
    -------
    angular_synchrotron_frequency
            Angular synchrotron frequency, in [rad].
    """
    args = locals()
    check_inputs_consistency(arguments=args)

    angular_synchrotron_tune = get_angular_synchrotron_tune(
        energy=energy,
        voltage=voltage,
        harmonic_number=harmonic_number,
        synchronous_phase=synchronous_phase,
        phase_slip_factor=phase_slip_factor,
    )
    angular_synchrotron_frequency = (
        2 * np.pi * revolution_frequency * angular_synchrotron_tune
    )
    return angular_synchrotron_frequency
