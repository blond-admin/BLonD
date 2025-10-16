from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import e

from ..._core.beam.particle_types import ParticleType

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def get_linear_angular_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
) -> float | NumpyArray:
    """
    Calculation of the linear synchrotron angular frequency

    Parameters
    ----------
    energy
        Total energy, in [eV]
    voltage
        RF cavity voltage, in [V]
    harmonic_number
        Harmonic number, from the synchronous condition
    synchronous_phase
        Phase of the synchronous particle, in [rad]
    phase_slip_factor
        Phase slip factor, [dimensionless]
    revolution_frequency
        Revolution frequency, in [Hz]

    Returns
    -------
    angular_synchrotron_frequency
            Angular synchrotron frequency, in [rad]
    """
    args = locals()
    keys = inspect.signature(
        get_linear_angular_synchrotron_frequency
    ).parameters.keys()
    input_lengths = {
        len(args.get(arg)) if isinstance(args.get(arg), np.ndarray) else 1
        for arg in keys
    }
    if len(input_lengths) > 2:
        raise ValueError(
            "Input arrays of more than one element have different lengths."
        )

    return (
        np.pi
        * revolution_frequency
        * np.sqrt(
            (
                harmonic_number
                * e
                * voltage
                * np.abs(phase_slip_factor * np.cos(synchronous_phase))
            )
            / (2 * np.pi * energy)
        )
    )
