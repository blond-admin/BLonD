from __future__ import annotations
import numpy as np
from scipy.constants import e
from _core.beam.particle_types import ParticleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def get_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
):
    """
    Calculation of the linear synchrotron angular frequency
    :param energy:
    :param voltage:
    :param synchronous_phase:
    :param slip_factor:
    :return:
    """

    return (
        np.pi
        * revolution_frequency
        * np.sqrt(
            (
                harmonic_number
                * e
                * voltage
                * np.abs(slip_factor * np.cos(synchronous_phase))
            )
            / (2 * np.pi * energy)
        )
    )
