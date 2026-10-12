# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
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

from typing import TYPE_CHECKING

import numpy as np

from blond.generals.function_helpers import raise_on_uneven_array_sizes

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def get_small_amplitude_angular_synchrotron_tune(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    beta: float | NumpyArray,
    charge: float = 1.0,
) -> float | NumpyArray:
    r"""
    Calculation of the small amplitude synchrotron tune.

    Uses the single-harmonic synchrotron tune

    .. math::

        Q_s = \sqrt{\frac{h\,|q|\,V\,|\eta_0 \cos\phi_s|}
                          {2\pi\,\beta^2\,E}}

    where :math:`E` is the total energy in [eV] and :math:`V` the RF voltage in
    [V], so that :math:`|q|\,V` is an energy in [eV] and the expression is
    dimensionless (no elementary-charge factor is required).

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
    beta
        Relativistic beta factor, [dimensionless].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron ``charge=-1``. Default is ``1.0``.

    Returns
    -------
    small_amplitude_angular_synchrotron_tune
        Small amplitude synchrotron tune, [dimensionless].

    See Also
    --------
    blond.acc_math.analytic.hamilton.calc_synchrotron_tune_single_harmonic : Equivalent single-harmonic synchrotron tune.
    """
    raise_on_uneven_array_sizes(
        energy,
        voltage,
        harmonic_number,
        synchronous_phase,
        phase_slip_factor,
    )

    small_amplitude_angular_synchrotron_tune = np.sqrt(
        (
            harmonic_number
            * np.abs(charge)
            * voltage
            * np.abs(phase_slip_factor * np.cos(synchronous_phase))
        )
        / (2 * np.pi * beta**2 * energy)
    )
    return small_amplitude_angular_synchrotron_tune


def get_small_amplitude_angular_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
    beta: float | NumpyArray,
    charge: float = 1.0,
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
    beta
        Relativistic beta factor, [dimensionless].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron ``charge=-1``. Default is ``1.0``.

    Returns
    -------
    small_amplitude_angular_synchrotron_frequency
            Small amplitude angular synchrotron frequency, in [rad/s].
    """
    raise_on_uneven_array_sizes(
        energy,
        voltage,
        harmonic_number,
        synchronous_phase,
        phase_slip_factor,
        revolution_frequency,
    )

    small_amplitude_angular_synchrotron_tune = (
        get_small_amplitude_angular_synchrotron_tune(
            energy=energy,
            voltage=voltage,
            harmonic_number=harmonic_number,
            synchronous_phase=synchronous_phase,
            phase_slip_factor=phase_slip_factor,
            beta=beta,
            charge=charge,
        )
    )

    small_amplitude_angular_synchrotron_frequency = (
        2
        * np.pi
        * revolution_frequency
        * small_amplitude_angular_synchrotron_tune
    )
    return small_amplitude_angular_synchrotron_frequency


def get_angular_synchrotron_tune(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    beta: float | NumpyArray,
    charge: float = 1.0,
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
    beta
        Relativistic beta factor, [dimensionless].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron ``charge=-1``. Default is ``1.0``.

    Returns
    -------
    angular_synchrotron_tune
        Angular synchrotron tune, [dimensionless].
    """
    raise_on_uneven_array_sizes(
        energy,
        voltage,
        harmonic_number,
        synchronous_phase,
        phase_slip_factor,
    )

    small_amplitude_angular_synchrotron_tune = (
        get_small_amplitude_angular_synchrotron_tune(
            energy=energy,
            voltage=voltage,
            harmonic_number=harmonic_number,
            synchronous_phase=synchronous_phase,
            phase_slip_factor=phase_slip_factor,
            beta=beta,
            charge=charge,
        )
    )
    angular_synchrotron_tune = (
        1
        / (2 * np.pi)
        * np.arccos(
            1 - 2 * (np.pi * small_amplitude_angular_synchrotron_tune) ** 2
        )
    )
    return angular_synchrotron_tune


def get_angular_synchrotron_frequency(
    energy: float | NumpyArray,
    voltage: float | NumpyArray,
    harmonic_number: float | NumpyArray,
    synchronous_phase: float | NumpyArray,
    phase_slip_factor: float | NumpyArray,
    revolution_frequency: float | NumpyArray,
    beta: float | NumpyArray,
    charge: float = 1.0,
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
    beta
        Relativistic beta factor, [dimensionless].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron ``charge=-1``. Default is ``1.0``.

    Returns
    -------
    angular_synchrotron_frequency
        Angular synchrotron frequency, in [rad/s].
    """
    raise_on_uneven_array_sizes(
        energy,
        voltage,
        harmonic_number,
        synchronous_phase,
        phase_slip_factor,
        revolution_frequency,
    )

    angular_synchrotron_tune = get_angular_synchrotron_tune(
        energy=energy,
        voltage=voltage,
        harmonic_number=harmonic_number,
        synchronous_phase=synchronous_phase,
        phase_slip_factor=phase_slip_factor,
        beta=beta,
        charge=charge,
    )
    angular_synchrotron_frequency = (
        2 * np.pi * revolution_frequency * angular_synchrotron_tune
    )
    return angular_synchrotron_frequency
