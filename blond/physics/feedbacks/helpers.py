# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Helper functions for feedback models.

Notes
-----
Authors:
Birk Emil Karlsen-Baeck
Helga Timko
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy
from numpy import random as rnd
from scipy.constants import e

from blond.core.beam.base import BeamBaseClass
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.physics.profiles import StaticProfile

logger = logging.getLogger(__name__)


def generate_white_noise(n_points: int, seed1=1234, seed2=5678):
    """
    Generate white noise.

    Parameters
    ----------
    n_points
        Number of points to generate the white noise for.
    seed1
        Seed for the generation of the white noise.
    seed2
        Second seed for the generation of the white noise.

    Returns
    -------
    white_noise
        Array containing the generated white noise.
    """
    r1 = rnd.default_rng(seed1)
    r1 = r1.uniform(low=0.0, high=1.0, size=n_points)

    r2 = rnd.default_rng(seed2)
    r2 = r2.uniform(low=0.0, high=1.0, size=n_points)

    return np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))


def low_pass_filter(
    signal: NumpyArray, cutoff_frequency: float = 0.5
) -> NumpyArray:
    """
    Filter a signal using Butterworth 5th order digital low-pass filter.

    Parameters
    ----------
    signal
        Signal to be filtered.
    cutoff_frequency
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5.

    Returns
    -------
    filtered_signal
        Low-pass filtered signal.

    Notes
    -----
    See `scipy`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """
    b, a = scipy.signal.butter(5, cutoff_frequency, "low", analog=False)

    return scipy.signal.filtfilt(b, a, signal)


def rf_beam_current(
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    use_lowpass_filter: bool = True,
    downsample: dict | None = None,
    external_reference: bool = True,
    dT: float = 0,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""
    Calculate the beam charge at the (RF) frequency slice by slice.

    Function calculating the beam charge at the (RF) frequency, slice by
    slice. The charge distribution [C] of the beam is determined from the beam
    profile :math:`\lambda_i`, the particle charge :math:`q_p` and the real vs.
    macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`

    .. math::
        Q_i = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \lambda_i

    The total charge [C] in the beam is then

    .. math::
        Q_{\mathsf{tot}} = \sum_i{Q_i}

    The DC beam current [A] is the total number of charges per turn :math:`T_0`

    .. math:: I_{\mathsf{DC}} = \frac{Q_{\mathsf{tot}}}{T_0}

    The RF beam charge distribution [C] at a revolution frequency
    :math:`\omega_c` is the complex quantity

    .. math::
        \left( \begin{matrix} I_{rf,i} \\
        Q_{rf,i} \end{matrix} \right)
        = 2 Q_i \left( \begin{matrix} \cos(\omega_c t_i) \\
        \sin(\omega_c t_i)\end{matrix} \right) \, ,

    where :math:`t_i` are the time coordinates of the beam profile.
    After demodulation, a low-pass filter at 20 MHz is applied.

    For multi-bunch cases, make sure that the real beam intensity is the total
    number of charges in the ring.

    Parameters
    ----------
    beam
        A beam type object.
    profile
        A Profile type class.
    omega_c
        Revolution frequency [1/s] at which the current should be calculated.
    use_lowpass_filter
        Apply low-pass filter; default is True.
    downsample
        Dictionary containing float value for 'Ts' sampling time and int value
        for 'points'. Will downsample the RF beam charge onto a coarse time
        grid with 'Ts' sampling time and 'points' points.
    external_reference
        Option to include the changing external reference of the time-grid.
    dT
        The shift in time due to shifting reference frames.

    Returns
    -------
    charges_fine
        RF beam charge array [C] at 'frequency' omega_c, with the sampling time
        of the Profile object. To obtain current, divide by the sampling time.
    charges_coarse
        If time_coarse is specified, returns also the RF beam charge array [C]
        on the coarse time grid.
    """
    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    prof_time = copy_to_cpu(profile.hist_x)
    prof_density = copy_to_cpu(profile.hist_y)

    charges = (
        beam.ratio  # FIXME add to beam
        * beam.particle_type.charge
        * e
        * prof_density
    )
    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(profile.hist_y),
        np.sum(charges),
    )
    logger.debug("DC current is %.4e A/s", np.sum(charges))

    # Mix with frequency of interest; remember factor 2 demodulation
    I_f = 2.0 * charges * np.cos(omega_c * prof_time)
    Q_f = -2.0 * charges * np.sin(omega_c * prof_time)

    # Pass through a low-pass filter
    if use_lowpass_filter is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A/s", np.fabs(np.sum(I_f)))

    charges_fine = I_f + 1j * Q_f
    if external_reference:
        # slippage in phase due to a non-integer harmonic number
        dphi = dT * omega_c
        # Total phase correction
        phase = dphi
        charges_fine = charges_fine * np.exp(1j * phase)

    if downsample:
        if not ("Ts" in downsample and "points" in downsample):
            raise RuntimeError(
                "Downsampling input erroneous in rf_beam_current"
            )

        T_s = float(downsample["Ts"])
        n_points = int(downsample["points"])

        # Find which index in fine grid matches index in coarse grid.
        # `IQCavityFeedback.update_rf_variables` lays the coarse grid out as
        # `rf_centers[k] = (k + 0.5 / n_periods_coarse) * T_s + dT`, and it
        # sets `omega_carrier == omega_rf`, so `0.5 / n_periods_coarse * T_s`
        # is exactly `pi / omega_c` and the grid is
        # `k * T_s + pi / omega_c + dT`. Inverting that for the
        # fine->coarse map therefore *subtracts* `dT`; adding it misplaced
        # the beam-loading current by `round(2 * dT / T_s)` buckets.
        ind_fine = np.round((prof_time - dT - np.pi / omega_c) / T_s)
        ind_fine = ind_fine.astype(int)

        # Accumulate every fine bin into the bucket it belongs to. Walking
        # contiguous runs instead needed bookkeeping that went wrong three
        # ways: transitions were detected with `== 1`, so a gap in the
        # filling pattern (which jumps the index by more than one) went
        # unrecorded and every later group was dropped; each run was summed
        # over a half-open window, so it took the previous run's closing bin
        # and lost its own; and the final run was never emitted, because a
        # run was only written out once a later transition closed it.
        # Scattering per bin has none of those failure modes and needs no
        # run bookkeeping at all. The modulo keeps charge that straddles the
        # end of the turn inside the array; `np.bincount` takes no complex
        # weights, so real and imaginary parts are accumulated separately.
        bucket = ind_fine % n_points
        charges_coarse = np.bincount(
            bucket, weights=charges_fine.real, minlength=n_points
        ) + 1j * np.bincount(
            bucket, weights=charges_fine.imag, minlength=n_points
        )

        return charges_fine, charges_coarse

    else:
        return charges_fine


def cartesian_to_polar(
    IQ_vector: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """
    Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector
        Signal with in-phase and quadrature (I,Q) components.

    Returns
    -------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].
    """
    return np.absolute(IQ_vector), np.angle(IQ_vector)


def polar_to_cartesian(
    amplitude: float | NumpyArray,
    phase: float | NumpyArray,
) -> NumpyArray | complex:
    """
    Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].

    Returns
    -------
    cartesian_signal
        Signal with in-phase and quadrature (I,Q) components.
    """
    return amplitude * (np.cos(phase) + 1j * np.sin(phase))
