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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy
from numpy import random as rnd
from scipy.constants import e

from blond.core.beam.base import BeamBaseClass
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.hashing_ import hash_linspace

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


@dataclass
class CoarseGridSegments:
    """
    Mapping of the fine profile grid onto the coarse grid.

    Coarse sample ``targets[k]`` collects the fine bins
    ``starts[k]`` up to ``starts[k + 1]``, the last one ending at
    :attr:`end`.

    Attributes
    ----------
    starts
        First fine bin of each coarse sample.
    end
        One past the last fine bin that contributes to any coarse sample.
    targets
        Coarse-grid index each segment is accumulated into.
    """

    starts: NumpyArray
    end: int
    targets: NumpyArray


def demodulation_vector(prof_time: NumpyArray, omega_c: float) -> NumpyArray:
    r"""
    Demodulation of the beam charge at the carrier frequency.

    Multiplying the charge per fine bin with this vector gives the
    complex RF beam charge, :math:`2 Q_i e^{-i \omega_c t_i}`, i.e. the
    factor two of the demodulation is included. The external reference
    rotation is *not*, so that the vector depends only on the profile
    grid and the carrier frequency and can be reused between turns.

    Parameters
    ----------
    prof_time
        Time coordinates [s] of the profile bins.
    omega_c
        Carrier frequency [1/s] to demodulate at.

    Returns
    -------
    demodulation
        Complex demodulation factor per fine bin.
    """
    angle = omega_c * prof_time
    return 2.0 * np.cos(angle) - 2.0j * np.sin(angle)


def coarse_grid_segments(
    prof_time: NumpyArray,
    omega_c: float,
    sampling_time: float,
    dT: float,
) -> CoarseGridSegments:
    """
    Find which fine bins belong to which coarse sample.

    Parameters
    ----------
    prof_time
        Time coordinates [s] of the profile bins.
    omega_c
        Carrier frequency [1/s] the coarse grid is centred on.
    sampling_time
        Sampling time [s] of the coarse grid.
    dT
        Shift [s] in time due to shifting reference frames.

    Returns
    -------
    segments
        The fine-to-coarse mapping.
    """
    ind_fine = np.round(
        (prof_time + dT - np.pi / omega_c) / sampling_time
    ).astype(int)
    # A coarse sample ends wherever the coarse index steps by one.
    indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

    starts = np.empty(len(indices), dtype=int)
    starts[0] = 0
    starts[1:] = indices[:-1]

    return CoarseGridSegments(
        starts=starts,
        end=int(indices[-1]),
        targets=ind_fine[0] + np.arange(len(indices)),
    )


def downsample_rf_beam_charge(
    charges_fine: NumpyArray,
    segments: CoarseGridSegments,
    n_points: int,
) -> NumpyArray:
    """
    Sum the RF beam charge of each coarse sample.

    Parameters
    ----------
    charges_fine
        RF beam charge [C] per fine bin.
    segments
        Fine-to-coarse mapping from :func:`coarse_grid_segments`.
    n_points
        Number of points of the coarse grid.

    Returns
    -------
    charges_coarse
        RF beam charge [C] per coarse sample.
    """
    charges_coarse = np.zeros(n_points, dtype=complex)
    charges_coarse[segments.targets] = np.add.reduceat(
        charges_fine[: segments.end], segments.starts
    )
    return charges_coarse


class RFBeamCurrentCache:
    """
    Grid-dependent quantities of :func:`rf_beam_current`, kept between turns.

    The demodulation vector and the fine-to-coarse mapping depend on the
    profile grid, the carrier frequency and the coarse sampling, but not
    on the beam. Recomputing them costs more than the rest of the beam
    current, so they are recomputed only when one of those changes.

    The grid is compared with :func:`~blond.generals.hashing_.hash_linspace`,
    which samples a few elements rather than the whole array.
    """

    def __init__(self):
        self._demodulation_key: int | None = None
        self._demodulation: NumpyArray | None = None
        self._segments_key: int | None = None
        self._segments: CoarseGridSegments | None = None

    def demodulation(
        self, prof_time: NumpyArray, omega_c: float
    ) -> NumpyArray:
        """
        Return the demodulation vector for this grid and carrier.

        Parameters
        ----------
        prof_time
            Time coordinates [s] of the profile bins.
        omega_c
            Carrier frequency [1/s] to demodulate at.

        Returns
        -------
        demodulation
            Complex demodulation factor per fine bin.
        """
        key = hash_linspace(prof_time, salt=omega_c)
        if key != self._demodulation_key:
            self._demodulation = demodulation_vector(prof_time, omega_c)
            self._demodulation_key = key
        return self._demodulation

    def segments(
        self,
        prof_time: NumpyArray,
        omega_c: float,
        sampling_time: float,
        dT: float,
    ) -> CoarseGridSegments:
        """
        Return the fine-to-coarse mapping for this grid.

        Parameters
        ----------
        prof_time
            Time coordinates [s] of the profile bins.
        omega_c
            Carrier frequency [1/s] the coarse grid is centred on.
        sampling_time
            Sampling time [s] of the coarse grid.
        dT
            Shift [s] in time due to shifting reference frames.

        Returns
        -------
        segments
            The fine-to-coarse mapping.
        """
        key = hash_linspace(prof_time, salt=(omega_c, sampling_time, dT))
        if key != self._segments_key:
            self._segments = coarse_grid_segments(
                prof_time, omega_c, sampling_time, dT
            )
            self._segments_key = key
        return self._segments


def rf_beam_current(
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    use_lowpass_filter: bool = True,
    downsample: dict | None = None,
    external_reference: bool = True,
    dT: float = 0,
    cache: RFBeamCurrentCache | None = None,
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
    cache
        Grid-dependent quantities reused between turns. Without one, they
        are recomputed on every call.

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
    # The sums are over the whole fine grid, so they are worth skipping
    # when debug logging is off.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Sum of particles: %d, total charge: %.4e C",
            np.sum(profile.hist_y),
            np.sum(charges),
        )
        logger.debug("DC current is %.4e A/s", np.sum(charges))

    if cache is None:
        cache = RFBeamCurrentCache()

    # Mix with frequency of interest; remember factor 2 demodulation
    charges_fine = charges * cache.demodulation(prof_time, omega_c)

    # Pass through a low-pass filter
    if use_lowpass_filter is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        charges_fine = low_pass_filter(
            charges_fine.real, cutoff_frequency=cutoff
        ) + 1j * low_pass_filter(charges_fine.imag, cutoff_frequency=cutoff)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "RF total current is %.4e A/s",
            np.fabs(np.sum(charges_fine.real)),
        )

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

        segments = cache.segments(prof_time, omega_c, T_s, dT)
        charges_coarse = downsample_rf_beam_charge(
            charges_fine, segments, n_points
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
