# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Fitting and filters routines to be used alone or with the Profile class in
the beam package.**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz
from scipy.optimize import curve_fit
from scipy.signal import cheb2ord, cheby2, filtfilt, freqz

from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from cupy.typing import NDArray as CupyArray

    from ..utils.types import FilterExtraOptionsType


@handle_legacy_kwargs
def beam_profile_filter_chebyshev(
    y_array: NumpyArray,
    x_array: NumpyArray,
    filter_option: FilterExtraOptionsType,
) -> NumpyArray:
    """
    This routine is filtering the beam profile with a type II Chebyshev
    filter. The input is a library having the following structure and
    information:

    filter_option = {'type':'chebyshev', 'pass_frequency':pass_frequency,
    'stop_frequency':stop_frequency, 'gain_pass':gain_pass,
    'gain_stop':gain_stop}

    The function returns n_coefficients, the number of coefficients used
    in the filter. You can also add the following option to plot and return
    the filter transfer function:

    filter_option = {..., 'transfer_function_plot':True}
    """

    noisy_profile = np.array(y_array)

    freq_sampling = 1 / float(x_array[1] - x_array[0])
    nyq_freq = freq_sampling / 2.0

    frequency_pass = filter_option["pass_frequency"] / nyq_freq
    frequency_stop = filter_option["stop_frequency"] / nyq_freq
    gain_pass = filter_option["gain_pass"]
    gain_stop = filter_option["gain_stop"]

    # Compute the lowest order for a Chebyshev Type II digital filter
    n_coefficients, wn = cheb2ord(
        frequency_pass, frequency_stop, gain_pass, gain_stop
    )

    # Compute the coefficients a Chebyshev Type II digital filter
    b, a = cheby2(n_coefficients, gain_stop, wn, btype="low")

    # Apply the filter forward and backwards to cancel the group delay
    y_array = filtfilt(b, a, noisy_profile)
    y_array = np.ascontiguousarray(y_array)

    if ("transfer_function_plot" in filter_option) and filter_option[
        "transfer_function_plot"
    ]:
        # Plot the filter transfer function
        w, transferGain = freqz(b, a=a, worN=len(y_array))
        transferFreq = w / np.pi * nyq_freq
        group_delay = -np.diff(-np.unwrap(-np.angle(transferGain))) / -np.diff(
            w * freq_sampling
        )

        plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(transferFreq, 20 * np.log10(abs(transferGain)))
        plt.ylabel("Magnitude [dB]")
        plt.subplot(312, sharex=ax1)
        plt.plot(transferFreq, np.unwrap(-np.angle(transferGain)))
        plt.ylabel("Phase [rad]")
        plt.subplot(313, sharex=ax1)
        plt.plot(transferFreq[:-1], group_delay)
        plt.ylabel("Group delay [s]")
        plt.xlabel("Frequency [Hz]")

        # Plot the bunch spectrum and the filter transfer function
        plt.figure()
        plt.plot(
            np.fft.fftfreq(len(y_array), x_array[1] - x_array[0]),
            20.0 * np.log10(np.abs(np.fft.fft(noisy_profile))),
        )
        plt.xlabel("Frequency [Hz]")
        plt.twinx()
        plt.plot(transferFreq, 20 * np.log10(abs(transferGain)), "r")
        plt.xlim(0, plt.xlim()[1])

        plt.show()

        return y_array

    else:
        return y_array


@handle_legacy_kwargs
def gaussian_fit(
    y_array: NumpyArray | CupyArray,
    x_array: NumpyArray | CupyArray,
    p0: list[float],
) -> NumpyArray:
    """
    Gaussian fit of the profile, in order to get the bunch length and
    position. Returns fit values in units of s.
    """
    if bm.device == "GPU":
        x_array = x_array.get()
        y_array = y_array.get()

    return curve_fit(gauss, x_array, y_array, p0)[0]


def gauss(x: NumpyArray, *p) -> NumpyArray:
    r"""
    Defined as:

    .. math:: A \, e^{\\frac{\\left(x-x_0\\right)^2}{2\\sigma_x^2}}

    """

    A, x0, sx = p
    return A * np.exp(-((x - x0) ** 2) / 2.0 / sx**2)


@handle_legacy_kwargs
def rms(y_array: NumpyArray, x_array: NumpyArray) -> tuple[float, float]:
    """
    Computation of the RMS bunch length and position from the line
    density (bunch length = 4sigma).
    """

    timeResolution = x_array[1] - x_array[0]

    lineDenNormalized = y_array / np.trapezoid(y_array, dx=timeResolution)

    bp_rms = np.trapezoid(x_array * lineDenNormalized, dx=timeResolution)

    bl_rms = 4 * np.sqrt(
        np.trapezoid(
            (x_array - bp_rms) ** 2 * lineDenNormalized, dx=timeResolution
        )
    )

    return bp_rms, bl_rms


@handle_legacy_kwargs
def fwhm(
    y_array: NumpyArray, x_array: NumpyArray, shift: float = 0
) -> tuple[float, float]:
    """
    Computation of the bunch length and position from the FWHM
    assuming Gaussian line density.
    """

    half_max = shift + 0.5 * (y_array.max() - shift)

    # First aproximation for the half maximum values
    taux = np.where(y_array >= half_max)
    t1 = taux[0][0]
    t2 = taux[0][-1]
    # Interpolation of the time, where the line density is half the maximum
    bin_size = x_array[1] - x_array[0]
    try:
        t_left = x_array[t1] - bin_size * (y_array[t1] - half_max) / (
            y_array[t1] - y_array[t1 - 1]
        )
        t_right = x_array[t2] + bin_size * (y_array[t2] - half_max) / (
            y_array[t2] - y_array[t2 + 1]
        )

        bl_fwhm = 4 * (t_right - t_left) / (2 * np.sqrt(2 * np.log(2)))
        bp_fwhm = (t_left + t_right) / 2
    except Exception:
        bl_fwhm = np.nan
        bp_fwhm = np.nan

    return bp_fwhm, bl_fwhm


@handle_legacy_kwargs
def fwhm_multibunch(
    y_array: NumpyArray | CupyArray,
    x_array: NumpyArray | CupyArray,
    n_bunches: int,
    bunch_spacing_buckets: int,
    bucket_size_tau: float,
    bucket_tolerance: float = 0.40,
    shift: float = 0,
) -> tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]:
    """
    Computation of the bunch length and position from the FWHM
    assuming Gaussian line density for multibunch case.
    """

    bl_fwhm = np.zeros(n_bunches)
    bp_fwhm = np.zeros(n_bunches)

    if bm.device == "GPU":
        x_array = x_array.get()
        y_array = y_array.get()
    for indexBunch in range(0, n_bunches):
        left_edge = (
            indexBunch * bunch_spacing_buckets * bucket_size_tau
            - bucket_tolerance * bucket_size_tau
        )
        right_edge = (
            indexBunch * bunch_spacing_buckets * bucket_size_tau
            + bucket_size_tau
            + bucket_tolerance * bucket_size_tau
        )
        indexes_bucket = np.where(
            (x_array > left_edge) * (x_array < right_edge)
        )[0]

        bp_fwhm[indexBunch], bl_fwhm[indexBunch] = fwhm(
            y_array[indexes_bucket], x_array[indexes_bucket], shift
        )

    return bp_fwhm, bl_fwhm


@handle_legacy_kwargs
def rms_multibunch(
    y_array: NumpyArray | CupyArray,
    x_array: NumpyArray | CupyArray,
    n_bunches: int,
    bunch_spacing_buckets: int,
    bucket_size_tau: float,
    bucket_tolerance: float = 0.40,
) -> tuple[float, float]:
    """
    Computation of the rms bunch length (4sigma) and position.
    """

    bl_rms = np.zeros(n_bunches)
    bp_rms = np.zeros(n_bunches)

    if bm.device == "GPU":
        x_array = x_array.get()
        y_array = y_array.get()

    for indexBunch in range(0, n_bunches):
        left_edge = (
            indexBunch * bunch_spacing_buckets * bucket_size_tau
            - bucket_tolerance * bucket_size_tau
        )
        right_edge = (
            indexBunch * bunch_spacing_buckets * bucket_size_tau
            + bucket_size_tau
            + bucket_tolerance * bucket_size_tau
        )

        indexes_bucket = np.where(
            (x_array > left_edge) * (x_array < right_edge)
        )[0]

        bp_rms[indexBunch], bl_rms[indexBunch] = rms(
            y_array[indexes_bucket], x_array[indexes_bucket]
        )

    return bp_rms, bl_rms
