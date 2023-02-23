# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Fitting and filters routines to be used alone or with the Profile class in
    the beam package. **

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**
'''

import numpy as np
from scipy.signal import cheb2ord, cheby2, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ..utils import bmath as bm

def beam_profile_filter_chebyshev(Y_array, X_array, filter_option):
    """
    This routine is filtering the beam profile with a type II Chebyshev
    filter. The input is a library having the following structure and
    informations:

    filter_option = {'type':'chebyshev', 'pass_frequency':pass_frequency,
    'stop_frequency':stop_frequency, 'gain_pass':gain_pass,
    'gain_stop':gain_stop}

    The function returns nCoefficients, the number of coefficients used
    in the filter. You can also add the following option to plot and return
    the filter transfer function:

    filter_option = {..., 'transfer_function_plot':True}
    """

    noisyProfile = np.array(Y_array)

    freqSampling = 1 / (X_array[1] - X_array[0])
    nyqFreq = freqSampling / 2.

    frequencyPass = filter_option['pass_frequency'] / nyqFreq
    frequencyStop = filter_option['stop_frequency'] / nyqFreq
    gainPass = filter_option['gain_pass']
    gainStop = filter_option['gain_stop']

    # Compute the lowest order for a Chebyshev Type II digital filter
    nCoefficients, wn = cheb2ord(frequencyPass, frequencyStop, gainPass,
                                 gainStop)

    # Compute the coefficients a Chebyshev Type II digital filter
    b, a = cheby2(nCoefficients, gainStop, wn, btype='low')

    # Apply the filter forward and backwards to cancel the group delay
    Y_array = filtfilt(b, a, noisyProfile)
    Y_array = np.ascontiguousarray(Y_array)

    if (('transfer_function_plot' in filter_option)
            and filter_option['transfer_function_plot']):
        # Plot the filter transfer function
        w, transferGain = freqz(b, a=a, worN=len(Y_array))
        transferFreq = w / np.pi * nyqFreq
        group_delay = -np.diff(-np.unwrap(-np.angle(transferGain))) / \
                      -np.diff(w*freqSampling)

        plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(transferFreq, 20 * np.log10(abs(transferGain)))
        plt.ylabel('Magnitude [dB]')
        plt.subplot(312, sharex=ax1)
        plt.plot(transferFreq, np.unwrap(-np.angle(transferGain)))
        plt.ylabel('Phase [rad]')
        plt.subplot(313, sharex=ax1)
        plt.plot(transferFreq[:-1], group_delay)
        plt.ylabel('Group delay [s]')
        plt.xlabel('Frequency [Hz]')

        # Plot the bunch spectrum and the filter transfer function
        plt.figure()
        plt.plot(
            np.fft.fftfreq(len(Y_array), X_array[1]-X_array[0]),
            20.*np.log10(np.abs(np.fft.fft(noisyProfile))))
        plt.xlabel('Frequency [Hz]')
        plt.twinx()
        plt.plot(transferFreq, 20 * np.log10(abs(transferGain)), 'r')
        plt.xlim(0, plt.xlim()[1])

        plt.show()

        return Y_array

    else:

        return Y_array


def gaussian_fit(Y_array, X_array, p0):
    """
    Gaussian fit of the profile, in order to get the bunch length and
    position. Returns fit values in units of s.
    """
    if bm.device == 'GPU':
        X_array = X_array.get()
        Y_array = Y_array.get()
        
    return curve_fit(gauss, X_array, Y_array, p0)[0]


def gauss(x, *p):
    r"""
    Defined as:

    .. math:: A \, e^{\\frac{\\left(x-x_0\\right)^2}{2\\sigma_x^2}}

    """

    A, x0, sx = p
    return A*np.exp(-(x-x0)**2/2./sx**2)


def rms(Y_array, X_array):
    """
    Computation of the RMS bunch length and position from the line
    density (bunch length = 4sigma).
    """

    timeResolution = X_array[1]-X_array[0]

    lineDenNormalized = Y_array / np.trapz(Y_array, dx=timeResolution)

    bp_rms = np.trapz(X_array * lineDenNormalized, dx=timeResolution)

    bl_rms = 4 * np.sqrt(
        np.trapz((X_array-bp_rms)**2 * lineDenNormalized, dx=timeResolution))

    return bp_rms, bl_rms


def fwhm(Y_array, X_array, shift=0):
    """
    Computation of the bunch length and position from the FWHM
    assuming Gaussian line density.
    """

    half_max = shift + 0.5 * (Y_array.max() - shift)

    # First aproximation for the half maximum values
    taux = np.where(Y_array >= half_max)
    t1 = taux[0][0]
    t2 = taux[0][-1]
    # Interpolation of the time where the line density is half the maximum
    bin_size = X_array[1]-X_array[0]
    try:
        t_left = X_array[t1] - bin_size * \
            (Y_array[t1] - half_max) / \
            (Y_array[t1] - Y_array[t1-1])
        t_right = X_array[t2] + bin_size * \
            (Y_array[t2] - half_max) / \
            (Y_array[t2]-Y_array[t2+1])

        bl_fwhm = 4 * (t_right-t_left) / (2 * np.sqrt(2 * np.log(2)))
        bp_fwhm = (t_left+t_right)/2
    except:
        bl_fwhm = np.nan
        bp_fwhm = np.nan

    return bp_fwhm, bl_fwhm


def fwhm_multibunch(Y_array, X_array, n_bunches,
                    bunch_spacing_buckets, bucket_size_tau,
                    bucket_tolerance=0.40, shift=0):
    """
    Computation of the bunch length and position from the FWHM
    assuming Gaussian line density for multibunch case.
    """

    bl_fwhm = np.zeros(n_bunches)
    bp_fwhm = np.zeros(n_bunches)

    if bm.device == 'GPU':
        X_array = X_array.get()
        Y_array = Y_array.get()
    for indexBunch in range(0, n_bunches):

        left_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau -\
            bucket_tolerance * bucket_size_tau
        right_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau +\
            bucket_size_tau + bucket_tolerance * bucket_size_tau
        indexes_bucket = np.where((X_array > left_edge) *
                                  (X_array < right_edge))[0]

        bl_fwhm[indexBunch], bp_fwhm[indexBunch] = fwhm(
            Y_array[indexes_bucket], X_array[indexes_bucket], shift)

    return bl_fwhm, bp_fwhm


def rms_multibunch(Y_array, X_array, n_bunches,
                   bunch_spacing_buckets, bucket_size_tau,
                   bucket_tolerance=0.40):
    """
    Computation of the rms bunch length (4sigma) and position.
    """

    bl_rms = np.zeros(n_bunches)
    bp_rms = np.zeros(n_bunches)

    if bm.device == 'GPU':
        X_array = X_array.get()
        Y_array = Y_array.get()

    for indexBunch in range(0, n_bunches):
        left_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau -\
            bucket_tolerance * bucket_size_tau
        right_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau +\
            bucket_size_tau + bucket_tolerance * bucket_size_tau

        indexes_bucket = np.where((X_array > left_edge) *
                                  (X_array < right_edge))[0]

        bl_rms[indexBunch], bp_rms[indexBunch] = rms(
            Y_array[indexes_bucket],
            X_array[indexes_bucket])

    return bl_rms, bp_rms

