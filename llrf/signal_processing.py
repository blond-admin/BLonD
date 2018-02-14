# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Filters and methods for control loops**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
from scipy.constants import e
from scipy import signal as sgn


# Set up logging
import logging
logger = logging.getLogger(__name__)


def polar_to_cartesian(amplitude, phase):
    """Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude : float array
        Amplitude of signal
    phase : float array
        Phase of signal

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    """

    logger.debug("Converting from polar to Cartesian")

    return amplitude*(np.cos(phase) + 1j*np.sin(phase))


def cartesian_to_polar(IQ_vector):
    """Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components

    Returns
    -------
    float array
        Amplitude of signal
    float array
        Phase of signal

    """

    logger.debug("Converting from Cartesian to polar")

    return np.absolute(IQ_vector), np.angle(IQ_vector)


def modulator(signal, omega_i, omega_f, T_sampling):
    """Demodulate a signal from initial frequency to final frequency. The two
    frequencies should be close.

    Parameters
    ----------
    signal : float array
        Signal to be demodulated
    omega_i : float
        Initial revolution frequency [1/s] of signal (before demodulation)
    omega_f : float
        Final revolution frequency [1/s] of signal (after demodulation)
    T_sampling : float
        Sampling period (temporal bin size) [s] of the signal

    Returns
    -------
    float array
        Demodulated signal at f_final

    """

    if len(signal) < 2:
        raise RuntimeError("ERROR in filters.py/demodulator: signal should" +
                           " be an array!")
    delta_phi = (omega_i - omega_f)*T_sampling * np.arange(len(signal))
    # Pre compute sine and cosine for speed up
    cs = np.cos(delta_phi)
    sn = np.sin(delta_phi)
    I_new = cs*signal.real - sn*signal.imag
    Q_new = sn*signal.real + cs*signal.imag

    return I_new + 1j*Q_new


def rf_beam_current(Profile, omega_c, T_rev, lpf=True):
    r"""Function calculating the beam charge at the (RF) frequency, slice by
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

    where :math:`t_i` are the time coordinates of the beam profile. After de-
    modulation, a low-pass filter at 20 MHz is applied.

    Parameters
    ----------
    Profile : class
        A Profile type class
    omega_c : float
        Revolution frequency [1/s] at which the current should be calculated
    T_rev : float
        Revolution period [s] of the machine
    lpf : bool
        Apply low-pass filter; default is True

    Returns
    -------
    complex array
        RF beam charge array [C] at 'frequency' omega_c. To obtain current,
        divide by the sampling time

    """

    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    charges = Profile.Beam.ratio*Profile.Beam.Particle.charge*e\
        * np.copy(Profile.n_macroparticles)
    logger.debug("Sum of particles: %d, total charge: %.4e C",
                 np.sum(Profile.n_macroparticles), np.sum(charges))
    logger.debug("DC current is %.4e A", np.sum(charges)/T_rev)

    # Mix with frequency of interest; remember factor 2 demodulation
    I_f = 2.*charges*np.cos(omega_c*Profile.bin_centers)
    Q_f = 2.*charges*np.sin(omega_c*Profile.bin_centers)

    # Pass through a low-pass filter
    if lpf is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.e6*2.*Profile.bin_size
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f))/T_rev)

    return I_f + 1j*Q_f


def comb_filter(y, x, a):
    """Feedback comb filter.
    """

    return a*y + (1 - a)*x


def low_pass_filter(signal, cutoff_frequency=0.5):
    """Low-pass filter based on Butterworth 5th order digital filter from
    scipy,
    http://docs.scipy.org

    Parameters
    ----------
    signal : float array
        Signal to be filtered
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5

    Returns
    -------
    float array
        Low-pass filtered signal

    """

    b, a = sgn.butter(5, cutoff_frequency, 'low', analog=False)

    return sgn.filtfilt(b, a, signal)


def moving_average(x, N, x_prev=None):
    """Function to calculate the moving average (or running mean) of the input
    data.

    Parameters
    ----------
    x : float array
        Data to be smoothed
    N : int
        Window size in points
    x_prev : float array
        Data to pad with in front

    Returns
    -------
    float array
        Smoothed data array of size
            * len(x) - N + 1, if x_prev = None
            * len(x) + len(x_prev) - N + 1, if x_prev given

    """

    if x_prev is not None:
        # Pad in front with x_prev signal
        x = np.concatenate((x_prev, x))

    # based on https://stackoverflow.com/a/14314054
    mov_avg = np.cumsum(x)
    mov_avg[N:] = mov_avg[N:] - mov_avg[:-N]
    return mov_avg[N-1:] / N
