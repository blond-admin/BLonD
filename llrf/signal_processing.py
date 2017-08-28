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


def real_to_cartesian(signal):
    """Convert a real signal to Cartesian (I,Q) coordinates.
    
    Parameters
    ----------
    signal : float array
        Input signal, real array
    
    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    
    """
    
    amplitude = np.max(signal)
    phase = np.arccos(signal / amplitude)
    logger.debug("Creating complex IQ array from real array")
    
    return signal + 1j*amplitude*np.sin(phase)
    
    
def modulator(signal, f_initial, f_final, T_sampling):
    """Demodulate a signal from initial frequency to final frequency. The two
    frequencies should be close.
    
    Parameters
    ----------
    signal : float array
        Signal to be demodulated
    f_initial : float
        Initial frequency [Hz] of signal (before demodulation)
    f_final : float
        Final frequency [Hz] of signal (after demodulation)
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
    delta = 2*np.pi*(f_initial - f_final)*T_sampling
    indices = np.arange(len(signal))
    try:
        I_new = np.cos(delta*indices)*signal.real \
            - np.sin(delta*indices)*signal.imag  
        Q_new = np.sin(delta*indices)*signal.real \
            + np.cos(delta*indices)*signal.imag  
    except:
        raise RuntimeError("ERROR in filters.py/demodulator: signal should" +
                           " be complex!")
        
    return I_new + 1j*Q_new

    
def rf_beam_current(Profile, frequency, T_rev):
    r"""Function calculating the beam current at the (RF) frequency, slice by
    slice. The total charge [C] in the beam is determined from the sum of the 
    beam profile :math:`\lambda_i`, the particle charge :math:`q_p` and the 
    real vs. macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`
    
    .. math:: 
        Q_{\mathsf{tot}} = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \sum_i{\lambda_i}
    
    The DC beam current [A] is the total number of charges per turn :math:`T_0`
    
    .. math:: I_{\mathsf{DC}} = \frac{Q_{\mathsf{tot}}}{T_0}
    
    Parameters
    ----------
    Profile : class
        A Profile type class
    frequency : float
        Revolution frequency [1/s] at which the current should be calculated
    T_rev : float 
        Revolution period [s] of the machine
        
    Returns
    -------
    complex array
        RF beam current array [A] at 'frequency' frequency
        
    """
    
    # Convert real signal to complex IQ
    profile = Profile.n_macroparticles
    
    # Convert from dimensionless to Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    q_m = Profile.Beam.ratio*Profile.Beam.charge*e
    profile *= q_m/T_rev
    logger.debug("Sum of slices: %d, total charge: %.4e C", 
                 np.sum(Profile.n_macroparticles), np.sum(profile)*T_rev)
    logger.debug("DC current is %.4e A", np.sum(profile))
    
    # Mix with frequency of interest
    I_f = profile*np.cos(frequency*Profile.bin_centers)
    Q_f = profile*np.sin(frequency*Profile.bin_centers)
    
    # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
    cutoff = 20.e6*2.*Profile.bin_size
    # Pass through a low-pass filter
    I_filt = low_pass_filter(I_f, cutoff_frequency=cutoff)
    Q_filt = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)))

    return I_filt + 1j*Q_filt


def comb_filter(x, y, a):
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
    

def cavity_filter():
    """Model of the SPS cavity filter.
    """   
    
        
def cavity_impedance():
    """Model of the SPS cavity impedance.
    """


def moving_average(x, N, center=False):
    """Function to calculate the moving average (or running mean) of the input
    data.
    
    Parameters
    ----------
    x : float array
        Data to be smoothed
    N : int
        Window size in points; rounded up to next impair value if center is 
        True
    center : bool    
        Window function centered
        
    Returns
    -------
    float array
        Smoothed data array of has the size 
        * len(x) - N + 1, if center = False
        * len(x), if center = True
        
    """
    
    if center is True:
        # Round up to next impair number
        N_half = int(N/2)
        N = N_half*2 + 1
        # Pad with first and last values
        x = np.concatenate((x[0]*np.ones(N_half), x, x[-1]*np.ones(N_half)))
        
    cumulative_sum = np.cumsum(np.insert(x, 0, 0))
   
    return (cumulative_sum[N:] - cumulative_sum[:-N]) / N



