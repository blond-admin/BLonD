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



def polar_to_cartesian(amplitude, phase):
    """Convert data from polar to cartesian (I,Q) coordinates.
    """
    
    return amplitude*(np.cos(phase) + 1j*np.sin(phase))


def cartesian_to_polar(IQ_vector):
    """Convert data from cartesian (I,Q) to polar coordinates.
    """
    
    return np.absolute(IQ_vector), np.angle(IQ_vector)


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
        raise RuntimeError("ERROR in filters.py/demodulator: signal should be"+
                           " an array!")
    delta = 2*np.pi*(f_initial - f_final)*T_sampling
    indices = np.arange(len(signal))
    try:
        I_new = np.cos(delta*indices)*signal.real \
            - np.sin(delta*indices)*signal.imag  
        Q_new = np.sin(delta*indices)*signal.real \
            + np.cos(delta*indices)*signal.imag  
    except:
        raise RuntimeError("ERROR in filters.py/demodulator: signal should be"+
                           " complex!")
        
    return I_new + 1j*Q_new

    
def rf_beam_current(frequency, profile):
    """Function calculating the beam current at the (RF) frequency, bucket by
    bucket.
    
    Parameters
    ----------
    frequency : float
        Frequency at which the current should be calculated
    profile : float array
        
    """
    
    
    return current


def comb_filter(x, y, a):
    """Feedback comb filter.
    """
    
    return a*y + (1 - a)*x


def cavity_filter():
    """Model of the SPS cavity filter.
    """   
    
        
def cavity_impedance():
    """Model of the SPS cavity impedance.
    """

def moving_average(x, N, center = False):
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
    
    if center == True:
        # Round up to next impair number
        N_half = int(N/2)
        N = N_half*2 + 1
        # Pad with first and last values
        x = np.concatenate((x[0]*np.ones(N_half), x, x[-1]*np.ones(N_half)))
        
    cumulative_sum = np.cumsum(np.insert(x, 0, 0))
    #print(cumulative_sum) 
   
    return (cumulative_sum[N:] - cumulative_sum[:-N]) / N


