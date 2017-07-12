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
    """Convert data from polar to cartesian (I,Q) coordinates.
    """
    
    return np.absolute(IQ_vector), np.angle(IQ_vector)

    
def rf_beam_current(frequency, profile):
    """Function calculating the beam current at the RF frequency, bucket by
    bucket.
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