# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Various cavity loops for the CERN machines**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
from llrf.filters import comb_filter, cartesian_to_polar, polar_to_cartesian, \
    cavity_filter, cavity_impedance


class SPSOneTurnFeedback(object): 
    '''
    Voltage feedback around the cavity.
    '''
    
    def __init__(self, RFSectionParameters, Beam, Slices):

        self.rf_params = RFSectionParameters
        self.beam = Beam
        self.slices = Slices
        
        # Initialise bunch-by-bunch voltage correction array
        self.voltage = np.ones(self.slices.n_slices, dtype=float) + \
            1j*np.zeros(self.slices.n_slices, dtype=float)
        
        # Initialise comb filter
        self.a_comb_filter = float(15/16)
        self.voltage_IQ_prev = polar_to_cartesian(self.voltage) # ??? WHAT IS THE CORRECT INITIAL VALUE???
        
        # Initialise cavity filter
        self.cavity_filter_buckets = float(5) # Trev  / 4620 * 5
        
        
    def track(self):
        
        # Move from polar to cartesian coordinates
        self.voltage_IQ = polar_to_cartesian(self.voltage)
        # Apply comb filter
        self.voltage_IQ = comb_filter(self.voltage_IQ_prev, self.a_comb_filter, 
                                      self.voltage_IQ)
        # Memorize previous voltage ???HERE OR AT THE END OF TRACK???
        self.voltage_IQ_prev = np.copy(self.voltage_IQ)

        
        # Apply cavity filter
        cavity_filter()
        # Apply cavity impedance
        cavity_impedance()
        
        # Go back to polar coordinates
        self.voltage = cartesian_to_polar(self.voltage_IQ)







        