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
from scipy.constants import c
import logging



class TravellingWaveCavity(object):
    """Impulse responses of a travelling wave cavity.
    
    Parameters
    ----------
    l_cell : float
        Cavity cell length [m]
    N_cells : int
        Number of accelerating (interacting) cells in a cavity
    rho : float
        Series impedance [Ohms/m^2] of the cavity
    v_g : float
        Group velocity [c] in units of the speed of light
    omega_0 : flaot
        Central (resonance) revolution frequency [1/s] of the cavity
        
    Attributes
    ----------
    l_cav : float
        Length [m] of the interaction region
    tau : float
        Cavity filling time [s]
        
    """
        
    def __init__(self, l_cell, N_cells, rho, v_g, omega_0):
        
        self.l_cell = float(l_cell)
        self.N_cells = int(N_cells)
        self.rho = float(rho)
        if v_g > 0 and v_g < 1:
            self.v_g = float(v_g)
        else:
            raise RuntimeError("ERROR in TravellingWaveCavity: group" +
                " velocity out of limits (0,1)!")
        self.omega_0 = float(omega_0)
        
        self.l_cav = float(self.l_cell*self.N_cells)
        self.tau = self.l_cav*c/self.v_g*(1 + self.v_g) # v_g opposite to wave!
        
        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")
        self.logger.debug("Filling time %.4e s", self.tau)
        
        
    def cavity_to_beam(self, omega_c, time_array):
        """Impulse response from the cavity towards the beam. For a signal that
        is I,Q demodulated at a given carrier frequency. The approximations
        used in the formulae require the carrier frequency to be close to the
        central frequency and assume that the signal is low-pass filtered.
        
        Parameters
        ----------
        omega_c : float
            Carrier revolution frequency [1/s]
        time_array : float
            Time array to act on
        """
        
        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_0
    
        # Time in range (0, tau)
        self.time_array = np.zeros(len(time_array))
        indices = np.where((self.tau - time_array)*(time_array) >= 0)[0]
        self.time_array[indices] = time_array[indices]
        
        if np.fabs((self.d_omega)/self.omega_0) > 0.1:
            raise RuntimeError("ERROR in TravellingWaveCavity" +
                " cavity_to_beam(): carrier frequency should be close to" +
                " central frequency of the cavity!")
        else:
            
            # If on carrier frequency
            self.h_s = self.rho*self.l_cav**2/(8*self.tau) \
                *(1 - self.time_array/self.tau)
            self.h_c = None
            
            # If not on carrier frequency
            if np.fabs((self.d_omega)/self.omega_0) > 1e-12:
                self.h_c = np.copy(self.h_s)*np.sin(self.d_omega*
                                                    self.time_array)
                self.h_s *= np.cos(self.d_omega*self.time_array)
                

class SPS4Section200MHzTWC(TravellingWaveCavity):
        
    def __init__(self):        
        
        TravellingWaveCavity.__init__(self, 0.374, 43, 2.71e4, 0.0946)
        

class SPS5Section200MHzTWC(TravellingWaveCavity):
        
    def __init__(self):        
        
        TravellingWaveCavity.__init__(self, 0.374, 54, 2.71e4, 0.0946)
    