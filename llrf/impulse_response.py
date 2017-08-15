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

# Set up logging
import logging
logger = logging.getLogger(__name__)



def rectangle(t, tau):
    r"""Rectangular function of time
    
    .. math:: \mathsf{rect} \left( \frac{t}{\tau} \right) = 
        \begin{cases}
            1 \, , \, t \in (-\tau/2, \tau/2) \\
            0.5 \, , \, t = \pm \tau/2 \\
            0 \, , \, \textsf{otherwise}
        \end{cases}
        
    Parameters
    ----------
    t : float array
        Time array
    tau : float
        Time window of rectangular function
        
    Returns
    -------
    float array
        Rectangular function for given time array
        
    """
    
    dt = t[1] - t[0]
    limits = np.where((np.fabs(t - tau/2) < dt/2) | 
                      (np.fabs(t + tau/2) < dt/2))[0]
    logger.debug("In rectangle(), number of limiting indices is %d" 
                 %(len(limits)))
    print(limits)
    if len(limits) != 2:
        raise RuntimeError("ERROR in impulse_response.rectangle(): time" +
                           " array not in correct range!")
    y = np.zeros(len(t))
    y[limits] = 0.5
    y[limits[0]+1:limits[1]] = np.ones(limits[1] - limits[0] - 1)

    return y



def triangle(t, tau):
    r"""Triangular function of time
    
    .. math:: \mathsf{tri} \left( \frac{t}{\tau} \right) = 
        \begin{cases}
            1 - t/\tau\, , \, t \in (0, \tau) \\
            0.5 \, , \, t = 0 \\
            0 \, , \, \textsf{otherwise}
        \end{cases}
        
    Parameters
    ----------
    t : float array
        Time array
    tau : float
        Time window of rectangular function
        
    Returns
    -------
    float array
        Rectangular function for given time array
        
    """
    
    dt = t[1] - t[0]
    limits = np.where(np.fabs(t) < dt/2)[0]
    logger.debug("In triangle(), number of limiting indices is %d" 
                 %(len(limits)))
    print(limits)
    if len(limits) != 1:
        raise RuntimeError("ERROR in impulse_response.triangle(): time" +
                           " array not in correct range!")
    y = np.zeros(len(t))
    y[limits[0]] = 0.5
    y[limits[0]+1:] = 1 - t[limits[0]+1:]/tau
    y[np.where(y < 0)[0]] = 0

    return y



class TravellingWaveCavity(object):
    r"""Impulse responses of a travelling wave cavity. The induced voltage 
    :math:`V(t)` from the impulse response :math:`h(t)` and the I,Q (cavity or
    generator) current :math:`I(t)` can be written in matrix form,
    
    .. math:: 
        \left( \begin{matrix} V_I(t) \\ 
        V_Q(t) \end{matrix} \right)
        = \left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \end{matrix} \right)
        * \left( \begin{matrix} I_I(t) \\ 
        I_Q(t) \end{matrix} \right) \, ,
        
    where :math:`*` denotes convolution, 
    :math:`h(t)*x(t) = \int d\tau h(\tau)x(t-\tau)`. For the **cavity-to-beam 
    induced voltage**,
    
    .. math::
        h_s(t) &= \frac{\rho l^2}{8 \tau}\left( 1 - \frac{t}{\tau} \right) \cos((\omega_c - \omega_r)t) \, , \\
        h_c(t) &= \frac{\rho l^2}{8 \tau}\left( 1 - \frac{t}{\tau} \right) \sin((\omega_c - \omega_r)t) \, ,
        
    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time, and :math:`\omega_r` the central 
    frequency of the cavity; :math:`\omega_c` is the carrier frequency of the 
    I,Q demodulated current signal. On the carrier frequency, 
    :math:`\omega_c = \omega_r`,
    
    .. math::
        h_s(t) &= \frac{\rho l^2}{8 \tau}\left( 1 - \frac{t}{\tau} \right) \\
        h_c(t) &= 0 \, .
     
    
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
    omega_r : flaot
        Central (resonance) revolution frequency [1/s] of the cavity
        
    Attributes
    ----------
    l_cav : float
        Length [m] of the interaction region
    tau : float
        Cavity filling time [s]
        
    """
        
    def __init__(self, l_cell, N_cells, rho, v_g, omega_r):
        
        self.l_cell = float(l_cell)
        self.N_cells = int(N_cells)
        self.rho = float(rho)
        if v_g > 0 and v_g < 1:
            self.v_g = float(v_g)
        else:
            raise RuntimeError("ERROR in TravellingWaveCavity: group" +
                " velocity out of limits (0,1)!")
        self.omega_r = float(omega_r)
        
        self.l_cav = float(self.l_cell*self.N_cells)
        self.tau = self.l_cav*c/self.v_g*(1 + self.v_g) # v_g opposite to wave!
        
        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")
        self.logger.debug("Filling time %.4e s", self.tau)
        
        
#    def cavity_to_beam(self, omega_c, time_array):
    def impulse_response(self, omega_c, time_array):
        """Impulse response from the cavity towards the beam and towards the 
        generator. For a signal that is I,Q demodulated at a given carrier 
        frequency :math:`\omega_c`. The formulae assume that the carrier 
        frequency is be close to the central frequency 
        :math:`\omega_c/\omega_r \ll 1` and that the signal is low-pass
        filtered (i.e.\ high-frequency components can be neglected).
        
        Parameters
        ----------
        omega_c : float
            Carrier revolution frequency [1/s]
        time_array : float
            Time array to act on
        """
        
        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_r
        if np.fabs((self.d_omega)/self.omega_r) > 0.1:
            raise RuntimeError("ERROR in TravellingWaveCavity" +
                " impulse_response(): carrier frequency should be close to" +
                " central frequency of the cavity!")
    
        # Time in range (0, tau), otherwise zero
        self.time_array = np.zeros(len(time_array))
        indices = np.where((self.tau - time_array)*(time_array) >= 0)[0]
        self.time_array[indices] = time_array[indices]
        
        # If on carrier frequency
        self.h_s = self.rho*self.l_cav**2/(8*self.tau) \
            *(1 - self.time_array/self.tau)
        self.h_c = None
        
        # If not on carrier frequency
        if np.fabs((self.d_omega)/self.omega_r) > 1e-12:
            self.h_c = np.copy(self.h_s)*np.sin(self.d_omega*
                                                self.time_array)
            self.h_s *= np.cos(self.d_omega*self.time_array)
                

class SPS4Section200MHzTWC(TravellingWaveCavity):
        
    def __init__(self):        
        
        TravellingWaveCavity.__init__(self, 0.374, 43, 2.71e4, 0.0946)
        

class SPS5Section200MHzTWC(TravellingWaveCavity):
        
    def __init__(self):        
        
        TravellingWaveCavity.__init__(self, 0.374, 54, 2.71e4, 0.0946)
    