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
import ctypes
import logging
import numpy as np
from llrf.signal_processing import comb_filter, cartesian_to_polar, polar_to_cartesian, \
    cavity_filter, cavity_impedance
from llrf.signal_processing import rf_beam_current
from llrf.impulse_response import SPS4Section200MHzTWC, SPS5Section200MHzTWC
from setup_cpp import libblond



class SPSOneTurnFeedback(object): 
    '''
    Voltage feedback around the cavity.
    
    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        A Profile type class
    
    Attributes
    ----------
    
    
    '''
    
    def __init__(self, RFStation, Beam, Profile):

        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile
        
        # 200 MHz travelling wave cavity impulse responses
        # Make sure that the impulse reponse has the same length as the profile
        self.TWC_4 = SPS4Section200MHzTWC()
        self.TWC_5 = SPS5Section200MHzTWC()
        self.time = np.copy(self.profile.bin_centers) \
            - self.profile.bin_centers[0]
        
        # Initialise bunch-by-bunch voltage correction array
        self.voltage = np.ones(self.profile.n_slices, dtype=float) + \
            1j*np.zeros(self.profile.n_slices, dtype=float)
        
        # Initialise comb filter
        self.a_comb_filter = float(15/16)
        self.voltage_IQ_prev = np.zeros(len(self.voltage)) #polar_to_cartesian(self.voltage) # ??? WHAT IS THE CORRECT INITIAL VALUE???
        
        # Initialise cavity filter
        self.cavity_filter_buckets = float(5) # Trev  / 4620 * 5
        
        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")


        
    def track(self):
        """Turn-by-turn tracking method."""
        
        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0,self.counter]
        
#         # Move from polar to cartesian coordinates
#         self.voltage_IQ = polar_to_cartesian(self.voltage)
#         self.logger.info("Voltage is %.4e", np.sum(self.voltage))
#         # Apply comb filter
#         self.voltage_IQ = comb_filter(self.voltage_IQ_prev, self.a_comb_filter, 
#                                       self.voltage_IQ)
#         # Memorize previous voltage ???HERE OR AT THE END OF TRACK???
#         self.voltage_IQ_prev = np.copy(self.voltage_IQ)
# 
#         
#         # Apply cavity filter
#         cavity_filter()
#         # Apply cavity impedance
#         cavity_impedance()
#         # Before applying impulse response to beam, make sure that time_array
#         # Starts from zero and corresponds to non-empty slice
#         cavity_to_beam()
#         
#         # Go back to polar coordinates
#         self.voltage = cartesian_to_polar(self.voltage_IQ)


    def beam_induced_voltage(self):
        r"""Calculates the beam-induced voltage from the beam profile, at a
        given carrier frequency and turn. The beam-induced voltage 
        :math:`V(t)` is calculated from the impulse response matrix 
        :math:`h(t)` as follows:
        
        .. math:: 
            \left( \begin{matrix} V_I(t) \\ 
            V_Q(t) \end{matrix} \right)
            = \left( \begin{matrix} h_s(t) & - h_c(t) \\
            h_c(t) & h_s(t) \end{matrix} \right)
            * \left( \begin{matrix} I_I(t) \\ 
            I_Q(t) \end{matrix} \right) \, ,
        
        where :math:`*` denotes convolution,
        :math:`h(t)*x(t) = \int d\tau h(\tau)x(t-\tau)`. If the carrier
        frequency is close to the cavity resonant frequency, :math:`h_c = 0`.
        
        :seealso: :py:class:`llrf.impulse_response.TravellingWaveCavity`
        
        Impulse response is made to be the same length as the beam profile.
        
        Attributes
        ----------
        I_beam : complex array
            RF component of the beam current [A] at the present time step
        Vind_beam : complex array
            Induced voltage [V] from beam-cavity interaction
        
        """
        
        # Beam current from profile
        self.I_beam = rf_beam_current(self.profile, self.TWC_4.omega_r, 
                                      self.rf.t_rev[self.counter])
        
        # Calculate impulse response at omega_c
        self.TWC_4.impulse_response(self.omega_c, self.time)
        self.TWC_5.impulse_response(self.omega_c, self.time)
        
        # Total beam-induced voltage
        if self.TWC_4.hc_beam == None:
            self.Vind_beam = self.diag_conv(self.I_beam, self.TWC_4.hs_beam)
            self.logger.debug("Diagonal convolution for TWC_4")
            
#             print(self.I_beam.real[45:55])
#             print(self.Vind_beam.real[45:55])
#             print(self.call_conv(self.I_beam.real, self.TWC_4.hs_beam)[45:55])
#             print("")
#             print(self.I_beam.imag[45:55])
#             print(self.Vind_beam.imag[45:55])
#             print(self.call_conv(self.I_beam.imag, self.TWC_4.hs_beam)[45:55])
#             print("")
        else:
            self.Vind_beam = self.matr_conv(self.I_beam, self.TWC_4.hs_beam, 
                                            self.TWC_4.hc_beam)
            self.logger.debug("Matrix convolution for TWC_4")
        if self.TWC_5.hc_beam == None:
            self.Vind_beam += self.diag_conv(self.I_beam, self.TWC_5.hs_beam) 
            self.logger.debug("Diagonal convolution for TWC_5")
        else:
            self.Vind_beam += self.matr_conv(self.I_beam, self.TWC_5.hs_beam,
                                             self.TWC_5.hc_beam)
            self.logger.debug("Matrix convolution for TWC_5")
        # Cut the proper length and scale
        self.Vind_beam = -2*self.Vind_beam[:self.profile.n_slices]#*\
            #2#*self.profile.bin_size # 2 cavities each       


    def diag_conv(self, I, hs):
        """Convolution of beam current with impulse response; diagonal
        components only."""
        
        #return ( self.call_conv(I.real, hs) + 1j*self.call_conv(I.imag, hs) )
        return ( np.convolve(I.real, hs, mode='full') + 1j*np.convolve(I.imag, hs, mode='full') )
        
        
    def matr_conv(self, I, hs, hc):
        """Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        #return ( self.call_conv(I.real, hs) - self.call_conv(I.imag, hc) \
        #    + 1j*(self.call_conv(I.real, hc) + self.call_conv(I.imag, hs)) )
        return ( np.convolve(I.real, hs, mode='full') \
                 - np.convolve(I.imag, hc, mode='full') \
                 + 1j*(np.convolve(I.real, hc, mode='full') 
                       + np.convolve(I.imag, hs, mode='full')) )


    def call_conv(self, signal, kernel):
        """Routine to call optimised C++ convolution"""
        
        result = np.zeros(len(kernel) + len(signal) - 1)
        libblond.convolution(signal.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(signal)),
                             kernel.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(kernel)),
                             result.ctypes.data_as(ctypes.c_void_p))
        
        return result



