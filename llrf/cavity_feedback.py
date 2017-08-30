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
from llrf.signal_processing import comb_filter, cartesian_to_polar, polar_to_cartesian
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
    G_tx : float
        Transmitter gain [A/V]; default is :math:`(50 \Omega)^{-1}`
    
    Attributes
    ----------
    TWC_4 : class
        An SPS4Section200MHzTWC type class
    TWC_5 : class
        An SPS5Section200MHzTWC type class
    time : float array
        Time array of impulse responses
    V_tot : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates;
        :math:`V_{\mathsf{tot}}`
    V_tot_prev : complex array
        Cavity voltage [V] of the previous turn in (I,Q) coordinates
    a_comb_filter : float
        Recursion constant of the comb filter; :math:`a_{\mathsf{comb}}=15/16`
    logger : logger
        Logger of the present class
    
    '''
    
    def __init__(self, RFStation, Beam, Profile, G_tx=0.02):

        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile
        self.G_tx = float(G_tx)
        
        # 200 MHz travelling wave cavity impulse responses
        # Make sure that the impulse reponse has the same length as the profile
        self.TWC_4 = SPS4Section200MHzTWC()
        self.TWC_5 = SPS5Section200MHzTWC()
#        self.time = np.copy(self.profile.bin_centers) \
#            - self.profile.bin_centers[0]
        
        # Initialise bunch-by-bunch voltage correction array
        self.V_tot = np.ones(self.profile.n_slices, dtype=float) + \
            1j*np.zeros(self.profile.n_slices, dtype=float)
        
        # Initialise comb filter
        self.V_gen_prev = np.zeros(len(self.V_tot)) #polar_to_cartesian(self.voltage) # ??? WHAT IS THE CORRECT INITIAL VALUE???
        self.a_comb_filter = float(15/16)
        
        # Initialise cavity filter
        #self.cavity_filter_buckets = float(5) # Trev  / 4620 * 5
        
        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")


        
    def track(self):
        """Turn-by-turn tracking method."""
        
        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0,self.counter]
        
        # On current measured (I,Q) voltage, apply LLRF model
        
        # Convert voltage to current, transmitter model
        self.I_gen = self.transmitter(self.V_gen)
        # Generator-induced voltage from generator current
        
        # Beam-induced voltage from beam profile
        
        # Sum and convert to voltage amplitude and phase


    def llrf_model(self):
        """Models the LLRF part of the OTFB.
        
        Attributes
        ----------
        V_set : complex array
            Voltage set point [V] in (I,Q); :math:`V_{\mathsf{set}}`
        V_gen : complex array
            Generator voltage [V] in (I,Q); 
            :math:`V_{\mathsf{gen}} = V_{\mathsf{set}} - V_{\mathsf{tot}}`
        """
        
        # Voltage set point of current turn (I,Q)
        self.V_set = polar_to_cartesian(self.rf.voltage[0,self.counter],
                                        self.rf.phi_rf[0,self.counter])
        # Convert to array
        self.V_set *= np.ones(self.profile.n_slices)
        
        # Difference of set point and actual voltage
        self.V_gen = self.V_set - self.V_tot
        
        # One-turn delay comb filter; memorise the value of the previous turn
        V_tmp = comb_filter(self.V_gen_prev, self.V_gen, self.a_comb_filter)
        self.V_gen_prev = np.copy(self.V_gen)
        self.V_gen = np.copy(V_tmp)
        
        # Modulate from omega_rf to omega_r
        
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


    def transmitter(self, voltage):
        r"""Transmitter model: a simple gain [A/V] converting voltage to
        current.
        
        ..math:: I = G_{\mathsfs{tx}} V
        
        """
        
        current = self.G_tx*voltage
        
        return current


        
    def generator_induced_voltage(self):
        """
        """
        
    def beam_induced_voltage(self, lpf=True):
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
        
        Parameters
        ----------
        lpf : bool
            Apply low-pass filter for beam current calculation; default is True 
            
        Attributes
        ----------
        I_beam : complex array
            RF component of the beam current [A] at the present time step
        Vind_beam : complex array
            Induced voltage [V] from beam-cavity interaction
        
        """
        
        # Beam current from profile
        self.I_beam = rf_beam_current(self.profile, self.TWC_4.omega_r, 
                                      self.rf.t_rev[self.counter], lpf=lpf)
        
        # Calculate impulse response at omega_c
        self.TWC_4.impulse_response(self.omega_c, self.profile.bin_centers)#self.time)
        self.TWC_5.impulse_response(self.omega_c, self.profile.bin_centers)#, self.time)
        
        # Total beam-induced voltage
        if self.TWC_4.hc_beam == None:
            self.V_ind_beam = self.diag_conv(self.I_beam, self.TWC_4.hs_beam)
            self.logger.debug("Diagonal convolution for TWC_4")
        else:
            self.V_ind_beam = self.matr_conv(self.I_beam, self.TWC_4.hs_beam, 
                                            self.TWC_4.hc_beam)
            self.logger.debug("Matrix convolution for TWC_4")
        if self.TWC_5.hc_beam == None:
            self.V_ind_beam += self.diag_conv(self.I_beam, self.TWC_5.hs_beam) 
            self.logger.debug("Diagonal convolution for TWC_5")
        else:
            self.V_ind_beam += self.matr_conv(self.I_beam, self.TWC_5.hs_beam,
                                             self.TWC_5.hc_beam)
            self.logger.debug("Matrix convolution for TWC_5")
        # Cut the proper length and scale; 2 cavities each -> factor 2
        self.V_ind_beam = -2*self.V_ind_beam[:self.profile.n_slices]


    def diag_conv(self, I, hs):
        """Convolution of beam current with impulse response; diagonal
        components only."""
        
        #return ( self.call_conv(I.real, hs) + 1j*self.call_conv(I.imag, hs) )
        return ( np.convolve(I.real, hs, mode='full') \
                 + 1j*np.convolve(I.imag, hs, mode='full') )
        
        
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



