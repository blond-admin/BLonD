'''
Created on 12.06.2014

@author: Danilo Quartullo, Helga Timko
'''

from __future__ import division
import numpy as np
from warnings import filterwarnings

from scipy.constants import c, e

class Ring_and_RFstation(object):
    '''
    Definition of an RF station and part of the ring until the next station
    
    .. image:: https://github.com/jakubroztocil/httpie/raw/master/httpie.png
        :alt: HTTPie compared to cURL
        :width: 835
        :height: 835
        :align: center
    '''
    
    def __init__(self, circumference, momentum_program, alpha_array, length=None, 
                 harmonic_list=None, voltage_list=None, phi_offset_list=None):
        
        # Obligatory input
        self.circumference = circumference # in m
        self.radius = circumference / 2 / np.pi # in m       
        self.momentum_program = momentum_program # in eV
        if alpha_array != None and len(alpha_array) > 3:
            print "WARNING: Slippage factor implemented only till second order. Higher orders in alpha ignored. "
        self.alpha_array = alpha_array
        self.counter = 0 # To step in the momentum program  
                      
        # Optional parameters
        if length != None and circumference != length:
            print "ATTENTION: The total length of RF stations should sum up to the circumference."
        self.length = length # in m              
        self.harmonic = harmonic_list
        self.voltage = voltage_list # in V
        self.phi_offset = phi_offset_list # in rad

    # Derived energy-related properties 
    # Energy and momentum in units of eV   
    def p0_i(self):
        return self.momentum_program[self.counter]
    
    def p0_f(self):
        return self.momentum_program[self.counter + 1]

    def p0(self):
        return (self.p0_i() + self.p0_f()) / 2   
        
    def beta_i(self, beam):
        return np.sqrt( 1 / (1 + (beam.mass * c**2)**2 / (self.p0_i() * e)**2) )
        
    def beta_f(self, beam):
        return np.sqrt( 1 / (1 + (beam.mass * c**2)**2 / (self.p0_f() * e)**2) )
 
    def beta(self, beam):
        return (self.beta_i(beam) + self.beta_f(beam)) / 2
        
    def gamma_i(self, beam):
        return np.sqrt( 1 + (self.p0_i() * e)**2 / (beam.mass * c**2)**2 )
    
    def gamma_f(self, beam):
        return np.sqrt( 1 + (self.p0_f() * e)**2 / (beam.mass * c**2)**2 )
    
    def gamma(self, beam):
        return (self.gamma_i(beam) + self.gamma_f(beam)) / 2
    
    def energy_i(self, beam):
        return np.sqrt( self.p0_i()**2 + (beam.mass * c**2 / e)**2 )
    
    def energy_f(self, beam):
        return np.sqrt( self.p0_f()**2 + (beam.mass * c**2 / e)**2 )

    def energy(self, beam):
        return (self.energy_i(beam) + self.energy_f(beam)) / 2    
 
    
#    def potential(self, z, beam):
        
#        """the potential well of the rf system"""
#        phi_0 = self.accelerating_kick.calc_phi_0(beam)
#        h1 = self.accelerating_kick.harmonic
#        def fetch_potential(kick):
#            phi_0_i = kick.harmonic / h1 * phi_0
#            return kick.potential(z, beam, phi_0_i)
#        potential_list = map(fetch_potential, self.kicks)
#        return sum(potential_list)
    
    def hamiltonian(self, beam, theta, dE, delta):
        """Single RF sinusoidal Hamiltonian.
        Uses beta, energy averaged over the turn.
        To be generalized."""
        h0 = self.harmonic[0]
        V0 = self.voltage[0]
        c1 = self.eta(beam, delta) * c * np.pi / (self.circumference * 
             self.beta_i(beam) * self.energy_i(beam) )
        c2 = c * self.beta_i(beam) * V0 / (h0 * self.circumference)
        phi_s = self.calc_phi_s(beam, self.voltage)

        return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                                   (h0 * theta - phi_s) * np.sin(phi_s))

    def calc_phi_s(self, beam, voltage):
        """The synchronous phase calculated from the rate of momentum change.
        Below transition, for decelerating bucket: phi_s is in (-Pi/2,0)
        Below transition, for accelerating bucket: phi_s is in (0,Pi/2)
        Above transition, for accelerating bucket: phi_s is in (Pi/2,Pi)
        Above transition, for decelerating bucket: phi_s is in (Pi,3Pi/2)
        The synchronous phase is calculated at a certain moment.
        Uses beta, energy averaged over the turn."""
        V0 = voltage[0]
        phi_s = np.arcsin(self.beta(beam) * (self.p0_f() - self.p0_i()) / V0 )
        if self.eta(beam, 0) > 0:
            phi_s = np.pi - phi_s

        return phi_s      
    
    def separatrix(self, beam, theta):
        """Single RF sinusoidal separatrix.
        Uses beta, energy averaged over the turn.
        To be generalized."""
        h0 = self.harmonic[0]
        V0 = self.voltage[0]
        phi_s = self.calc_phi_s(beam, self.voltage)
        
        filterwarnings('ignore')
        
        separatrix_array = np.sqrt(self.beta_i(beam)**2 * self.energy_i(beam) *
                        V0 / (np.pi * self.eta(beam, 0) * h0) * 
                        (-np.cos(h0 * theta) - np.cos(phi_s) + 
                        (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))
        
        filterwarnings('default')
           
        return separatrix_array

    def is_in_separatrix(self, beam, theta, dE, delta):
        """Condition for being inside the separatrix.
        Single RF sinusoidal.
        Uses beta, energy averaged over the turn.
        To be generalized."""
        h0 = self.harmonic[0]
        phi_s = self.calc_phi_s(beam, self.voltage)
        Hsep = self.hamiltonian(beam, (np.pi - phi_s) / h0, 0, 0) 
        isin = np.fabs(self.hamiltonian(beam, theta, dE, delta)) < np.fabs(Hsep)

        return isin

    def eta(self, beam, delta):
        
        """Depending on the number of entries in self.alpha_array the 
        according order of \eta = \sum_i \eta_i * \delta^i where
        \delta = \Delta p / p0 will be included in this gathering function.

        Note: Please implement higher slippage factor orders as static methods
        with name _eta<N> where <N> is the order of delta in eta(delta)
        and with signature (alpha_array, beam).
        
        Use initial momentum (beta_i, gamma_i).
        Update done in longitudinal_tracker.Kick_acceleration.
        """
        eta = 0
        for i in xrange( len(self.alpha_array) ):   # order = len - 1
            eta_i = getattr(self, '_eta' + str(i))(beam, self.alpha_array)
            eta  += eta_i * (delta**i)
        return eta

    
    def _eta0(self, beam, alpha_array):
        
        return alpha_array[0] - self.gamma_i(beam)**-2
   
    
    def _eta1(self, beam, alpha_array):
        
        return 3 * self.beta_i(beam)**2 / (2 * self.gamma_i(beam)**2) + alpha_array[1] \
            - alpha_array[0] * (alpha_array[0] - self.gamma_i(beam)**-2)
    
    
    def _eta2(self, beam, alpha_array):
        
        return - self.beta_i(beam)**2 * (5 * self.beta_i(beam)**2 - 1) / (2 * self.gamma_i(beam)**2) \
            + alpha_array[2] - 2 * alpha_array[0] * alpha_array[1] + alpha_array[1] \
            / self.gamma_i(beam)**2 + alpha_array[0]**2 * (alpha_array[0] - self.gamma_i(beam)**-2) \
            - 3 * self.beta_i(beam)**2 * alpha_array[0] / (2 * self.gamma_i(beam)**2)
    
    
