from __future__ import division
import numpy as np
from scipy.constants import c, e

class Ring_and_RF(object):
    
    def __init__(self, circumference, length, harmonic_list, voltage_list, phi_offset_list, alpha_array, momentum_program_array):
        
        self.circumference = circumference
        self.harmonic = harmonic_list
        self.voltage = voltage_list
        self.phi_offset = phi_offset_list
        self.alpha_array = alpha_array
        self.momentum_program_array = momentum_program_array
        self.length = length
        self.counter = 0
        self.p0_i = momentum_program_array[self.counter]
        self.p0_f = momentum_program_array[self.counter + 1]
        
    

    def beta_i(self, beam):
        return np.sqrt( 1 / (1 + (beam.mass * c)**2 / (self.p0_i * e / c)**2) )
        
    def beta_f(self, beam):
        return np.sqrt( 1 / (1 + (beam.mass * e)**2 / (self.p0_f * e / c)**2) )
        
    def gamma_i(self, beam):
        return np.sqrt( 1 + (self.p0_i * e / c)**2 / (beam.mass * c)**2 )
    
    def gamma_f(self, beam):
        return np.sqrt( 1 + (self.p0_f * e / c)**2 / (beam.mass * c)**2 )
    
    def energy_i(self, beam):
        return np.sqrt( (self.p0_i * e)**2 + (beam.mass * c**2)**2 )
    
    def energy_f(self, beam):
        return np.sqrt( (self.p0_f * e)**2 + (beam.mass * c**2)**2 )
    
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
        To be generalized."""
        h0 = self.harmonic_list[0]
        V0 = self.voltage_list[0]
        c1 = self.eta(delta, beam) * c * np.pi / (self.circumference * 
                                                  beam.beta * beam.energy )
        c2 = c * e * V0 / (h0 * self.circumference)
        phi_s = self.calc_phi_s(beam, self.voltage_list)

        return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                                  (h0 * theta) * np.sin(phi_s))

    def calc_phi_s(self, beam, voltage):
        """The synchronous phase calculated from the rate of momentum change.
        Below transition, for decelerating bucket: phi_s is in (-Pi/2,0)
        Below transition, for accelerating bucket: phi_s is in (0,Pi/2)
        Above transition, for accelerating bucket: phi_s is in (Pi/2,Pi)
        Above transition, for decelerating bucket: phi_s is in (Pi,3Pi/2)
        The synchronous phase is calculated at a certain moment."""
        V0 = voltage[0]
        phi_s = np.arcsin(beam.beta * c / (e * V0) * self.kick_acceleration.p_increment)
        if self.eta(beam, 0) > 0:
            phi_s = np.pi - phi_s

        return phi_s      
    
    def separatrix(self, beam, theta):
        """Single RF sinusoidal separatrix.
        To be generalized."""
        h0 = self.harmonic_list[0]
        V0 = self.voltage_list[0]
        phi_s = self.calc_phi_s(beam, self.voltage_list)

        return np.sqrt(beam.beta**2 * beam.energy * e * V0 / 
                       (np.pi * self.eta(beam, 0) * h0) * 
                       (-np.cos(h0 * theta) - np.cos(phi_s) + 
                         (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))

    def is_in_separatrix(self, beam, theta, dE, delta):
        """Condition for being inside the separatrix.
        Single RF sinusoidal.
        To be generalized."""
        h0 = self.harmonic_list[0]
        phi_s = self.calc_phi_s(beam, self.voltage_list)
        Hsep = self.hamiltonian((np.pi - phi_s) / h0, 0, 0, beam) 
        isin = np.fabs(self.hamiltonian(theta, dE, delta, beam)) < np.fabs(Hsep)

        return isin

    def eta(self, beam, delta):
        
        """Depending on the number of entries in self.alpha_array the 
        according order of \eta = \sum_i \eta_i * \delta^i where
        \delta = \Delta p / p0 will be included in this gathering function.

        Note: Please implement higher slippage factor orders as static methods
        with name _eta<N> where <N> is the order of delta in eta(delta)
        and with signature (alpha_array, beam).
        """
        eta = 0
        for i in xrange( len(self.alpha_array) ):   # order = len - 1
            eta_i = getattr(self, '_eta' + str(i))(beam, self.alpha_array)
            eta  += eta_i * (delta ** i)
        return eta

    @staticmethod
    def _eta0(beam, alpha_array):
        
        return alpha_array[0] - beam.gamma ** -2
   
    @staticmethod
    def _eta1(beam, alpha_array):
        
        return 3 * beam.beta ** 2 / (2 * beam.gamma ** 2) + alpha_array[1] - alpha_array[0] * (alpha_array[0] - beam.gamma ** -2)
    
    @staticmethod
    def _eta2(beam, alpha_array):
        
        return - beam.beta ** 2 * (5 * beam.beta ** 2 - 1) / (2 * beam.gamma ** 2) \
            + alpha_array[2] - 2 * alpha_array[0] * alpha_array[1] + alpha_array[1] / beam.gamma ** 2 \
            + alpha_array[0] ** 2 * (alpha_array[0] - beam.gamma ** -2) - 3 * beam.beta ** 2 * alpha_array[0] / (2 * beam.gamma ** 2)
    
    
