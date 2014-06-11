from __future__ import division
import numpy as np
import math
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e
from ring_and_RF import *

# class LongitudinalMap(object):
#     
#     """A longitudinal map represents a longitudinal dynamical element 
#     (e.g. a kick or a drift...), i.e. an abstraction of a cavity 
#     of an RF system etc.
#     LongitudinalMap objects can compose a longitudinal one turn map!
#     Definitions of various orders of the slippage factor eta(delta)
#     for delta = (Delta p / p0) should be implemented in this class. 
#     Any derived objects will access self.eta(beam).
#     
#     Note: the momentum compaction factors are defined by the change of radius
#         \Delta R / R0 = \sum_i \\alpha_i * \delta^(i + 1)
#         hence yielding expressions for the higher slippage factor orders
#         \Delta w / w0 = \sum_j  \eta_j  * \delta^(i + 1)
#         (for the revolution frequency w)
#     """
#     __metaclass__ = ABCMeta
# 
#     def __init__(self, ring):
#         
#         """The length of the momentum compaction factor array /alpha_array/
#         defines the order of the slippage factor expansion. """
#         
#         self.ring = ring
# 
#     @abstractmethod
#     def track(self, beam):
#         
#         pass


# class LongitudinalMap(object):
#     
#     """Can map a full turn or a piece of the ring,
#     depending on the definition of the Ring_and_RF class."""
# 
#     __metaclass__ = ABCMeta
# 
#     def __init__(self, ring):
# #    def __init__(self, beam, alpha_array, circumference, length):
#                 
#         """LongitudinalOneTurnMap objects know their circumference: 
#         this is THE ONE place to store the circumference in the simulations!"""
# #        
#         
#         self.circumference = ring.circumference
# 
#     @abstractmethod
#     def track(self, beam):
#         
#         """Contract: advances the longitudinal coordinates 
#         of the beam over a full turn."""
#         pass


class Kick(object):
    
    """The Kick class represents the kick by a single RF element in a ring!
    The kick (i.e. Delta dp) of the particle's dp coordinate is given by
    the (separable) Hamiltonian derived by z, i.e. the force.

    self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.

    self.phi_offset reflects an offset of the cavity's reference system."""

    def __init__(self, ring, i):
        
        #super(Kick, self).__init__(ring)
        self.harmonic = ring.harmonic[i]
        self.voltage = ring.voltage[i]
        self.phi_offset = ring.phi_offset[i]
        
    def track(self, beam):
        
        beam.dE += e * self.voltage * np.sin(self.harmonic * beam.theta + self.phi_offset)


class Kick_acceleration(object):
    
    def __init__(self, p_increment, ring):
        
        self.p_increment = p_increment
        self.ring = ring
        
    def track(self, beam, ):
        
        """Using the average beta during the acceleration to account for 
        the change in momentum over one turn."""
        beam.dE += - (self.ring.beta_i(beam) + self.ring.beta_f(beam)) / 2 * c * self.p_increment
        

class Drift(object):
    
    """the drift (i.e. Delta z) of the particle's z coordinate is given by
    the (separable) Hamiltonian derived by dp (defined by (p - p0) / p0).

    self.length is the drift length,
    self.beta_factor is the change ratio of \\beta_{n+1} / \\beta_n
    which can often be neglected (and be set to one). [Otherwise it may
    continuously be adapted by the user according to Kick.p_increment.]
    """

    def __init__(self, ring, solver):
        
        self.ring = ring
        self.solver = solver
            
        
    def track(self, beam):   
        beam.theta = { 'full' : self.ring.beta_f(beam) / self.ring.beta_i(beam)  * beam.theta \
                      + 2 * np.pi * (1 / (1 - self.ring.eta(beam, beam.delta) * beam.delta) - 1) * self.ring.length / self.ring.circumference,
                      'simple' : 2,
                      'linear' : 1 }[self.solver]
    
        
class RFSystems(object):
    
    """
        With one RFSystems object in the ring layout (with all kicks applied 
        at the same longitudinal position), the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!
    """

    def __init__(self, ring, solver='full'): 
        
        """The first entry in harmonic_list, voltage_list and phi_offset_list
        defines the parameters for the one accelerating Kick object 
        (i.e. the accelerating RF system).

        The length of the momentum compaction factor array alpha_array
        defines the order of the slippage factor expansion. 
        See the LongitudinalMap class for further details.

        self.p_increment is the momentum step per turn of the synchronous 
        particle, it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.
        See the Kick class for further details.
        self.kicks
        self.elements
        self.fundamental_kick
        self.accelerating_kick"""
        
        self.solver = solver
        self.ring = ring
        self.harmonic_list = ring.harmonic
        self.voltage_list = ring.voltage
        self.circumference = ring.circumference
        
        if not len(ring.harmonic) == len(ring.voltage) == len(ring.phi_offset):
            print ("Warning: parameter lists for RFSystems do not have the same length!")


        """Separating the kicks from the RF and the magnets.
        kick can contain multiple contributions
        kick_acceleration is only used once per time step"""
        self.kicks = []
        for i in xrange(len(ring.harmonic)):
            kick = Kick(ring, i)
            self.kicks.append(kick)
        self.kick_acceleration = Kick_acceleration(0, ring)
        self.elements = self.kicks + [self.kick_acceleration] + [Drift(ring, solver)]
        self.turn_number = 0
        
    def track(self, beam):
        
        self.kick_acceleration.p_increment = self.ring.momentum_program_array[self.turn_number+1] - self.ring.momentum_program_array[self.turn_number]
        beam.p0 = self.ring.momentum_program_array[self.turn_number+1]
        for longMap in self.elements:
            longMap.track(beam)
        self.turn_number += 1
        
    
class LinearMap(object):
    
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    self.alpha is the linear momentum compaction factor.
    '''

    def __init__(self, circumference, alpha, Qs):
        
        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        self.circumference = circumference
        self.alpha = alpha
        self.Qs = Qs

    def track(self, beam):

        eta = self.alpha - beam.gamma ** -2

        omega_0 = 2 * np.pi * beam.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        z0 = beam.z
        dp0 = beam.dp

        beam.z = z0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        beam.dp = dp0 * cosdQs + omega_s / eta / c * z0 * sindQs

