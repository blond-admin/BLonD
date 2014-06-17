'''
Created on 12.06.2014

@author: Danilo Quartullo, Helga Timko
'''


from __future__ import division
import numpy as np
import math
import sys
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e
from trackers.ring_and_RFstation import *


class Kick(object):
    
    """The Kick class represents the kick by a single RF element in a ring!
    The kick (i.e. Delta dp) of the particle's dp coordinate is given by
    the (separable) Hamiltonian derived by z, i.e. the force.

    self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.

    self.phi_offset reflects an offset of the cavity's reference system."""

    def __init__(self, ring, i):
        
        self.harmonic = ring.harmonic[i]
        self.voltage = ring.voltage[i]
        self.phi_offset = ring.phi_offset[i]
        
    def track(self, beam):
       
        beam.dE += self.voltage * np.sin(self.harmonic * beam.theta + 
                                         self.phi_offset) # in eV
    
    
class Kick_acceleration(object):
    
    def __init__(self, ring, p_increment):
        
        self.ring = ring
        self.p_increment = p_increment
        
    def track(self, beam):
        
        """Using the average beta during the acceleration to account for 
        the change in momentum over one turn."""
        beam.dE += - self.ring.beta(beam) * self.p_increment # in eV
        # Update momentum in ring_and_RFstation
        self.ring.counter += 1
        

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
        try: 
            beam.theta = \
            {'full' : self.ring.beta_f(beam) / self.ring.beta_i(beam) * beam.theta \
                    + 2 * np.pi * (1 / (1 - self.ring.eta(beam, beam.delta) \
                    * beam.delta) - 1) * self.ring.length / self.ring.circumference,
             'simple' : beam.theta + 2 * np.pi * self.ring.
                    _eta0(beam, self.ring.alpha_array) * beam.delta * \
                    self.ring.length / self.ring.circumference
            }[self.solver]
        except KeyError:
            print "ERROR: Choice of longitudinal solver not recognized! Aborting..."
            sys.exit()
    
        
class Longitudinal_tracker(object):
    
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
        self.kick_acceleration = Kick_acceleration(ring, 0)
        self.elements = self.kicks + [self.kick_acceleration] + [Drift(ring, solver)]
        
    def track(self, beam):
        
        self.kick_acceleration.p_increment = self.ring.p0_f() - self.ring.p0_i()
        for longMap in self.elements:
            longMap.track(beam)
        
    
class LinearMap(object):
    
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    self.alpha is the linear momentum compaction factor.
    '''

    def __init__(self, ring, Qs):

        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        
        self.circumference = ring.circumference
        self.alpha = ring.alpha_array[0]
        self.Qs = Qs

    def track(self, beam):

        eta = self.alpha - 1 / beam.ring.gamma_i(beam)**2

        omega_0 = 2 * np.pi * beam.ring.beta_i(beam) * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        z0 = beam.z
        delta0 = beam.delta

        beam.z = z0 * cosdQs - eta * c / omega_s * delta0 * sindQs
        beam.delta = delta0 * cosdQs + omega_s / eta / c * z0 * sindQs


