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
    
    """The Kick represents the kick(s) by an RF station at a certain position 
    of the ring. The kicks are summed over the different harmonic RF systems 
    in the station.

    The cavity phase can be shifted by the user via phi_offset."""

    def __init__(self, ring, i):
        
        self.harmonic = ring.harmonic[i]
        self.voltage = ring.voltage[i]
        self.phi_offset = ring.phi_offset[i]
        
    def track(self, beam):
       
        beam.dE += self.voltage * np.sin(self.harmonic * beam.theta + 
                                         self.phi_offset) # in eV
    
    
class Kick_acceleration(object):
    
    """Kick_acceleration gives a single accelerating kick to the bunch. 
    The accelerating kick is defined by the change in the design momentum 
    (synchronous momentum). 
    
    The acceleration is assumed to be distributed over the length of the 
    RF station, so the average beta is used in the calculation of the kick."""
    
    def __init__(self, ring, p_increment):
        
        self.ring = ring
        self.p_increment = p_increment
        
    def track(self, beam):
        
        beam.dE += - self.ring.beta(beam) * self.p_increment # in eV
        # Update momentum in ring_and_RFstation
        self.ring.counter += 1
        

class Drift(object):
    
    """The drift updates the longitudinal coordinate of the particle after 
    applying the energy kick; self.length is the drift length.  

    The correction factor \\beta_{n+1} / \\beta_n is necessary when the 
    synchronous energy is low and the range is synchronous energy is large,
    to avoid a shrinking phase space."""

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
        The Longitudinal_tracker tracks the bunch through a given RF station
        and takes care that kicks and the drift are done in correct order.
        
        Different solvers can be used:
        
        'full' -- accurate solution of the drift
        
        'simple' -- drift with no correction for low energy/large energy range and zeroth order in the slippage factor
        
        For de-bunching, simply pass zero voltage.
        For synchrotron radiation, energy loss term yet to be implemented.
    """

    def __init__(self, ring, solver='full'): 
        
        """self.p_increment is the momentum step per turn of the synchronous 
        particle (defined via user input, see ring_and_RFstation).
        See the Kick_acceleration class for further details."""
        
        self.solver = solver
        self.ring = ring
        self.harmonic_list = ring.harmonic
        self.voltage_list = ring.voltage
        self.circumference = ring.circumference
        
        if not len(ring.harmonic) == len(ring.voltage) == len(ring.phi_offset):
            print ("Warning: parameter lists for RFSystems do not have the same length!")


        """Separating the kicks from the RF and the magnets.
        kick can contain multiple contributions:
        self.kicks -- kick due to RF station passage
        self.kick_acceleration -- kick due to acceleration
        self.elements contains the full map of kicks and drift in the RF station."""
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
    Qs is forced to be constant.
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


