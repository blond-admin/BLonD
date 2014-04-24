from __future__ import division
'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np
import sys


from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e

sin = np.sin
cos = np.cos



class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def hamiltonian():

        return None

    @abstractmethod
    def separatrix():

        return None

    @abstractmethod
    def isin_separatrix():

        return None


class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, frequency, voltage, phi_s, integrator='rk4'):
        '''
        Constructor
        '''        
        dict_integrator = {'rk4': self.integrator_rk4, 'euler-chromer': self.integrator_euler_chromer, 'ruth4': self.integrator_ruth4}
        self.integrator = dict_integrator[integrator]

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.h = frequency
        self.voltage = voltage
        self.phi_s = phi_s

    def eta(self, bunch):

        eta = 1. / self.gamma_transition ** 2 - 1. / bunch.gamma ** 2

        return eta

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        return Qs
    
    def hamiltonian(self, dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        phi = self.h / R * dz + self.phi_s

        H = -1 / 2 * eta * bunch.beta * c * dp ** 2 \
           + e * self.voltage / (p0 * 2 * np.pi * self.h) \
           * (np.cos(phi) - np.cos(self.phi_s) + (phi - self.phi_s) * np.sin(self.phi_s))

        return H

    def separatrix(self, dz, bunch):
        '''
        Separatriox defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''
        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi) 
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) # np.sqrt(e * self.voltage * np.abs(eta) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s 
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        p =  cf1 * (1 + np.cos(phi) + (phi - np.pi) * np.sin(self.phi_s))
        p = np.sqrt(p)

        return p

    def isin_separatrix(self, dz, dp, bunch):

        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) #np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        zmax = np.pi * R / self.h
        pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(pmax)

        return isin

    #~ @profile
    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
         
        cf1 = self.h / R
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)
        
        self.integrator(bunch,p0,R,eta,cf1,cf2)
        
        bunch.update_slices()

    # Definition of integrators
    def integrator_rk4(self,bunch,p0,R,eta,cf1,cf2):
        # Initialize
        dz0 = bunch.dz
        dp0 = bunch.dp

        # Integration
        k1 = -eta * self.length * dp0
        kp1 = cf2 * sin(cf1 * dz0 + self.phi_s)
        k2 = -eta * self.length * (dp0 + kp1 / 2)
        kp2 = cf2 * sin(cf1 * (dz0 + k1 / 2) + self.phi_s)
        k3 = -eta * self.length * (dp0 + kp2 / 2)
        kp3 = cf2 * sin(cf1 * (dz0 + k2 / 2) + self.phi_s)
        k4 = -eta * self.length * (dp0 + kp3);
        kp4 = cf2 * sin(cf1 * (dz0 + k3) + self.phi_s)

        # Finalize
        bunch.dz = dz0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        bunch.dp = dp0 + kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6

    def integrator_euler_chromer(self,bunch,p0,R,eta,cf1,cf2):
        # Length L drift
        bunch.dz += - eta * self.length * bunch.dp
        # Full turn kick
        bunch.dp += cf2 * sin(cf1 * bunch.dz + self.phi_s)

    def integrator_ruth4(self,bunch,p0,R,eta,cf1,cf2):
            alpha = 1 - 2 ** (1 / 3)
            d1 = 1 / (2 * (1 + alpha))
            d2 = alpha / (2 * (1 + alpha))
            d3 = alpha / (2 * (1 + alpha))
            d4 = 1 / (2 * (1 + alpha))
            c1 = 1 / (1 + alpha)
            c2 = (alpha - 1) / (1 + alpha)
            c3 = 1 / (1 + alpha)
            # c4 = 0;

            # Initialize
            dz0 = bunch.dz
            dp0 = bunch.dp

            dz1 = dz0
            dp1 = dp0
            # Drift
            dz1 += d1 * -eta * self.length * dp1
            # Kick
            dp1 += c1 * cf2 * sin(cf1 * dz1 + self.phi_s)

            dz2 = dz1
            dp2 = dp1
            # Drift
            dz2 += d2 * -eta * self.length * dp2
            # Kick
            dp2 += c2 * cf2 * sin(cf1 * dz2 + self.phi_s)

            dz3 = dz2
            dp3 = dp2
            # Drift
            dz3 += d3 * -eta * self.length * dp3
            # Kick
            dp3 += c3 * cf2 * sin(cf1 * dz3 + self.phi_s)

            dz4 = dz3
            dp4 = dp3
            # Drift
            dz4 += d4 * -eta * self.length * dp4

            # Finalize
            bunch.dz = dz4
            bunch.dp = dp4
            

