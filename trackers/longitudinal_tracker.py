'''
**Module containing all the elements to track the beam in the longitudinal plane.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Adrian Oeftiger**, **Alexandre Lasheen**
'''

from __future__ import division
import numpy as np
from scipy.constants import c
from scipy.integrate import cumtrapz


class RingAndRFSection(object):
    '''
    *Definition of an RF station and part of the ring until the next station, 
    see figure.*
    
    .. image:: ring_and_RFstation.png
        :align: center
        :width: 600
        :height: 600
        
    *The time step is fixed to be one turn, but the tracking can consist of 
    multiple RingAndRFSection objects. In this case, the user should make sure 
    that the lengths of the stations sum up exactly to the circumference or use
    the FullRingAndRF object in order to let the code pre-process the parameters.
    Each RF station may contain several RF harmonic systems which are considered
    to be in the same location. First, a kick from the cavity voltage(s) is applied, 
    then an accelerating kick in case the momentum program presents variations, 
    and finally a drift kick between stations.*
    '''
        
    def __init__(self, RFSectionParameters, solver = 'full'):
        
        #: *Copy of the counter (from RFSectionParameters)*
        self.counter = RFSectionParameters.counter
        
        ### Import RF section parameters for RF kick
        #: *Copy of length (from RFSectionParameters)*
        self.section_length = RFSectionParameters.section_length
        #: *Copy of length ratio (from RFSectionParameters)*
        self.length_ratio = RFSectionParameters.length_ratio
        #: *Copy of the number of RF systems (from RFSectionParameters)*
        self.n_rf = RFSectionParameters.n_rf
        #: *Copy of harmonic number program (from RFSectionParameters)*
        self.harmonic = RFSectionParameters.harmonic
        #: *Copy of voltage program in [V] (from RFSectionParameters)*
        self.voltage = RFSectionParameters.voltage
        #: *Copy of phi_offset program in [rad] (from RFSectionParameters)*
        self.phi_offset = RFSectionParameters.phi_offset
        #: *Copy of phi_s program in [rad] (from RFSectionParameters)*
        self.phi_s = RFSectionParameters.phi_s
        
        ### Import RF section parameters for accelerating kick
        #: *Copy of the momentum program in [eV/c] (from RFSectionParameters)*
        self.momentum = RFSectionParameters.momentum
        #: *Copy of the momentum increment in [eV/c] (from RFSectionParameters)*
        self.p_increment = RFSectionParameters.p_increment
        #: *Copy of the relativistic beta (from RFSectionParameters)*
        self.beta_r = RFSectionParameters.beta_r
        #: *Copy of the averaged relativistic beta (from RFSectionParameters)*
        self.beta_av = RFSectionParameters.beta_av
        #: *Copy of the relativistic gamma (from RFSectionParameters)*        
        self.gamma_r = RFSectionParameters.gamma_r
        #: *Copy of the relativistic energy (from RFSectionParameters)*                
        self.energy = RFSectionParameters.energy
        
        #: *Acceleration kick* :math:`: \quad - <\beta> \Delta p`
        self.acceleration_kick = - self.beta_av * self.p_increment  
        
        ### Import RF section parameters for the drift
        #: *Slippage factor (order 0) for the given RF section*
        self.eta_0 = RFSectionParameters.eta_0
        #: *Slippage factor (order 1) for the given RF section*
        self.eta_1 = RFSectionParameters.eta_1
        #: *Slippage factor (order 2) for the given RF section*
        self.eta_2 = RFSectionParameters.eta_2
        #: *Copy of the slippage factor order number (from RFSectionParameters)*                
        self.alpha_order = RFSectionParameters.alpha_order
            
        #: *Beta ratio*  :math:`: \quad \frac{\beta_{n+1}}{\beta_{n}}`  
        self.beta_ratio = self.beta_r[1:] / self.beta_r[0:-1]
        
        #: | *Choice of solver for the drift*
        #: | *Set to 'simple' if only 0th order of slippage factor eta*
        #: | *Set to 'full' if higher orders of slippage factor eta*
        self.solver = solver
        if self.alpha_order == 1:
            self.solver = 'simple'         
        
                   
    def kick(self, beam):
        '''
        *The Kick represents the kick(s) by an RF station at a certain position 
        of the ring. The kicks are summed over the different harmonic RF systems 
        in the station. The cavity phase can be shifted by the user via phi_offset.
        The increment in energy is given by the discrete equation of motion:*
        
        .. math::
            \Delta E_{n+1} = \Delta E_n + \sum_{j=0}^{n_{RF}}{V_{j,n}\,\sin{\\left(h_{j,n}\,\\theta + \phi_{j,n}\\right)}}
            
        '''

        for i in range(self.n_rf):
            beam.dE += self.voltage[i,self.counter[0]] * \
                       np.sin(self.harmonic[i,self.counter[0]] * 
                              beam.theta + self.phi_offset[i,self.counter[0]])
    
    
    def kick_acceleration(self, beam):
        '''
        *KickAcceleration gives a single accelerating kick to the bunch. 
        The accelerating kick is defined by the change in the design momentum 
        (synchronous momentum). 
        The acceleration is assumed to be distributed over the length of the 
        RF station, so the average beta is used in the calculation of the kick.
        An extra increment in the equation of motion with respect to the Kick
        object is given by:*
        
        .. math::
            \Delta E_{n+1} = \Delta E_n + <\\beta> \Delta p_{n\\rightarrow n+1}
            
        '''
        
        beam.dE += self.acceleration_kick[self.counter[0]]

        
    def drift(self, beam):
        '''
        *The drift updates the longitudinal coordinate of the particle after 
        applying the energy kick. The two options of tracking are: full, 
        corresponding to the cases where beta the slippage factor may be of 
        higher orders; and simple, where the slippage factor is of order 0 (the
        code is then faster).*
        
        *Corresponding to the equations:*
        
        .. math::
            \\theta_{n+1} = \\frac{\\beta_{n+1}}{\\beta_n}\\theta_n + 2\\pi\\left(\\frac{1}{1 - \\eta\\delta_n} - 1\\right)\\frac{L}{C} \quad \\text{(full)}
            
        .. math::
            \\approx> \\theta_{n+1} = \\frac{\\beta_{n+1}}{\\beta_n}\\theta_n + 2\\pi\\eta_0\\delta_n\\frac{L}{C} \quad \\text{(simple)}
        
        '''
        
        if self.solver == 'full': 
            beam.theta = self.beta_ratio[self.counter[0]] * beam.theta \
                         + 2 * np.pi * (1 / (1 - self.eta_tracking(beam.delta) * 
                                             beam.delta) - 1) * self.length_ratio
        elif self.solver == 'simple':
            beam.theta = self.beta_ratio[self.counter[0]] *beam.theta \
                         + 2 * np.pi * self.eta_0[self.counter[0]] \
                         * beam.delta * self.length_ratio
        else:
            raise RuntimeError("ERROR: Choice of longitudinal solver not \
                               recognized! Aborting...")
                
                
    def eta_tracking(self, delta):
        '''
        *The slippage factor is calculated as a function of the relative momentum
        (delta) of the beam. By definition, the slippage factor is:*
        
        .. math:: 
            \eta = \sum_{i}(\eta_i \, \delta^i)
    
        '''
        
        if self.alpha_order == 1:
            return self.eta_0[self.counter[0]]
        else:
            eta = 0
            for i in xrange( self.alpha_order ):
                eta_i = getattr(self, 'eta_' + str(i))[self.counter[0]]
                eta  += eta_i * (delta**i)
            return eta  
        
        
    def track(self, beam):
        '''
        | *Tracking method for the section, applies the equations in this order:*
        | *kick -> kick_acceleration -> drift*
        |
        | *Updates the relativistic information of the beam.*
        '''
        
        self.kick(beam)
        self.kick_acceleration(beam)
        self.drift(beam)
        
        self.counter[0] += 1
        
        # Updating the beam synchronous momentum etc.
        beam.beta_r = self.beta_r[self.counter[0]]
        beam.gamma_r = self.gamma_r[self.counter[0]]
        beam.energy = self.energy[self.counter[0]]
        beam.momentum = self.momentum[self.counter[0]]
      


class LinearMap(object):
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    Qs is forced to be constant.
    '''

    def __init__(self, GeneralParameters, Qs):
        
        #: *Copy of the relativistic beta (from GeneralParameters)*
        self.beta_r = GeneralParameters.beta_r[0,0]        
        #: *Copy of the ring circumference (from GeneralParameters)*
        self.ring_circumference = GeneralParameters.ring_circumference        
        #: *Copy of the 0th order slippage factor (from GeneralParameters)*
        self.eta = GeneralParameters.eta0[0,0]
        
        #: *Synchrotron tune (constant)*
        self.Qs = Qs
        
        #: *Copy of the revolution angular frequency (from GeneralParameters)*
        self.omega_0 = GeneralParameters.omega_rev[0]
        
        #: *Synchrotron angular frequency in [rad/s]* 
        #: :math:`: \quad \omega_s = Q_s \omega_0`
        self.omega_s = self.Qs * self.omega_0
        
        self.dQs = 2 * np.pi * self.Qs
        self.cosdQs = np.cos(self.dQs)
        self.sindQs = np.sin(self.dQs)
        

    def track(self, beam):

        z0 = beam.z
        delta0 = beam.delta

        beam.z = z0 * self.cosdQs - self.eta * c / self.omega_s * delta0 * self.sindQs
        beam.delta = delta0 * self.cosdQs + self.omega_s / self.eta / c * z0 * self.sindQs
        
        

