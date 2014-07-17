'''
**Module containing all the elements to track the beam in the longitudinal plane.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Adrian Oeftiger**, **Alexandre Lasheen**
'''

from __future__ import division
import numpy as np
from scipy.constants import c


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
        
    def __init__(self, rf_params, solver='full'):
        
        #: | *Choice of solver for the drift*
        #: | *Use 'full' for full eta solver*
        #: | *Use 'simple' for 0th order eta solver*
        self.solver = solver
        
        #: | *Counter to keep track of time step (used in momentum and voltage)*
        self.counter = rf_params.counter
        
        #: | *Import RF section parameters for RF kick*
        #: | *Length ratio between drift and ring circumference*  
        #: | :math:`: \quad \frac{L}{C}`
        self.length_ratio = rf_params.length_ratio
        #: | *Harmonic number list* :math:`: \quad h_{j,n}`
        self.harmonic = rf_params.harmonic
        #: | *Voltage program list in [V]* :math:`: \quad V_{j,n}`
        self.voltage = rf_params.voltage
        #: | *Phase offset list in [rad]* :math:`: \quad \phi_{j,n}`
        self.phi_offset = rf_params.phi_offset
        #: | *Number of RF systems in the RF station* :math:`: \quad n_{RF}`
        self.n_rf = rf_params.n_rf
        
        #: | *Import RF section parameters for accelerating kick*
        #: | *Momentum (program) in [eV/c]* :math:`: \quad p_n`
        self.momentum = rf_params.momentum
        self.p_increment = rf_params.p_increment
        #: | *... and derived relativistic quantities*
        self.beta_r = rf_params.beta_r
        
        self.beta_av = rf_params.beta_av
        self.gamma_r = rf_params.gamma_r
        self.energy = rf_params.energy
        #: *Acceleration kick* :math:`: \quad - <\beta> \Delta p`
        
        self.acceleration_kick = - self.beta_av * self.p_increment  

        #: *Beta ratio*  :math:`: \quad \frac{\beta_{n+1}}{\beta_{n}}`  
        self.beta_ratio = self.beta_r[1:] / self.beta_r[0:-1]

        #: *Slippage factor up to desired order*
        self.alpha_order = rf_params.alpha_order
        for i in xrange( self.alpha_order ):
            dummy = getattr(rf_params, 'eta_' + str(i))
            setattr(self, "eta_%s" %i, dummy)
        # For the eta tracking, import the RF section class
        self.rf_params = rf_params    
            
        
                   
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
            \Delta E_{n+1} = \Delta E_n + \\beta_{av} \Delta p_{n\\rightarrow n+1}
            
        '''
        
        beam.dE += self.acceleration_kick[self.counter[0]]

        
    def drift(self, beam):
        '''
        *The drift updates the longitudinal coordinate of the particle after 
        applying the energy kick. The two options of tracking are: full, 
        corresponding to the cases where beta is not considered constant and
        the slippage factor may be of higher orders; and simple, where beta
        is approximatively one and the slippage factor is of order 0. Corresponding
        to the equations:*
        
        .. math::
            \\theta_{n+1} = \\frac{\\beta_{n+1}}{\\beta_n}\\theta_n + 2\\pi\\left(\\frac{1}{1 - \\eta\\delta_n} - 1\\right)\\frac{L}{C} \quad \\text{(full)}
            
        .. math::
            \\approx> \\theta_{n+1} = \\theta_n + 2\\pi\\eta_0\\delta_n\\frac{L}{C} \quad \\text{(simple)}
        
        '''
        
        if self.solver == 'full': 
            beam.theta = self.beta_ratio[self.counter[0]] * beam.theta \
                         + 2 * np.pi * (1 / (1 - self.rf_params.eta_tracking(beam.delta) * 
                                             beam.delta) - 1) * self.length_ratio
        elif self.solver == 'simple':
            beam.theta = self.beta_ratio[self.counter[0]] *beam.theta \
                         + 2 * np.pi * self.eta_0[self.counter[0]] \
                         * beam.delta * self.length_ratio
        else:
            raise RuntimeError("ERROR: Choice of longitudinal solver not \
                               recognized! Aborting...")
                
        
    def track(self, beam):
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
    self.alpha is the linear momentum compaction factor.
    Qs is forced to be constant.
    '''

    def __init__(self, GeneralParameters, Qs):

        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        
        self.beta_r = GeneralParameters.beta_r[0,0]
        
        self.ring_circumference = GeneralParameters.ring_circumference
        self.eta = GeneralParameters.eta0[0,0]
        self.Qs = Qs
        self.omega_0 = 2 * np.pi * self.beta_r * c / self.ring_circumference
        self.omega_s = self.Qs * self.omega_0
        
        self.dQs = 2 * np.pi * self.Qs
        self.cosdQs = np.cos(self.dQs)
        self.sindQs = np.sin(self.dQs)
        

    def track(self, beam):

        z0 = beam.z
        delta0 = beam.delta

        beam.z = z0 * self.cosdQs - self.eta * c / self.omega_s * delta0 * self.sindQs
        beam.delta = delta0 * self.cosdQs + self.omega_s / self.eta / c * z0 * self.sindQs
        
        

