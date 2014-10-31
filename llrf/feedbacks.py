
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Various feedbacks**

:Authors: **Helga Timko**
'''

from __future__ import division
from scipy.constants import c
import numpy as np


class PhaseLoop(object): 
    '''
    One-turn phase loop for different machines with different hardware. Phase 
    loop is active only in certain turns, in which the frequency or the phase
    in the longitudinal tracker is modified. Note that phase shift is not yet
    implemented in the tracker!
    '''    
    def __init__(self, general_params, rf_params, gain, sampling_frequency = 1, 
                 machine = 'LHC', coefficients = None, RFnoise = None, FB = None):
        
        self.gain = gain # feedback gain, can be an array depending on machine
        self.dt = sampling_frequency # either in turns or in time, depending on machine
        self.machine = machine # machine name
        
        self.domega_RF_next = 0 # PL correction in RF revolution frequency, next time step
        self.domega_RF_prev = 0 # PL correction in RF revolution frequency, prev time step
        self.dphi = 0 # phase difference between bunch/beam and RF
        
        # Pre-processing
        if self.machine == 'PSB':
            self.counter = 0
            self.precalculate_time(general_params)
        elif self.machine == 'LHC':
            self.dt = 1 # make sure PL is on every turn
            
        self.phi_s_design = rf_params.phi_s
       
        # Import phase noise       
        self.RFnoise = RFnoise
        if (RFnoise != None and (len(RFnoise.dphi) != general_params.n_turns + 1)):
            raise RuntimeError('Phase noise has to have a length of n_turns + 1')
        
        # Import amplitude scaling for phase noise
        self.FB = FB
            
    def track(self, beam, tracker):
        
        # Update the correction of the previous time step
        self.domega_RF_prev = self.domega_RF_next

        # Update the correction of the next time step    
        getattr(self, self.machine)(beam, tracker)
        
    
    def precalculate_time(self, general_params):
        # Any cleverer way to do this?
        n = 0
        while n < general_params.t_rev.size: 
            summa = 0
            while summa < self.dt:
                summa += general_params.t_rev[n]
                n += 1
            np.append(self.on_time, n)
            
      
    def phase_difference_design(self, beam, tracker):
        # We compare the bunch COM phase with the actual synchronous phase (w/ intensity effects)
        # Actually, we should compare a the RF harmonic component of the beam spectrum to the RF phase!
        self.dphi = tracker.harmonic[0,tracker.counter[0]] * beam.mean_theta - self.phi_s_design[tracker.counter[0]]
        
        # Possibility to add RF phase noise through the PL
        if self.RFnoise != None:
            if self.FB != None:
                self.dphi += self.FB.x*self.RFnoise.dphi[tracker.counter[0]]
            else:
                self.dphi += self.RFnoise.dphi[tracker.counter[0]]
        

    def LHC(self, beam, tracker):
        '''
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is
        
        .. math::
            \\Delta \\omega_{RF} = g (\\Delta\\Phi_{PL} + \\Delta\\Phi_{N}) 
            
        where the phase noise for the controlled blow-up can be optionally 
        activated.       
        '''
        self.phase_difference_design(beam, tracker)
        
        if np.fabs(self.dphi) < 1.e-10 :
            self.domega_RF_next = 0.
        else:
            self.domega_RF_next = self.gain * self.dphi


    def PSB(self, beam, tracker):
        '''
        Calculation of the PSB RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is
        
        .. math::
            \\Delta \\omega_{RF} = 2 \\Pi g \\frac{a_0 \\Delta\\Phi_{PL}^2 + a_1 \\Delta\\Phi_{PL} + a_2 }{\\Delta\\Phi_{PL}^2 - b_1 \\Delta\\Phi_{PL} - b_2} 
            
        Input g through gain and [a_0, a_1, a_2, b_1, b_2] through coefficients.       
        '''

        # Update phase difference on certain turns
        if tracker.counter[0] == self.on_time[self.counter]:
            self.phase_difference_design(beam, tracker) # to be replaced
        
            if np.fabs(self.dphi) < 1.e-10 :
                self.domega_RF_next = 0.
            else:
                self.domega_RF_next = 2*np.pi*self.gain*( self.coefficients[0]
                *self.dphi*self.dpi + self.coefficients[1]*self.dphi 
                + self.coefficients[2]) / (self.dphi*self.dpi 
                + self.coefficients[3]*self.dphi + self.coefficients[4])
             
        # Counter to pick the next time step when the PL will be active
        self.counter += 1 
    
    

class LHCNoiseFB(object): 
    
    # Feedback based on bunch length, acting on phase noise used for blow-up 
    def __init__(self, general_params, bl_target, gain = 1.5e9, factor = 0.8, 
                 self_statistics = False ):

        self.x = 1 # multiplication factor; initially 1
        self.bl_targ = bl_target # 4 sigma, in s
        self.bl_meas = bl_target # set initial value for measured bunch length
        self.g = gain            # FB gain
        self.a = factor
        self.self_stat = self_statistics # using external/internal statistics
        self.R = general_params.ring_radius
        

    def FB(self, rf_params, beam, RFnoise, slices = None, CCL = False):

        # Update bunch length, every x turns determined in main file
        if slices:
            self.bl_meas = 4.*fwhm(slices)/2.3548
            if slices.slicing_coord == 'theta':
                self.bl_meas *= self.R/(beam.beta_r*c)
            elif slices.slicing_coord == 'z':
                self.bl_meas /= beam.beta_r*c                
            print "In FB, 4-sigma FWHM bunch length: %.4e s (4-sigma RMS: %.4e s)" \
                %(self.bl_meas, 4.*beam.sigma_tau)
        elif self.self_stat:
            itemindex = np.where(beam.id != 0)[0]
            self.bl_meas = 4.*np.std(beam.theta[itemindex]) \
                         *beam.ring_radius/(beam.beta_r*c)
        else:
            self.bl_meas = 4.*beam.sigma_tau
        
        # Update multiplication factor
        self.x = self.a*self.x + self.g*(self.bl_targ - self.bl_meas)               
        
        # Limit to range [0,1]
        if self.x < 0:
            self.x = 0
        if self.x > 1:
            self.x = 1      
            
        print "In FB, phase noise multiplication factor x is %.4e" %self.x     
        
        # Update the whole phase noise array of main RF system  
        if CCL == True:
            rf_params.phi_offset[0] = self.x*RFnoise.dphi    
        

def fwhm(slices): 
    
    height = np.max(slices.n_macroparticles)
    index = np.where(slices.n_macroparticles > height/2.)[0]
    return slices.bins_centers[index[-1]] - slices.bins_centers[index[0]]

    
    
    
        