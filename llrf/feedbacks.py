
# Copyright 2015 CERN. This software is distributed under the
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
import numpy as np
from input_parameters.constants import *
cfwhm = np.sqrt(2./np.log(2.))



class PhaseLoop(object): 
    '''
    One-turn beam phase loop for different machines with different hardware. 
    Use 'sampling_frequency' for a phase loop that is active only in certain 
    turns. The phase loop acts directly on the RF frequency of all harmonics and
    affects the RF phase as well.
    '''    
    def __init__(self, GeneralParameters, RFSectionParameters, Slices, gain, 
                 gain2 = 0, machine = 'LHC', period = None, 
                 window_coefficient = 0, coefficients = None, PhaseNoise = None, 
                 LHCNoiseFB = None):
        
        #: | *Import RFSectionParameters*
        self.rf_params = RFSectionParameters
        
        #: | *Import Slices*
        self.slices = Slices

        #: | *Feedback gain, single value. Implementation depends on machine.*        
        self.gain = gain 
        
        #: | *Feedback gain of synchro/radial loop associated to PL.*
        #: | *Implementation depends on machine.*        
        self.gain2 = gain2
        
        #: | *Optional: PL acting only in certain time intervals/turns.*
        self.dt = period
        
        #: | *Machine name; see description of each machine.*
        self.machine = machine 

        #: | *Band-pass filter window coefficient for beam phase calculation.*
        self.alpha = window_coefficient 

        #: | *Transfer function coefficient array used for certain machines.*
        self.coefficients = coefficients  
        
        #: | *Phase loop frequency correction of the main RF system.*
        self.domega_RF = 0
        
        #: | *Beam phase measured at the main RF frequency.*        
        self.phi_beam = 0
        
        #: | *Phase difference between beam and RF.*        
        self.dphi = 0
        
        #: | *Some pre-processing depending on machine.*
        if self.machine == 'PSB':
            self.precalculate_time(GeneralParameters)
        elif self.machine == 'LHC':
            self.dt = 1 # make sure PL is on every turn
            
        #: | *Optional import of RF PhaseNoise*       
        self.RFnoise = PhaseNoise
        if (self.RFnoise != None and 
            (len(self.RFnoise.dphi) != GeneralParameters.n_turns + 1)):
            raise RuntimeError('Phase noise has to have a length of n_turns + 1')
        
        #: | *Optional import of RF PhaseNoise amplitude scaling LHCNoiseFB*       
        self.FB = LHCNoiseFB
        
    
    def track(self):
        '''
        Calculate PL correction on main RF frequency depending on machine.
        Update the RF phase and frequency of the next turn for all systems.
        '''    
        
        # Calculate PL correction on RF frequency    
        getattr(self, self.machine)()

        # Update the RF frequency of all systems for the next turn
        counter = self.rf_params.counter[0] + 1
        self.rf_params.omega_RF[:,counter] += self.domega_RF* \
            self.rf_params.harmonic[:,counter]/self.rf_params.harmonic[0,counter]  
            
        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        self.rf_params.dphi_RF += 2.*np.pi*self.rf_params.harmonic[:,counter]* \
            self.domega_RF/self.rf_params.omega_RF_d[:,counter] 
                    
        # Total phase offset
        self.rf_params.phi_RF[:,counter] += self.rf_params.dphi_RF
    

    def precalculate_time(self, GeneralParameters):
        '''
        *For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.*
        '''    
        
        # Counter of turns passed since last time the PL was active
        self.counter = 0
        n = 0
        while n < GeneralParameters.t_rev.size: 
            summa = 0
            while summa < self.dt:
                summa += GeneralParameters.t_rev[n]
                n += 1
            np.append(self.on_time, n)
            

    def beam_phase(self):
        '''
        *Beam phase measured at the main RF frequency. The beam is convoluted 
        with the window function of the band-pass filter of the machine. The
        coefficients of sine and cosine components determine the beam phase.*
        '''    
        
        # Main RF frequency at the present turn
        omega_RF = self.rf_params.omega_RF[:,self.rf_params.counter[0]]
        phi_RF = self.rf_params.phi_RF[:,self.rf_params.counter[0]]
        
        # Convolve with window function
        scoeff = np.trapz( np.exp(self.alpha*self.slices.bin_centers) \
                           *np.sin(omega_RF*self.slices.bin_centers + phi_RF) \
                           *self.slices.n_macroparticles, self.slices.bin_centers )
        ccoeff = np.trapz( np.exp(self.alpha*self.slices.bin_centers) \
                           *np.cos(omega_RF*self.slices.bin_centers + phi_RF) \
                           *self.slices.n_macroparticles, self.slices.bin_centers )
        
        # Project beam phase to (-pi/2,3pi/2) range
        if np.fabs(ccoeff) > 1.e-20: 
            self.phi_beam = np.arctan(scoeff/ccoeff)
            if ccoeff < 0:
                self.phi_beam += np.pi
        else:
            self.phi_beam = np.pi/2.


    def phase_difference(self):               
        '''
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        '''    
        
        # Correct for design stable phase
        counter = self.rf_params.counter[0]
        self.dphi = self.phi_beam - self.rf_params.phi_s[counter]
        
        # Possibility to add RF phase noise through the PL
        if self.RFnoise != None:
            if self.FB != None:
                self.dphi += self.FB.x*self.RFnoise.dphi[counter]
            else:
                self.dphi += self.RFnoise.dphi[counter]
        

    def LHC(self):
        '''
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is
        
        .. math::
            \\Delta \\omega_{RF} = g (\\Delta\\Phi_{PL} + \\Delta\\Phi_{N}) 
            
        where the phase noise for the controlled blow-up can be optionally 
        activated.  
        Using 'gain2', a frequency loop can be activated in addition to remove
        long-term frequency drifts.     
        '''
        
        counter = self.rf_params.counter[0]
        self.beam_phase()
        self.phase_difference()
        self.domega_RF = - self.gain*self.dphi \
            - self.gain2*(self.rf_params.omega_RF[:,counter] 
               - self.rf_params.omega_RF_d[:,counter]) #frequency loop


    def PSB(self):
        '''
        Calculation of the PSB RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is
        
        .. math::
            \\Delta \\omega_{RF} = 2 \\Pi g \\frac{a_0 \\Delta\\Phi_{PL}^2 + a_1 \\Delta\\Phi_{PL} + a_2 }{\\Delta\\Phi_{PL}^2 - b_1 \\Delta\\Phi_{PL} - b_2} 
            
        Input g through gain and [a_0, a_1, a_2, b_1, b_2] through coefficients.       
        '''

        # Update phase difference on certain turns
        if self.rf_params.counter[0] == self.on_time[self.counter]:
            self.beam_phase()
            self.phase_difference_design()
        
            if np.fabs(self.dphi) < 1.e-10 :
                self.domega_RF = 0.
            else:
                self.domega_RF = - 2*np.pi*self.gain*( self.coefficients[0]
                *self.dphi*self.dpi + self.coefficients[1]*self.dphi 
                + self.coefficients[2]) / (self.dphi*self.dpi 
                + self.coefficients[3]*self.dphi + self.coefficients[4])
                
        else:
            self.domega_RF = 0.
             
        # Counter to pick the next time step when the PL will be active
        self.counter += 1 
    
    

class LHCNoiseFB(object): 
    '''
    *Feedback on phase noise amplitude for LHC controlled longitudinal emittance
    blow-up using noise injection through cavity controller or phase loop.
    The feedback compares the FWHM bunch length of the bunch to a target value 
    and scales the phase noise to keep the targeted value.*
    '''    

    def __init__(self, bl_target, gain = 1.5, factor = 0.8):

        #: | *Phase noise scaling factor. Initially 0.*
        self.x = 0.
        
        #: | *Target bunch length [s], 4-sigma value.*        
        self.bl_targ = bl_target
        
        #: | *Measured bunch length [s], FWHM.*          
        self.bl_meas = bl_target
        
        #: | *Feedback gain [1/s].*  
        self.g = gain            # FB gain
        
        #: | *Feedback recursion scaling factor.*  
        self.a = factor
        

    def FB(self, RFSectionParameters, Beam, PhaseNoise, Slices, CC = False):
        '''
        *Calculate PhaseNoise Feedback scaling factor as a function of measured
        FWHM bunch length.*
        '''    

        # Update bunch length, every x turns determined in main file
        self.bl_meas = fwhm(Slices)
        
        # Update multiplication factor
        self.x = self.a*self.x + self.g*(self.bl_targ - self.bl_meas)               
        
        # Limit to range [0,1]
        if self.x < 0:
            self.x = 0
        if self.x > 1:
            self.x = 1           
        
        # Update the whole phase noise array of main RF system  
        if CC == True:
            RFSectionParameters.phi_noise[0] = self.x*PhaseNoise.dphi    
        


def fwhm(Slices): 
    '''
    *Fast FWHM bunch length calculation with slice width precision.*
    '''    
    
    height = np.max(Slices.n_macroparticles)
    index = np.where(Slices.n_macroparticles > height/2.)[0]
    return cfwhm*(Slices.bin_centers[index[-1]] - Slices.bin_centers[index[0]])

    
    
        