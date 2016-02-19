
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to generate RF phase noise from noise spectrum and feedback noise
amplitude as a function of bunch length**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
import numpy.random as rnd

from scipy.constants import c
from plots.plot import *
from plots.plot_llrf import *
from input_parameters.rf_parameters import calc_phi_s
cfwhm = np.sqrt(2./np.log(2.))



class PhaseNoise(object): 
    
    def __init__(self, frequency_array, real_part_of_spectrum, seed1=None, 
                 seed2=None):

        self.f = frequency_array # in [Hz] 
        self.ReS = real_part_of_spectrum # in [rad^2/Hz]
        
        self.nf = len(self.ReS)
        self.fmax = self.f[self.nf-1]
        self.seed1 = seed1
        self.seed2 = seed2
        self.nt = 0
        self.dt = 0
        

    def spectrum_to_phase_noise(self, transform=None):
    
        # Resolution in time domain
        if transform==None or transform=='r':
            self.nt = 2*(self.nf - 1) 
            self.dt = 1/(2*self.fmax) # in [s]
        elif transform=='c':  
            self.nt = self.nf 
            self.dt = 1./self.fmax # in [s]
        else:
            raise RuntimeError('ERROR: The choice of Fourier transform for the\
             RF noise generation could not be recognized. Use "r" or "c".')
            
        # Generate white noise in time domain
        rnd.seed(self.seed1)
        r1 = rnd.random_sample(self.nt)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(self.nt)
        if transform==None or transform=='r':
            Gt = np.cos(2*np.pi*r1) * np.sqrt(-2*np.log(r2))     
        elif transform=='c':  
            Gt = np.exp(2*np.pi*1j*r1)*np.sqrt(-2*np.log(r2)) 

        # FFT to frequency domain
        if transform==None or transform=='r':
            Gf = np.fft.rfft(Gt)  
        elif transform=='c':
            Gf = np.fft.fft(Gt)   
                
        # Multiply by desired noise probability density
        if transform==None or transform=='r':
            s = np.sqrt(2*self.fmax*self.ReS) # in [rad]
        elif transform=='c':
            s = np.sqrt(self.fmax*self.ReS) # in [rad]
        dPf = s*Gf.real + 1j*s*Gf.imag  # in [rad]
                
        # FFT back to time domain to get final phase shift
        if transform==None or transform=='r':
            dPt = np.fft.irfft(dPf) # in [rad]
        elif transform=='c':
            dPt = np.fft.ifft(dPf) # in [rad]
                    
        # Use only real part for the phase shift and normalize
        self.t = np.linspace(0, self.nt*self.dt, self.nt) 
        self.dphi = dPt.real


    
class LHCFlatSpectrum(object): 
    
    def __init__(self, GeneralParameters, RFSectionParameters, time_points, 
                 corr_time = 10000, fmin = 0.8571, fmax = 1.1, 
                 initial_amplitude = 1.e-6, seed1 = 1234, seed2 = 7564, 
                 predistortion = None):

        '''
        Generate LHC-type phase noise from a band-limited spectrum.
        Input frequency band using 'fmin' and 'fmax' w.r.t. the synchrotron 
        frequency. Input double-sided spectrum amplitude [rad^2/Hz] using 
        'initial_amplitude'. Fix seeds to obtain reproducible phase noise.
        Select 'time_points' suitably to resolve the spectrum in frequency 
        domain. After 'corr_time' turns, the seed is changed to cut numerical
        correlated sequences of the random number generator.
        '''

        self.f0 = GeneralParameters.f_rev  # revolution frequency in Hz
        self.nt = time_points           # no. points in time domain to generate
        self.corr = corr_time           # adjust noise every 'corr' time steps
        self.fmin = fmin                # spectrum lower bound in synchr. freq.
        self.fmax = fmax                # spectrum upper bound in synchr. freq.
        self.A_i = initial_amplitude    # initial spectrum amplitude [rad^2/Hz]
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        if self.predistortion != None:
            # Overwrite frequencies
            self.fmin = 0.8571
            self.fmax = 1.001
              
        if self.nt < 2*self.corr:
            raise RuntimeError('ERROR: Need more time points in LHCFlatSpectrum.')
        
        # Synchrotron frequency array
        phis = calc_phi_s(RFSectionParameters, 
                          accelerating_systems='as_single')   
        self.fs = c/GeneralParameters.ring_circumference*np.sqrt( 
            RFSectionParameters.harmonic[0]*RFSectionParameters.voltage[0]
            *np.fabs(RFSectionParameters.eta_0*np.cos(phis))
            /(2.*np.pi*RFSectionParameters.energy) ) # Hz
                
        self.dphi = np.zeros(GeneralParameters.n_turns+1)
        
        self.n_turns = GeneralParameters.n_turns
 
 
    def generate(self):
       
        for i in xrange(0, int(self.n_turns/self.corr)):
        
            # Scale amplitude to keep area (phase noise amplitude) constant
            k = i*self.corr       # current time step
            ampl = self.A_i*self.fs[0]/self.fs[k]

            # Calculate the frequency step
            nf = int(self.nt/2 + 1)     # no. points in freq. domain
            df = self.f0[k]/self.nt          

            # Construct spectrum   
            nmin = np.floor(self.fmin*self.fs[k]/df)  
            nmax = np.ceil(self.fmax*self.fs[k]/df)    
            freq = np.linspace(0, nf*df, nf) #np.arange(0, nf*df, df) 
            
            # To compensate the notch due to PL at central frequency
            if self.predistortion == 'exponential':
                
                spectrum = np.concatenate(( np.zeros(nmin), ampl*np.exp(
                    np.log(100.)*np.arange(0,nmax-nmin+1)/(nmax-nmin) ), 
                                           np.zeros(nf-nmax-1) ))
             
            elif self.predistortion == 'linear':
                
                spectrum = np.concatenate((np.zeros(nmin), 
                    np.linspace(0, ampl, nmax-nmin+1), np.zeros(nf-nmax-1)))   
                
            elif self.predistortion == 'hyperbolic':

                spectrum = np.concatenate(( np.zeros(nmin), 
                    ampl*np.ones(nmax-nmin+1)* \
                    1/(1 + 0.99*(nmin - np.arange(nmin,nmax+1))
                       /(nmax-nmin)), np.zeros(nf-nmax-1) ))

            elif self.predistortion == 'weightfunction':

                frel = freq[nmin:nmax+1]/self.fs[k] # frequency relative to fs0
                frel[np.where(frel > 0.999)[0]] = 0.999 # truncate center freqs
                sigma = 0.754 # rms bunch length in rad corresponding to 1.2 ns
                gamma = 0.577216
                weight = (4.*np.pi*frel/sigma**2)**2 * \
                    np.exp(-16.*(1. - frel)/sigma**2) + \
                    0.25*( 1 + 8.*frel/sigma**2 * 
                           np.exp(-8.*(1. - frel)/sigma**2) * 
                           ( gamma + np.log(8.*(1. - frel)/sigma**2) + 
                             8.*(1. - frel)/sigma**2 ) )**2
                weight /= weight[0] # normalise to have 1 at fmin
                spectrum = np.concatenate(( np.zeros(nmin), ampl*weight, 
                                            np.zeros(nf-nmax-1) ))

            else:
                
                spectrum = np.concatenate((np.zeros(nmin), 
                    ampl*np.ones(nmax-nmin+1), np.zeros(nf-nmax-1)))               
            
            noise = PhaseNoise(freq, spectrum, self.seed1, self.seed2)
            noise.spectrum_to_phase_noise()
            self.seed1 +=239
            self.seed2 +=158
            
            
            # Fill phase noise array
            if i < int(self.n_turns/self.corr) - 1:
                kmax = (i + 1)*self.corr
            else:
                kmax = self.n_turns + 1
            self.dphi[k:kmax] = noise.dphi[0:(kmax-k)]
            
            fig_folder('fig_noise')
            plot_noise_spectrum(freq, spectrum, sampling=1, figno=i, 
                                dirname = 'fig_noise')
            plot_phase_noise(noise.t[0:(kmax-k)], noise.dphi[0:(kmax-k)], 
                             sampling=1, figno=i, dirname = 'fig_noise')
            rms_noise = np.std(noise.dphi)
            print "RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)" \
                %(noise.t[1], i, rms_noise, rms_noise*180/np.pi)



class LHCNoiseFB(object): 
    '''
    *Feedback on phase noise amplitude for LHC controlled longitudinal emittance
    blow-up using noise injection through cavity controller or phase loop.
    The feedback compares the FWHM bunch length of the bunch to a target value 
    and scales the phase noise to keep the targeted value.
    Activate the feedback either by passing it in RFSectionParameters or in
    the PhaseLoop object.
    Update the noise amplitude scaling using track().*
    '''    

    def __init__(self, RFSectionParameters, Slices, bl_target, gain = 1.5e-9, 
                 factor = 0.8, update_frequency = 22500, variable_gain = False):

        #: | *Import RFSectionParameters*
        self.rf_params = RFSectionParameters

        #: | *Import Slices*
        self.slices = Slices
              
        #: | *Phase noise scaling factor. Initially 0.*
        self.x = 0.
        
        #: | *Target bunch length [s], 4-sigma value.*        
        self.bl_targ = bl_target
        
        #: | *Measured bunch length [s], FWHM.*          
        self.bl_meas = bl_target
        
        #: | *Feedback recursion scaling factor.*  
        self.a = factor
        
        #: | *Update feedback every n_update turns.*
        self.n_update = update_frequency
        
        #: | *Switch to use constant or variable gain*
        self.variable_gain = variable_gain
        
        #: | *Feedback gain [1/s].*  
        if self.variable_gain == True:
            self.g = gain*(self.rf_params.omega_s0[0]/
                           self.rf_params.omega_s0)**2
        else:
            self.g = gain*np.ones(self.rf_params.n_turns + 1)            

        

    def track(self):
        '''
        *Calculate PhaseNoise Feedback scaling factor as a function of measured
        FWHM bunch length.*
        '''    

        # Track only in certain turns
        if (self.rf_params.counter[0] % self.n_update) == 0:
            
            # Update bunch length, every x turns determined in main file
            self.bl_meas = fwhm(self.slices)
        
            # Update noise amplitude-scaling factor
            self.x = self.a*self.x + self.g[self.rf_params.counter[0]]* \
                     (self.bl_targ - self.bl_meas)               
        
            # Limit to range [0,1]
            if self.x < 0:
                self.x = 0
            if self.x > 1:
                self.x = 1           
                   


def fwhm(Slices): 
    '''
    *Fast FWHM bunch length calculation with slice width precision.*
    '''    
    
    height = np.max(Slices.n_macroparticles)
    index = np.where(Slices.n_macroparticles > height/2.)[0]
    return cfwhm*(Slices.bin_centers[index[-1]] - Slices.bin_centers[index[0]])



