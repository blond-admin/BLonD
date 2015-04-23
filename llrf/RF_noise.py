
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
import numpy.random as rnd

from plots.plot_llrf import *
from input_parameters.rf_parameters import calc_phi_s
from scipy.constants import c


class PhaseNoise(object): 
    
    def __init__(self, frequency_array, real_part_of_spectrum, seed1=None, 
                 seed2=None):

        self.f = frequency_array # in [Hz] (or [GHz])
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
            self.dt = 1/(2*self.fmax) # in [s] (or [ns])
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

        # LLRF to frequency domain
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
                
        # LLRF back to time domain to get final phase shift
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
                 initial_amplitude = 1.e-6, seed1 = 1234, seed2 = 7564):

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
              
        if self.nt < 2*self.corr:
            raise RuntimeError('ERROR: Need more time points in LHCFlatSpectrum.')
        
        # Synchrotron frequency array
        phis = calc_phi_s(RFSectionParameters, accelerating_systems='as_single')   
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
            spectrum = np.concatenate((np.zeros(nmin), ampl*np.ones(nmax-nmin+1), 
                                       np.zeros(nf-nmax-1)))
            
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
            
            plot_noise_spectrum(freq, spectrum, sampling=1, figno=i)
            plot_phase_noise(noise.t[0:(kmax-k)], noise.dphi[0:(kmax-k)], sampling=1, figno=i)
            rms_noise = np.std(noise.dphi)
            print "RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)" \
                %(noise.t[1], i, rms_noise, rms_noise*180/np.pi)



