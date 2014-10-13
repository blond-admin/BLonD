'''
**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
'''

import numpy as np
import numpy.random as rnd


class PhaseNoise(object): 
    
    def __init__(self, frequency_array, real_part_of_spectrum, seed1=None, 
                 seed2=None):

        self.f = frequency_array # in Hz
        self.ReS = real_part_of_spectrum # in rad^2/Hz
        
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
            self.dt = 1/(2*self.fmax) # s
        elif transform=='c':  
            self.nt = self.nf 
            self.dt = 1./self.fmax # s
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
            s = np.sqrt(2*self.fmax*self.ReS) # in rad
        elif transform=='c':
            s = np.sqrt(self.fmax*self.ReS) # in rad
        dPf = s*Gf.real + 1j*s*Gf.imag  # in rad
                
        # LLRF back to time domain to get final phase shift
        if transform==None or transform=='r':
            dPt = np.fft.irfft(dPf) # in rad
        elif transform=='c':
            dPt = np.fft.ifft(dPf) # in rad
                    
        # Use only real part for the phase shift and normalize
        self.t = np.arange(0, self.nt*self.dt, self.dt)
        self.dphi = dPt.real

    



    
