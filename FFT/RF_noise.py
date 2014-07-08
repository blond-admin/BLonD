'''
**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
'''


#from pylab import *
import numpy as np
import numpy.random as rnd
from scipy import integrate


class Phase_noise(object):
    '''
    *The Phase noise class contains the spectrum of RF phase noise and the 
    actual phase noise randomly generated from this spectrum (via mixing with
    white noise).
    
    The spectrum has to be input as double-sided spectrum, in units of 
    radian-square per hertz.
    
    Both hermitian to real and complex to complex FFTs are available.*
    '''    
    
    def __init__(self, frequency_array, real_part_of_spectrum, seed=None):

        self.f = frequency_array # in Hz
        self.ReS = real_part_of_spectrum # in rad^2/Hz
        
        self.nf = len(self.ReS)
        self.fmax = self.f[self.nf-1]
        self.seed = seed
        self.nt = 0
        self.dt = 0
        

    def spectrum_to_phase_noise(self, transform=None):
        '''
        *Transforms a the noise spectrum to phase noise data.
        Use transform=None or 'r' to transform hermitian spectrum to real phase.
        Use transform='c' to transform complex spectrum to complex phase.
        
        Returns only the real part of the phase noise.*
        '''
    
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
        rnd.RandomState(self.seed)
        r1 = rnd.random_sample(self.nt)
        r2 = rnd.random_sample(self.nt)
        if transform==None or transform=='r':
            Gt = np.cos(2*np.pi*r1) * np.sqrt(-2.*np.log(r2))     
        elif transform=='c':  
            Gt = np.cos(2.*np.pi*r1)*np.sqrt(-2.*np.log(r2)) \
                + 1j*np.sin(2.*np.pi*r1)*np.sqrt(-2.*np.log(r2))         
               
        # FFT to frequency domain
        if transform==None or transform=='r':
            Gf = np.fft.rfft(Gt)  
        elif transform=='c':
            Gf = np.fft.fft(Gt)   
                
        # Multiply by desired noise probability density
        A = integrate.trapz(self.ReS, self.f)
        if transform==None or transform=='r':
            s = np.sqrt(2*self.fmax*self.ReS) # in rad
        elif transform=='c':
            s = np.sqrt(self.ReS/A) # in rad
        dPf = s*Gf.real + 1j*s*Gf.imag 
                
        # FFT back to time domain to get final phase shift
        if transform==None or transform=='r':
            dPt = np.fft.irfft(dPf) 
        elif transform=='c':
            dPt = np.fft.ifft(dPf)
                    
        # Use only real part for the phase shift and normalize
        self.t = np.arange(0, self.nt*self.dt, self.dt)
        self.dphi = dPt.real
        print "   Generated RF phase noise from the noise spectrum"
        
        return self.t, self.dphi 
    



    
