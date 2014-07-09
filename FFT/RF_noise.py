'''
**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
'''

import numpy as np
import numpy.random as rnd


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
        In this case, input only the positive part of the double-sided spectrum.
        
        Use transform='c' to transform complex spectrum to complex phase.
        In this case, input first the zero and positive frequency components, 
        then the decreasingly negative frequency components of the double-sided
        spectrum.
        
        E.g. the following two ways of usage are equivalent:
        .. image:: RF_noise.png
            :align: center
            :width: 600
            :height: 600       
        
        Returns only the real part of the phase noise.*
        '''
    
        # Resolution in time domain
        '''
            STEP 1: Set the resolution in time domain
        For hermitian spectrum to real phase noise,
        
        .. math:: n_t = 2 (n_f - 1) \text{and} \Delta t = 1/(2 f_{\text{max}}) 
        
        for complex spectrum to complex phase noise,
        
        .. math:: n_t = n_f \text{and} \Delta t = 1/f_{\text{max}} ,
        
        where f_{max} is the maximum frequency in the input in both cases.         
        '''
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
        '''
            STEP 2: Generate white (carrier) noise in time domain
        
        .. math:: 
            w_k(t) = \cos(2 \pi r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{case 'r'},
            
            w_k(t) = \exp(2 \pi i r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{case 'c'},           
        
        '''
        rnd.RandomState(self.seed)
        r1 = rnd.random_sample(self.nt)
        r2 = rnd.random_sample(self.nt)
        if transform==None or transform=='r':
            Gt = np.cos(2*np.pi*r1) * np.sqrt(-2*np.log(r2))     
        elif transform=='c':  
            Gt = np.exp(2*np.pi*1j*r1)*np.sqrt(-2*np.log(r2)) 

        # FFT to frequency domain
        '''
            STEP 3: Transform the generated white noise to frequency domain
            
        .. math:: W_l(f) = \sum_{k=1}^N w_k(t) e^{-2 \pi i \frac{k l}{N}}
        '''
        if transform==None or transform=='r':
            Gf = np.fft.rfft(Gt)  
        elif transform=='c':
            Gf = np.fft.fft(Gt)   
                
        # Multiply by desired noise probability density
        '''
            STEP 4: In frequency domain, colour the white noise with the desired
            noise probability density (unit: radians)
        
        The noise probability density derived from the double-sided spectrum is
        .. math:: s_l(f) = \sqrt{A S_l^{\text{DB}} f_{\text{max}}} ,  
        where A=2 for transform = 'r' and A=1 for transform = 'c'.
        The coloured noise is obtained by multiplication in frequency domain
        .. math:: \Phi_l(f) = \s_l(f) W_l(f)
        '''
        if transform==None or transform=='r':
            s = np.sqrt(2*self.fmax*self.ReS) # in rad
        elif transform=='c':
            s = np.sqrt(self.fmax*self.ReS) # in rad
        dPf = s*Gf.real + 1j*s*Gf.imag  # in rad
                
        # FFT back to time domain to get final phase shift
        '''
            STEP 5: Transform back the coloured spectrum to time domain to 
            obtain the final phase shift array (we use only the real part)
        '''
        if transform==None or transform=='r':
            dPt = np.fft.irfft(dPf) # in rad
        elif transform=='c':
            dPt = np.fft.ifft(dPf) # in rad
                    
        # Use only real part for the phase shift and normalize
        self.t = np.arange(0, self.nt*self.dt, self.dt)
        self.dphi = dPt.real
        print "   Generated RF phase noise from the noise spectrum"
        
        return self.t, self.dphi 
    



    
