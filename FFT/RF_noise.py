'''
**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
'''


#from pylab import *
import numpy as np
import numpy.random as rnd
#from scipy import integrate


class Phase_noise(object):
    '''
    *The Phase noise class contains the spectrum of RF phase noise and the 
    actual phase noise randomly generated from this spectrum (via mixing with
    white noise).
    
    The spectrum has to be input as single-sided spectrum.
    
    Both hermitian to real and complex to complex FFTs are available.*
    '''    
    
    def __init__(self, frequency_array, real_part_of_spectrum, seed=None):

        self.f = frequency_array
        self.ReS = real_part_of_spectrum
        
        self.nf = len(self.ReS)
        self.fmax = self.f[self.nf-1]
        self.seed = seed
        self.nt = 0
        self.dt = 0
        

    def spectrum_to_phase_noise_r(self):
        '''
        *Transforms a hermitian spectrum to real phase noise.*
        '''
    
        # Resolution in time domain
        self.nt = 2*(self.nf - 1)    
       
        # Generate white noise in time domain
        rnd.RandomState(self.seed)
        r1 = rnd.random_sample(self.nt)
        r2 = rnd.random_sample(self.nt)
        Gt = np.cos(2*np.pi*r1) * np.sqrt(-2.*np.log(r2))     
               
        # FFT to frequency domain
        Gf = np.fft.rfft(Gt)   
        
        # Multiply by desired noise probability density
        s = np.sqrt(2*self.fmax*self.ReS)
        #dPf = np.complex(s*Gf.real, s*Gf.imag)
        dPf = s*Gf.real + 1j*s*Gf.imag
        
        # FFT back to time domain to get final phase shift
        dPt = np.fft.irfft(dPf)
    
        # Use only real part for the phase shift and normalize
        self.dt = 1/(2*self.fmax) # s
        self.t = np.arange(0, self.nt*self.dt, self.dt)
        self.dphi = dPt.real
        print "   Generated RF phase noise from the noise spectrum"
        return self.dphi
    
    def time_r(self):

        self.t = np.arange(0, self.nt*self.dt, self.dt)
        return self.t
# # Complex noise and spectrum
# def spectrum_to_phase_noise_option2(nf,nt,f,ReS,t,dPhi):
#     
#  
#     print "Generating RF phase noise from the phase spectrum..."
#    
#     # Generate white noise in time domain
#     Gt = zeros( (nt), dtype=complex )
#     random.seed()
#     for i in range (0,nt):
#         r1 = random.random()
#         r2 = random.random()
#         Gt[i] = complex( cos(2.*math.pi*r1)*math.sqrt(-2.*log(r2)), sin(2.*math.pi*r1)*math.sqrt(-2.*log(r2)) )   
# 
#     # FFT to frequency domain
#     Gf = numpy.fft.fft(Gt)
#     print "length of Gf is %d" % len(Gf)
#     
#     # Multiply by desired noise probability density
#     A = integrate.trapz(ReS,f)
#     #A = integrate.simps(ReS,f)
#     pd = []
#     dPf = zeros( (nf), dtype=complex )
#     for i in range (0,nf):
#         pd.append( math.sqrt(ReS[i]/A) )  
#         dPf[i] = complex( pd[i]*Gf.real[i], pd[i]*Gf.imag[i] )
#     print "length of dPf is %d" % len(dPf)
#     
#     # FFT back to time domain to get final phase shift
#     dPt = numpy.fft.ifft(dPf)
#     print "length of dPt is %d" % len(dPt)
# #    print dPt
# 
#     # Use only real part for the phase shift and normalize
#     dt = 1./f[nf-1] #1./2./f[nf-1] # s
#     for i in range(0,nt):
#         dPhi[i] = dPt.real[i] #*math.pi #*2.
#         t[i] = i*dt
#     print ""
    
