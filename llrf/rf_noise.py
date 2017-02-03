
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

:Authors: **Helga Timko, D. Quartullo, J. Repond**
'''

from __future__ import division, print_function
from builtins import range, object
from toolbox.next_regular import next_regular
import numpy as np
import numpy.random as rnd

from scipy.constants import c
from plots.plot import *
from plots.plot_llrf import *
from input_parameters.rf_parameters import calc_phi_s
cfwhm = np.sqrt(2./np.log(2.))
import matplotlib.pyplot as plt



class FlatSpectrum(object): 
    
    def __init__(self, GeneralParameters, RFSectionParameters, delta_f = 1, 
                 corr_time = 10000, fmin_s0 = 0.8571, fmax_s0 = 1.1, 
                 initial_amplitude = 1.e-6, seed1 = 1234, seed2 = 7564, 
                 predistortion = None, continuous_phase = False):

        '''
        Generate phase noise from a band-limited spectrum.
        Input frequency band using 'fmin' and 'fmax' w.r.t. the synchrotron 
        frequency. Input double-sided spectrum amplitude [rad^2/Hz] using 
        'initial_amplitude'. Fix seeds to obtain reproducible phase noise.
        Select 'time_points' suitably to resolve the spectrum in frequency 
        domain. After 'corr_time' turns, the seed is changed to cut numerical
        correlated sequences of the random number generator.
        '''

        self.f0 = GeneralParameters.f_rev  # revolution frequency in Hz
        self.delta_f = delta_f           # frequency resolution [Hz]
        self.corr = corr_time           # adjust noise every 'corr' time steps
        self.fmin_s0 = fmin_s0                # spectrum lower bound in synchr. freq.
        self.fmax_s0 = fmax_s0                # spectrum upper bound in synchr. freq.
        self.A_i = initial_amplitude    # initial spectrum amplitude [rad^2/Hz]
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        if self.predistortion != None:
            # Overwrite frequencies
            self.fmin_s0 = 0.8571
            self.fmax_s0 = 1.001
        self.fs = RFSectionParameters.omega_s0 / (2*np.pi) # synchrotron frequency in Hz
        self.n_turns = GeneralParameters.n_turns 
        self.dphi = np.zeros(self.n_turns+1)
        self.continuous_phase = continuous_phase
        if self.continuous_phase:
            self.dphi2 = np.zeros(self.n_turns+1+self.corr/4)
        
    
    def spectrum_to_phase_noise(self, freq, spectrum, transform=None):
        
        nf = len(spectrum)
        fmax = freq[nf-1]
        
        # Resolution in time domain
        if transform==None or transform=='r':
            nt = 2*(nf - 1) 
            dt = 1/(2*fmax) # in [s]
        elif transform=='c':  
            nt = nf 
            dt = 1./fmax # in [s]
        else:
            raise RuntimeError('ERROR: The choice of Fourier transform for the\
             RF noise generation could not be recognized. Use "r" or "c".')
            
        # Generate white noise in time domain
        rnd.seed(self.seed1)
        r1 = rnd.random_sample(nt)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(nt)
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
            s = np.sqrt(2*fmax*spectrum) # in [rad]
        elif transform=='c':
            s = np.sqrt(fmax*spectrum) # in [rad]
        dPf = s*Gf.real + 1j*s*Gf.imag  # in [rad]
                
        # FFT back to time domain to get final phase shift
        if transform==None or transform=='r':
            dPt = np.fft.irfft(dPf) # in [rad]
        elif transform=='c':
            dPt = np.fft.ifft(dPf) # in [rad]
                    
        # Use only real part for the phase shift and normalize
        self.t = np.linspace(0, nt*dt, nt) 
        self.dphi_output = dPt.real
 
    
    def generate(self):
       
        for i in range(0, int(self.n_turns/self.corr)):
        
            # Scale amplitude to keep area (phase noise amplitude) constant
            k = i*self.corr       # current time step
            ampl = self.A_i*self.fs[0]/self.fs[k]
            
            # Calculate the frequency step
            f_max = self.f0[k]/2
            n_points_pos_f_incl_zero = np.ceil(f_max/self.delta_f) + 1
            nt = 2*(n_points_pos_f_incl_zero - 1)
            nt_regular = next_regular(int(nt))
            if nt_regular%2!=0 or nt_regular < self.corr:
                raise RuntimeError('Error in noise generation!')
            n_points_pos_f_incl_zero = nt_regular/2 + 1  
            freq = np.linspace(0, f_max, n_points_pos_f_incl_zero)
            delta_f = f_max/(n_points_pos_f_incl_zero-1) 

            # Construct spectrum   
            nmin = np.floor(self.fmin_s0*self.fs[k]/delta_f)  
            nmax = np.ceil(self.fmax_s0*self.fs[k]/delta_f)    
            
            # To compensate the notch due to PL at central frequency
            if self.predistortion == 'exponential':
                
                spectrum = np.concatenate((np.zeros(nmin), ampl*np.exp(
                    np.log(100.)*np.arange(0,nmax-nmin+1)/(nmax-nmin) ), 
                                           np.zeros(n_points_pos_f_incl_zero-nmax-1) ))
             
            elif self.predistortion == 'linear':
                
                spectrum = np.concatenate((np.zeros(nmin), 
                    np.linspace(0, ampl, nmax-nmin+1), np.zeros(n_points_pos_f_incl_zero-nmax-1)))   
                
            elif self.predistortion == 'hyperbolic':

                spectrum = np.concatenate((np.zeros(nmin), 
                    ampl*np.ones(nmax-nmin+1)* \
                    1/(1 + 0.99*(nmin - np.arange(nmin,nmax+1))
                       /(nmax-nmin)), np.zeros(n_points_pos_f_incl_zero-nmax-1) ))

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
                spectrum = np.concatenate((np.zeros(nmin), ampl*weight, 
                                            np.zeros(n_points_pos_f_incl_zero-nmax-1)))

            else:
    
                spectrum = np.concatenate((np.zeros(nmin), 
                    ampl*np.ones(nmax-nmin+1), np.zeros(n_points_pos_f_incl_zero-nmax-1)))               
            
            
            # Fill phase noise array
            if i < int(self.n_turns/self.corr) - 1:
                kmax = (i + 1)*self.corr
            else:
                kmax = self.n_turns + 1
            
            
            self.spectrum_to_phase_noise(freq, spectrum)
            self.seed1 +=239
            self.seed2 +=158
            self.dphi[k:kmax] = self.dphi_output[0:(kmax-k)]
            
            if self.continuous_phase:
                if i==0:
                    self.spectrum_to_phase_noise(freq, spectrum)
                    self.seed1 +=239
                    self.seed2 +=158
                    self.dphi2[:self.corr/4] = self.dphi_output[:self.corr/4]
                    
                self.spectrum_to_phase_noise(freq, spectrum)
                self.seed1 +=239
                self.seed2 +=158
                self.dphi2[(k+self.corr/4):(kmax+self.corr/4)] = self.dphi_output[0:(kmax-k)]
            
            fig_folder('fig_noise')
            plot_noise_spectrum(freq, spectrum, sampling=1, figno=i, 
                                dirname = 'fig_noise')
            plot_phase_noise(self.t[0:(kmax-k)], self.dphi_output[0:(kmax-k)], 
                             sampling=1, figno=i, dirname = 'fig_noise')
            rms_noise = np.std(self.dphi_output)
            print("RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)" \
                %(self.t[1], i, rms_noise, rms_noise*180/np.pi))
                
        if self.continuous_phase:
            psi = np.arange(0, self.n_turns+1)*2*np.pi/self.corr
            self.dphi = self.dphi*np.sin(psi[:self.n_turns+1]) + self.dphi2[:(self.n_turns+1)]*np.cos(psi[:self.n_turns+1])


class LHCNoiseFB(object): 
    '''
    *Feedback on phase noise amplitude for LHC controlled longitudinal emittance
    blow-up using noise injection through cavity controller or phase loop.
    The feedback compares the FWHM bunch length of the bunch to a target value 
    and scales the phase noise to keep the targeted value.
    Activate the feedback either by passing it in RFSectionParameters or in
    the PhaseLoop object.
    Update the noise amplitude scaling using track().
    Pass the bunch pattern (occupied bucket numbers from 0...h-1) in buckets 
    for multi-bunch simulations; the feedback uses the average bunch length.*
    '''    

    def __init__(self, RFSectionParameters, Slices, bl_target, gain = 0.1e9, 
                 factor = 0.93, update_frequency = 22500, variable_gain = True,
                 bunch_pattern = None):

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

        #: | *Bunch pattern for multi-bunch simulations*
        self.bunch_pattern = bunch_pattern
        
        #: | *Function dictionary to calculate FWHM bunch length*
        fwhm_functions = {'single': self.fwhm_single_bunch,
                          'multi': self.fwhm_multi_bunch}
        if self.bunch_pattern == None:
            self.fwhm = fwhm_functions['single']
            self.bl_meas_bbb = None
        else: 
            self.bunch_pattern = np.ascontiguousarray(self.bunch_pattern)
            self.bl_meas_bbb = np.zeros(len(self.bunch_pattern))
            self.fwhm = fwhm_functions['multi']
        

    def track(self):
        '''
        *Calculate PhaseNoise Feedback scaling factor as a function of measured
        FWHM bunch length.*
        '''    

        # Track only in certain turns
        if (self.rf_params.counter[0] % self.n_update) == 0:
            
            # Update bunch length, every x turns determined in main file
            self.fwhm()
        
            # Update noise amplitude-scaling factor
            self.x = self.a*self.x + self.g[self.rf_params.counter[0]]* \
                     (self.bl_targ - self.bl_meas)               
        
            # Limit to range [0,1]
            if self.x < 0:
                self.x = 0
            if self.x > 1:
                self.x = 1           
                   

    def fwhm_interpolation(self, index, half_height):
    
        time_resolution = self.slices.bin_centers[1]-self.slices.bin_centers[0]
        
        left = self.slices.bin_centers[index[0]] - (self.slices.n_macroparticles[index[0]] -
               half_height)/(self.slices.n_macroparticles[index[0]] -
               self.slices.n_macroparticles[index[0]-1])*time_resolution
               
        right = self.slices.bin_centers[index[-1]] + (self.slices.n_macroparticles[index[-1]]
                - half_height)/(self.slices.n_macroparticles[index[-1]] -
                self.slices.n_macroparticles[index[-1]+1])*time_resolution

        return cfwhm*(right - left)
        
    
    def fwhm_single_bunch(self): 
        '''
        *Single-bunch FWHM bunch length calculation with interpolation.*
        '''    
        
        half_height = np.max(self.slices.n_macroparticles)/2.
        index = np.where(self.slices.n_macroparticles > half_height)[0]   
    
        self.bl_meas = self.fwhm_interpolation(index, half_height)

    
    def fwhm_multi_bunch(self):
        '''
        *Multi-bunch FWHM bunch length calculation with interpolation.*
        '''    

        # Find correct RF buckets
        phi_RF = self.rf_params.phi_RF[0,self.rf_params.counter[0]]
        omega_RF = self.rf_params.omega_RF[0,self.rf_params.counter[0]]
        bucket_min = (phi_RF + 2.*np.pi*self.bunch_pattern)/omega_RF
        bucket_max = bucket_min + 2.*np.pi/omega_RF

        # Bunch-by-bunch FWHM bunch length
        for i in range(len(self.bunch_pattern)):
            
            bind = np.where((self.slices.bin_centers - bucket_min[i])*
                            (self.slices.bin_centers - bucket_max[i]) < 0)[0]
            hheight = np.max(self.slices.n_macroparticles[bind])/2.
            index = np.where(self.slices.n_macroparticles[bind] > hheight)[0]
            self.bl_meas_bbb[i] = self.fwhm_interpolation(bind[index], hheight)
            
        # Average FWHM bunch length            
        self.bl_meas = np.mean(self.bl_meas_bbb)


class SPS_phase_noise_injection(object): 
    
    def __init__(self,noise_nturn_recalculation,noise_time_window,GeneralParameters, RFSectionParameters, delta_f = 1, 
                 corr_time = 10000, fmin = 0.8571, fmax = 1.1, 
                 initial_amplitude = 1.e-6, seed1 = 1234, seed2 = 7564, 
                 predistortion = None, rescale_amplitude = 'with_sync_freq'):
        
        self.rf_params = RFSectionParameters
        self.noise_nturn_recalculation = noise_nturn_recalculation
        self.noise_time_window = noise_time_window
        self.turn = 0.
        
        self.f_rev = GeneralParameters.f_rev  
        self.delta_f = delta_f           
        self.corr = corr_time           
        self.fmin = fmin                
        self.fmax = fmax
        self.initial_amplitude = initial_amplitude    
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        self.rescale_amplitude = rescale_amplitude
        self.f_s0 = RFSectionParameters.omega_s0 / (2*np.pi)
        self.n_turns = GeneralParameters.n_turns
        self.dphi = np.zeros(self.n_turns+1)
        self.totalnoise = np.zeros(self.n_turns+1)
        
        self.freqmax = self.fmax*self.f_s0
        self.freqmin = self.fmin*self.f_s0
        
        self.fs0_array = np.array([])
        self.fedge_array = np.array([])
        
#        self.totalnoisespectrum
        
    
    def generate(self):
        
        for i in xrange(0, int(self.n_turns/self.corr)+1):
            
            k = i*self.corr
            f_max = self.f_rev[k]/2
            f_s0 = self.f_s0[k]
            
            n_points_pos_f_incl_zero = int(f_max/self.delta_f)+2
            nt = 2*(n_points_pos_f_incl_zero - 1)
            nt_regular = next_regular(nt)
            n_points_pos_f_incl_zero = nt_regular/2 + 1  
            freq = np.linspace(0, f_max, n_points_pos_f_incl_zero)
            new_delta_f = f_max/(n_points_pos_f_incl_zero-1) 
            
#            f_s0 = 1.3e5
            nmin = np.floor(self.freqmin/new_delta_f)  
            nmax = np.ceil(self.freqmax/new_delta_f)
            
#            print 'nmin nmax : ',nmin,nmax
            
            if self.rescale_amplitude == 'with_sync_freq':
                ampl = self.initial_amplitude*self.f_s0[0]/f_s0
            elif self.rescale_amplitude == 'no_scaling':
                ampl = self.initial_amplitude
            
            if self.predistortion == 'linear':
                
                spectrum = np.concatenate((np.zeros(nmin), 
                    np.linspace(0, ampl, nmax-nmin+1), np.zeros(n_points_pos_f_incl_zero-nmax-1)))   
                
            else:
                
                spectrum = np.concatenate((np.zeros(int(nmin)), 
                    ampl*np.ones(int(nmax-nmin+1)), np.zeros(int(n_points_pos_f_incl_zero-nmax-1))))       
                    
#            self.totalnoisespectrum = spectrum
#            plt.figure('spectrum noise')
#            plt.plot(freq,spectrum)
#            x1,x2,y1,y2 = plt.axis()
#            plt.axis((0,300,y1,y2)) 
#            plt.pause(0.0001)
            
            noise = PhaseNoise(freq, spectrum, self.seed1, self.seed2)
            noise.spectrum_to_phase_noise()
        
            self.seed1 +=239
            self.seed2 +=158
            
            if i < int(self.n_turns/self.corr):
                self.dphi[(i*self.corr):((i+1)*self.corr)] = noise.dphi[:self.corr]
            else:
                self.dphi[int(self.n_turns/self.corr)*self.corr:] = noise.dphi[:(self.n_turns-int(self.n_turns/self.corr)*self.corr+1)]
            
#            rms_noise = np.std(noise.dphi)
#            print "RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)" \
#                %(noise.t[1], i, rms_noise, rms_noise*180/np.pi)
    def update_freq(self,fs0,fedge):  #temporary function to update the freq targeted by the noise
        if fs0 > fedge:
            self.freqmax = fs0            
            self.freqmin = fedge
        else:
            self.freqmax = fedge
            self.freqmin = fs0
        self.fs0_array = np.append(self.fs0_array,fs0)
        self.fedge_array = np.append(self.fedge_array,fedge)
            
    def track(self):
        if self.turn >= self.noise_time_window[0] and self.turn <= self.noise_time_window[1]:
            if self.turn%self.noise_nturn_recalculation==0:
                self.generate()
                phasenoise = self.dphi
                self.totalnoise[int(self.turn):int(self.turn+self.noise_nturn_recalculation)] += phasenoise[int(self.turn):int(self.turn+self.noise_nturn_recalculation)]  #save the noise for plotting purpose
                self.rf_params.phi_RF[0][int(self.turn):int(self.turn+self.noise_nturn_recalculation)] += phasenoise[int(self.turn):int(self.turn+self.noise_nturn_recalculation)]
        self.turn += 1