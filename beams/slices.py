
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute beam slicing**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import cheb2ord, cheby2, filtfilt, freqz
import ctypes
from setup_cpp import libblond



class Slices(object):
    '''
    *Contains the beam profile and related quantities including beam spectrum,
    profile derivative, and profile fitting.*
    '''
    
    def __init__(self, RFSectionParameters, Beam, n_slices, n_sigma = None, 
                 cut_left = None, cut_right = None, cuts_unit = 's', 
                 fit_option = None, direct_slicing = False, smooth = False):
        
        #: *Import (reference) Beam*
        self.Beam = Beam
        
        #: *Import (reference) RFSectionParameters*
        self.RFParams = RFSectionParameters
        
        #: *Number of slices*
        self.n_slices = n_slices
        
        #: *Left edge of the slicing; optional input in case of 'const_space' 
        #: mode. A default value will be set if no value is given.*
        self.cut_left = cut_left
        
        #: *Right edge of the slicing; optional input in case of 'const_space' 
        #: mode. A default value will be set if no value is given.*
        self.cut_right = cut_right
        
        #: *Optional input parameters, corresponding to the number of*
        #: :math:`\sigma_{RMS}` *of the Beam to slice (this will overwrite
        #: any input of cut_left and cut_right).*
        self.n_sigma = n_sigma
        
        #: | *Unit in which the cuts are given.*
        #: | *The options are: 's' (default) or 'rad' (RF phase).*
        self.cuts_unit = cuts_unit
               
        #: *Number of macro-particles per slice (~profile).*
        self.n_macroparticles = np.zeros(n_slices)
        
        #: *Edges positions of the slicing*
        self.edges = np.zeros(n_slices + 1)
        
        #: *Center of the bins*
        self.bin_centers = np.zeros(n_slices)
        
        # Pre-processing the slicing edges
        self.set_cuts()
        
        #: *Beam spectrum (arbitrary units)*
        self.beam_spectrum = 0
        
        #: *Frequency array corresponding to the beam spectrum in [Hz]*
        self.beam_spectrum_freq = 0
        
        #: *Smooth option produces smoother profiles at the expenses of a 
        #: slower computation time*
        if smooth:
            self.operations = [self._slice_smooth]
        else:
            self.operations = [self._slice]
            
        #: *Fit option allows to fit the Beam profile, with the options
        #: 'None' (default), 'gaussian'.*
        if fit_option is 'gaussian':    
            #: *Beam length with a Gaussian fit (needs fit_option to be 
            #: 'gaussian' defined as* :math:`\tau_{gauss} = 4\sigma_{\Delta t}`)
            self.bl_gauss = 0
            #: *Beam position at the peak of Gaussian fit*
            self.bp_gauss = 0
            #: *Gaussian parameters list obtained from fit*
            self.pfit_gauss = 0
            #: *Performs gaussian_fit each time the track method is called*
            self.operations.append(self.gaussian_fit())
        
        
        # Track at initialisation
        if direct_slicing:
            self.track()


    def set_cuts(self):
        '''
        *Method to set the self.cut_left and self.cut_right properties. This is
        done as a pre-processing.*
        
        *The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the 
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.*
        '''
        
        if self.cut_left is None and self.cut_right is None:
            
            if self.n_sigma is None:
                dt_min = self.Beam.dt.min()
                dt_max = self.Beam.dt.max()
                self.cut_left = dt_min - 0.05 * (dt_max - dt_min)
                self.cut_right = dt_max + 0.05 * (dt_max - dt_min)
            else:
                mean_coords = np.mean(self.Beam.dt)
                sigma_coords = np.std(self.Beam.dt)
                self.cut_left = mean_coords - self.n_sigma*sigma_coords/2
                self.cut_right = mean_coords + self.n_sigma*sigma_coords/2
        else:
            
            self.cut_left = self.convert_coordinates(self.cut_left, 
                                                     self.cuts_unit)
            self.cut_right = self.convert_coordinates(self.cut_right, 
                                                      self.cuts_unit)
            
        self.edges = np.linspace(self.cut_left, self.cut_right, 
                                 self.n_slices + 1)
        self.bin_centers = (self.edges[:-1] + self.edges[1:])/2


    def _slice(self):
        '''
        *Constant space slicing with a constant frame. It does not compute 
        statistics possibilities (the index of the particles is needed).*
        '''
        
        libblond.histogram(self.Beam.dt.ctypes.data_as(ctypes.c_void_p), 
                         self.n_macroparticles.ctypes.data_as(ctypes.c_void_p), 
                         ctypes.c_double(self.cut_left), 
                         ctypes.c_double(self.cut_right), 
                         ctypes.c_int(self.n_slices), 
                         ctypes.c_int(self.Beam.n_macroparticles))
    
    
    def _slice_smooth(self):
        '''
        At the moment 4x slower than _slice but smoother (filtered).
        '''
        
        libblond.smooth_histogram(self.Beam.dt.ctypes.data_as(ctypes.c_void_p), 
                         self.n_macroparticles.ctypes.data_as(ctypes.c_void_p), 
                         ctypes.c_double(self.cut_left), 
                         ctypes.c_double(self.cut_right), 
                         ctypes.c_uint(self.n_slices), 
                         ctypes.c_uint(self.Beam.n_macroparticles))
 
    
    def track(self):
        '''
        *Track method in order to update the slicing along with the tracker.
        This will update the beam properties (bunch length obtained from the
        fit, etc.).*
        '''
        
        for op in self.operations:
            op()
            
        
    def track_cuts(self):
        '''
        *Track the slice frame (limits and slice position) as the mean of the 
        bunch moves.
        Requires Beam statistics!
        Method to be refined!*
        '''

        delta = self.Beam.mean_dt - 0.5*(self.cut_left + self.cut_right)
        
        self.cut_left += delta
        self.cut_right += delta
        self.edges += delta
        self.bin_centers += delta
        

    def gaussian_fit(self):
        '''
        *Gaussian fit of the profile, in order to get the bunch length and
        position. Returns fit values in units of s.*
        '''
        
        if self.bl_gauss is 0 and self.bp_gauss is 0:
            p0 = [max(self.n_macroparticles), np.mean(self.Beam.dt), 
                  np.std(self.Beam.dt)]
        else:
            p0 = [max(self.n_macroparticles), self.bp_gauss, self.bl_gauss/4]
                                                                
        self.pfit_gauss = curve_fit(gauss, self.bin_centers, 
                                    self.n_macroparticles, p0)[0] 
        self.bl_gauss = 4 * abs(self.pfit_gauss[2]) 
        self.bp_gauss = abs(self.pfit_gauss[1])
    
 
    def rms(self):
        '''
        * Computation of the RMS bunch length and position from the line density 
        (bunch length = 4sigma).*
        '''

        timeResolution = self.bin_centers[1]-self.bin_centers[0]
        
        lineDenNormalized = self.n_macroparticles / np.trapz(self.n_macroparticles, dx=timeResolution)
        
        self.bp_rms = np.trapz(self.bin_centers * lineDenNormalized, dx=timeResolution)
        
        self.bl_rms = 4 * np.sqrt(np.trapz((self.bin_centers-self.bp_rms)**2*lineDenNormalized, dx=timeResolution))      
       
       
    def fwhm(self, shift=0):
        '''
        * Computation of the bunch length and position from the FWHM
        assuming Gaussian line density.*
        '''

        half_max = shift + 0.5 * (self.n_macroparticles.max() - shift)
        time_resolution = self.bin_centers[1]-self.bin_centers[0]    
        # First aproximation for the half maximum values
        taux = np.where(self.n_macroparticles>=half_max)
        taux1 = taux[0][0]
        taux2 = taux[0][-1]
        # Interpolation of the time where the line density is half the maximun
        try:
            t1 = self.bin_centers[taux1] - (self.n_macroparticles[taux1]-half_max)/(self.n_macroparticles[taux1]-self.n_macroparticles[taux1-1]) * time_resolution
            t2 = self.bin_centers[taux2] + (self.n_macroparticles[taux2]-half_max)/(self.n_macroparticles[taux2]-self.n_macroparticles[taux2+1]) * time_resolution
            
            self.bl_fwhm = 4 * (t2-t1)/ (2 * np.sqrt(2 * np.log(2)))
            self.bp_fwhm = (t1+t2)/2
        except:
            self.bl_fwhm = np.nan
            self.bp_fwhm = np.nan
    
    def fwhm_multibunch(self, n_bunches, n_slices_per_bunch, bunch_spacing_buckets, bucket_size_tau, bucket_tolerance=0.40):
        '''
        * Computation of the bunch length and position from the FWHM
        assuming Gaussian line density for multibunch case.*
        '''
        
        time_resolution = self.bin_centers[1]-self.bin_centers[0]
        
        self.bl_fwhm = np.zeros(n_bunches)
        self.bp_fwhm = np.zeros(n_bunches)
        
        for indexBunch in range(0,n_bunches):
            
            left_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau - bucket_tolerance * bucket_size_tau
            right_edge = indexBunch * bunch_spacing_buckets * bucket_size_tau + bucket_size_tau + bucket_tolerance * bucket_size_tau
            indexes_bucket = np.where((self.bin_centers > left_edge)*(self.bin_centers < right_edge))[0]
            
            try:
                half_max = 0.5 * self.n_macroparticles[indexes_bucket].max()
        
                # First approximation for the half maximum values
                taux = np.where(self.n_macroparticles[indexes_bucket]>=half_max)
                taux1 = taux[0][0]
                taux2 = taux[0][-1]
                
                # Interpolation of the time where the line density is half the maximum
                t1 = self.bin_centers[indexes_bucket][taux1] - \
                     (self.n_macroparticles[indexes_bucket][taux1]-half_max) / \
                     (self.n_macroparticles[indexes_bucket][taux1] - 
                      self.n_macroparticles[indexes_bucket][taux1-1]) * time_resolution
                
                t2 = self.bin_centers[indexes_bucket][taux2] + \
                     (self.n_macroparticles[indexes_bucket][taux2]-half_max) / \
                     (self.n_macroparticles[indexes_bucket][taux2] - 
                      self.n_macroparticles[indexes_bucket][taux2+1]) * time_resolution
    
                self.bl_fwhm[indexBunch] = 4 * (t2-t1)/ (2 * np.sqrt(2 * np.log(2)))
                self.bp_fwhm[indexBunch] = (t1+t2)/2
            
            except:
                print('Warning: The bunch index %d is empty !!' %(indexBunch))
                self.bl_fwhm[indexBunch] = 0
                self.bp_fwhm[indexBunch] = 0
    
    
    def beam_spectrum_generation(self, n_sampling_fft, only_rfft = False):
        '''
        *Beam spectrum calculation, to be extended (normalized profile, different
        coordinates, etc.)*
        '''
        
        self.beam_spectrum_freq = rfftfreq(n_sampling_fft, (self.bin_centers[1] 
                                           - self.bin_centers[0]))
        
        if not only_rfft:
            self.beam_spectrum = rfft(self.n_macroparticles, n_sampling_fft)         
     
     
    def beam_profile_derivative(self, mode = 'gradient'):      
        ''' 
        *The input is one of the two available methods for differentiating
        a function. The two outputs are the coordinate step and the discrete
        derivative of the Beam profile respectively.*
        '''
        
        
        x = self.bin_centers
        dist_centers = x[1] - x[0]
            
        if mode is 'filter1d':
            derivative = ndimage.gaussian_filter1d(self.n_macroparticles, sigma=1, 
                                                   order=1, mode='wrap') / dist_centers
        elif mode is 'gradient':
            derivative = np.gradient(self.n_macroparticles, dist_centers)
        elif mode is 'diff':
            derivative = np.diff(self.n_macroparticles) / dist_centers
            diffCenters = x[0:-1] + dist_centers/2
            derivative = np.interp(x, diffCenters, derivative)
        else:
            raise RuntimeError('Option for derivative is not recognized.')

        return x, derivative
    
    
    def beam_profile_filter_chebyshev(self, filter_option):      
        ''' 
        *This routine is filtering the beam profile with a type II Chebyshev
        filter. The input is a library having the following structure and
        informations:*
        
        filter_option = {'type':'chebyshev', 'pass_frequency':pass_frequency, 'stop_frequency':stop_frequency, 'gain_pass':gain_pass, 'gain_stop':gain_stop}
        
        *The function returns nCoefficients, the number of coefficients used 
        in the filter. You can also add the following option to plot and return
        the filter transfer function:*
        
        filter_option = {..., 'transfer_function_plot':True}
        '''
        
        noisyProfile = np.array(self.n_macroparticles)
        
        freqSampling = 1 / (self.bin_centers[1] - self.bin_centers[0])
        nyqFreq = freqSampling / 2.
        
        frequencyPass = filter_option['pass_frequency'] / nyqFreq
        frequencyStop = filter_option['stop_frequency'] / nyqFreq
        gainPass = filter_option['gain_pass']
        gainStop = filter_option['gain_stop']
        
        # Compute the lowest order for a Chebyshev Type II digital filter
        nCoefficients, wn = cheb2ord(frequencyPass, frequencyStop, gainPass, gainStop)
        
        # Compute the coefficients a Chebyshev Type II digital filter
        b,a = cheby2(nCoefficients, gainStop, wn, btype='low')
        
        # Apply the filter forward and backwards to cancel the group delay
        self.n_macroparticles = filtfilt(b, a, noisyProfile)
        self.n_macroparticles = np.ascontiguousarray(self.n_macroparticles)
        
        if 'transfer_function_plot' in filter_option and filter_option['transfer_function_plot']==True:
            # Plot the filter transfer function
            w, transferGain = freqz(b, a=a, worN=self.n_slices)
            transferFreq = w / np.pi * nyqFreq
            group_delay = -np.diff(-np.unwrap(-np.angle(transferGain))) / -np.diff(w*freqSampling) 
                       
            plt.figure()
            ax1 = plt.subplot(311)
            plt.plot(transferFreq, 20 * np.log10(abs(transferGain)))
            plt.ylabel('Magnitude [dB]')
            plt.subplot(312,sharex=ax1)
            plt.plot(transferFreq, np.unwrap(-np.angle(transferGain)))
            plt.ylabel('Phase [rad]')
            plt.subplot(313,sharex=ax1)
            plt.plot(transferFreq[:-1], group_delay)
            plt.ylabel('Group delay [s]')
            plt.xlabel('Frequency [Hz]')
                        
            ## Plot the bunch spectrum and the filter transfer function
            plt.figure()
            plt.plot(np.fft.fftfreq(self.n_slices, self.bin_centers[1]-self.bin_centers[0]), 20.*np.log10(np.abs(np.fft.fft(noisyProfile))))
            plt.xlabel('Frequency [Hz]')
            plt.twinx()
            plt.plot(transferFreq, 20 * np.log10(abs(transferGain)),'r')
            plt.xlim(0,plt.xlim()[1])
            
            plt.show()
            
            return nCoefficients, [transferFreq, transferGain]
        
        else:
            return nCoefficients, b, a
  
  
    def convert_coordinates(self, value, input_unit_type):
        '''
        *Method to convert a value from one input_unit_type to 's'.*
        '''
        
        if input_unit_type is 's':
            return value
        
        elif input_unit_type is 'rad':
            return value /\
                self.RFParams.omega_RF[0,self.RFParams.counter[0]]


    ### DEPRECATED METHODS. TO BE REMOVED IN A FUTURE VERSION ###
    def slice_constant_space_histogram(self, beam_dt):
        print('DEPRECATED METHOD: slice_constant_space_histogram method has \
               been replaced by the _slice() method')
        self._slice()
    
    
    def slice_constant_space_histogram_smooth(self):
        print('DEPRECATED METHOD: slice_constant_space_histogram_smooth method \
               has been replaced by the _slice_smooth() method')
        self._slice_smooth()
            
 
def gauss(x, *p):
    '''
    *Defined as:*
    
    .. math:: A \, e^{\\frac{\\left(x-x_0\\right)^2}{2\\sigma_x^2}}
    
    '''
    
    A, x0, sx = p
    return A*np.exp(-(x-x0)**2/2./sx**2) 