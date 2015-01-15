
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute beam slicing**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Juan Esteban Muller**
'''

from __future__ import division
import numpy as np
from random import sample
from scipy.constants import c
from numpy.fft import rfft, rfftfreq
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import cheb2ord, cheby2, filtfilt, freqz
import matplotlib.pyplot as plt
import warnings
import ctypes
from setup_cpp import libfib


class Slices(object):
    '''
    *Slices class that controls discretisation of a Beam. This
    include the Beam profiling (including computation of Beam spectrum,
    derivative, and profile fitting).*
    '''
    
    def __init__(self, Beam, n_slices, n_sigma = None, cut_left = None, 
                 cut_right = None, cuts_coord = 'tau', slicing_coord = 'tau', 
                 fit_option = 'off', filter_option = {'type':'off'}):
        
        #: *Copy (reference) of the beam to be sliced (from Beam)*
        self.Beam = Beam
        
        #: *Number of slices*
        self.n_slices = n_slices
        
        #: *Left edge of the slicing (is an optional input, in case you use
        #: the 'const_space' mode, a default value will be set if no value is 
        #: given).*
        self.cut_left = cut_left
        
        #: *Right edge of the slicing (is an optional input, in case you use
        #: the 'const_space' mode, a default value will be set if no value is 
        #: given).*
        self.cut_right = cut_right
        
        #: *Optional input parameters, corresponding to the number of*
        #: :math:`\sigma_{RMS}` *of the Beam to slice (this will overwrite
        #: any input of cut_left and cut_right).*
        self.n_sigma = n_sigma
        
        #: | *Type of coordinates in which the cuts are given.*
        #: | *The options are: 'tau' (default), 'theta', 'z'.*
        self.cuts_coord = cuts_coord
        
        #: | *Type of coordinates in which the slicing is done.*
        #: | *The options are: 'tau' (default), 'theta', 'z'.*
        self.slicing_coord = slicing_coord
        
        if (self.slicing_coord is not 'tau') and (self.slicing_coord is not 'z') and (self.slicing_coord is not 'theta') :
            raise RuntimeError('The slicing_coord is not recognized')
        
        #: *Number of macroparticles per slice (~profile).*
        self.n_macroparticles = np.zeros(n_slices)
        
        #: *Edges positions of the slicing*
        self.edges = np.zeros(n_slices + 1)
        
        #: *Center of the bins*
        self.bins_centers = np.zeros(n_slices)
        
        # Pre-processing the slicing edges
        self.set_cuts()
        
        #: *Fit option allows to fit the Beam profile, with the options
        #: 'off' (default), 'gaussian'.*
        self.fit_option = fit_option
        
        #: *Filter option*
        self.filter_option = filter_option
        
        #: *Beam spectrum (arbitrary units)*
        self.beam_spectrum = 0
        
        #: *Frequency array corresponding to the beam spectrum in [Hz]*
        self.beam_spectrum_freq = 0
        
        if self.fit_option is 'gaussian':    
            #: *Beam length with a gaussian fit (needs fit_option to be 
            #: 'gaussian' defined as* :math:`\tau_{gauss} = 4\sigma`)
            self.bl_gauss = 0
            #: *Beam position with a gaussian fit (needs fit_option to be 
            #: 'gaussian')*
            self.bp_gauss = 0
            #: *Gaussian parameters list obtained from fit*
            self.pfit_gauss = 0
                  
        
    def sort_particles(self):
        '''
        *Sort the particles with respect to their position.*
        '''
        
        if (self.slicing_coord is 'tau') or (self.slicing_coord is 'theta'):
            argsorted = np.argsort(self.Beam.theta)
        elif self.slicing_coord is 'z':
            argsorted = np.argsort(self.Beam.z)
                    
        self.Beam.theta = self.Beam.theta.take(argsorted)
        self.Beam.dE = self.Beam.dE.take(argsorted)
        self.Beam.id = self.Beam.id.take(argsorted)
        

    def set_cuts(self):
        '''
        *Method to set the self.cut_left and self.cut_right properties. This is
        done as a pre-processing if the mode is set to 'const_space', for
        'const_charge' this is calculated each turn.*
        
        *The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the 
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.*
        '''

        
        if self.cut_left is None and self.cut_right is None:
            
            if self.n_sigma is None:
                self.sort_particles()
                self.cut_left = self.beam_coordinates[0] - 0.05*(self.beam_coordinates[-1] - self.beam_coordinates[0])
                self.cut_right = self.beam_coordinates[-1] + 0.05*(self.beam_coordinates[-1] - self.beam_coordinates[0])
            else:
                mean_beam_coordinates = np.mean(self.beam_coordinates)
                sigma_beam_coordinates = np.std(self.beam_coordinates)
                self.cut_left = mean_beam_coordinates - self.n_sigma * sigma_beam_coordinates / 2
                self.cut_right = mean_beam_coordinates + self.n_sigma * sigma_beam_coordinates / 2
        else:
            
            self.cut_left = self.convert_coordinates(self.cut_left, self.cuts_coord, self.slicing_coord)
            self.cut_right = self.convert_coordinates(self.cut_right, self.cuts_coord, self.slicing_coord)
        
        self.edges = np.linspace(self.cut_left, self.cut_right, self.n_slices + 1)
        self.bins_centers = (self.edges[:-1] + self.edges[1:]) / 2


    def slice_constant_space_histogram(self):
        '''
        *Constant space slicing with the built-in numpy histogram function,
        with a constant frame. This gives the same profile as the 
        slice_constant_space method, but no compute statistics possibilities
        (the index of the particles is needed).*
        
        *This method is faster than the classic slice_constant_space method 
        for high number of particles (~1e6).*
        '''
        
        beam_hist = self.beam_coordinates
        macro_hist = self.n_macroparticles
        libfib.histogram(beam_hist.ctypes.data_as(ctypes.c_void_p), 
                      macro_hist.ctypes.data_as(ctypes.c_void_p), 
               ctypes.c_double(self.cut_left), ctypes.c_double(self.cut_right), 
               ctypes.c_uint(self.n_slices), ctypes.c_uint(self.Beam.n_macroparticles))

        
 
    def track(self, Beam):
        '''
        *Track method in order to update the slicing along with the tracker.
        This will update the beam properties (bunch length obtained from the
        fit, etc.).*
        '''
        
        self.slice_constant_space_histogram()
        
        if self.fit_option is 'gaussian':
            self.gaussian_fit()
            self.Beam.bl_gauss = self.convert_coordinates(self.bl_gauss, self.slicing_coord, 'theta')
            self.Beam.bp_gauss = self.convert_coordinates(self.bp_gauss, self.slicing_coord, 'theta')
            
        if self.filter_option['type'] is 'chebyshev':
            self.beam_profile_filter_chebyshev(self.filter_option)
        
        
    def gaussian_fit(self):
        '''
        *Gaussian fit of the profile, in order to get the bunch length and
        position.*
        '''
            
        if self.bl_gauss is 0 and self.bp_gauss is 0:
            p0 = [max(self.n_macroparticles), np.mean(self.beam_coordinates), np.std(self.beam_coordinates)]
        else:
            p0 = [max(self.n_macroparticles), self.bp_gauss, self.bl_gauss/4]
                                                                
        self.pfit_gauss = curve_fit(gauss, self.bins_centers, self.n_macroparticles, p0)[0] 
        self.bl_gauss = 4 * abs(self.pfit_gauss[2]) 
        self.bp_gauss = abs(self.pfit_gauss[1])
    
       
    def fwhm(self):
        '''
        * Computation of the bunch length and position from the FWHM
        assuming Gaussian line density.*
        '''

        half_max = 0.5 * self.n_macroparticles.max()
        time_resolution = self.bins_centers[1]-self.bins_centers[0]    
        # First aproximation for the half maximum values
        taux = np.where(self.n_macroparticles>=half_max)
        taux1 = taux[0][0]
        taux2 = taux[0][-1]
        # Interpolation of the time where the line density is half the maximun
        t1 = self.bins_centers[taux1] - (self.n_macroparticles[taux1]-half_max)/(self.n_macroparticles[taux1]-self.n_macroparticles[taux1-1]) * time_resolution
        t2 = self.bins_centers[taux2] + (self.n_macroparticles[taux2]-half_max)/(self.n_macroparticles[taux2]-self.n_macroparticles[taux2+1]) * time_resolution
        
        self.bl_fwhm = 4 * (t2-t1)/ (2 * np.sqrt(2 * np.log(2)))
        self.bp_fwhm = (t1+t2)/2
    
    
    def beam_spectrum_generation(self, n_sampling_fft, filter_option = None, only_rfft = False):
        '''
        *Beam spectrum calculation, to be extended (normalized profile, different
        coordinates, etc.)*
        '''
        
        time_step = self.convert_coordinates(self.bins_centers[1] - self.bins_centers[0], self.slicing_coord, 'tau')
        self.beam_spectrum_freq = rfftfreq(n_sampling_fft, time_step)
        
        if not only_rfft:
            self.beam_spectrum = rfft(self.n_macroparticles, n_sampling_fft)             
     
     
    def beam_profile_derivative(self, mode = 'gradient', coord = 'theta'):      
        ''' 
        *The input is one of the two available methods for differentiating
        a function. The two outputs are the coordinate step and the discrete
        derivative of the Beam profile respectively.*
        '''
            
        x = self.bins_centers
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
        
        freqSampling = 1 / (self.bins_centers[1] - self.bins_centers[0])
        nyqFreq = freqSampling / 2.
        
        frequencyPass = filter_option['pass_frequency'] / nyqFreq
        frequencyStop = filter_option['stop_frequency'] / nyqFreq
        gainPass = filter_option['gain_pass']
        gainStop = filter_option['gain_stop']
        
        # Compute the lowest order for a Chebyshev Type II digital filter
        nCoefficients, wn = cheb2ord(frequencyPass, frequencyStop, gainPass, gainStop)
        print nCoefficients
        
        # Compute the coefficients a Chebyshev Type II digital filter
        b,a = cheby2(nCoefficients, gainStop, wn, btype='low')
        
        # Apply the filter forward and backwards to cancel the group delay
        self.n_macroparticles = filtfilt(b, a, noisyProfile)
        
        if 'transfer_function_plot' in filter_option:
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
            plt.plot(np.fft.fftfreq(self.n_slices, self.bins_centers[1]-self.bins_centers[0]), 20.*np.log10(np.abs(np.fft.fft(noisyProfile))))
            plt.xlabel('Frequency [Hz]')
            plt.twinx()
            plt.plot(transferFreq, 20 * np.log10(abs(transferGain)),'r')
            plt.xlim(0,plt.xlim()[1])
            
            plt.show()
            
            return nCoefficients, [transferFreq, transferGain]
        
        else:
            return nCoefficients
  
  
    def convert_coordinates(self, value, input_coord_type, output_coord_type):
        '''
        *Method to convert a value from one input_coord_type to an output_coord_type.*
        '''
        
        if input_coord_type is output_coord_type:
            return value
        
        if input_coord_type is 'tau':
            if output_coord_type is 'theta':
                return value / (self.Beam.ring_radius / (self.Beam.beta_r * c))
            elif output_coord_type is 'z':
                return  - value * (self.Beam.beta_r * c)
        elif input_coord_type is 'theta':
            if output_coord_type is 'tau':
                return value * self.Beam.ring_radius / (self.Beam.beta_r * c)
            elif output_coord_type is 'z':
                return - value * self.Beam.ring_radius
        elif input_coord_type is 'z':
            if output_coord_type is 'theta':
                return - value / self.Beam.ring_radius
            elif output_coord_type is 'tau':
                return  - value / (self.Beam.beta_r * c)
            
            
    @property
    def beam_coordinates(self):
        '''
        *Returns the beam coordinates according to the slicing_coord option.*
        '''
        if self.slicing_coord is 'tau':
            return self.Beam.tau
        elif self.slicing_coord is 'theta':
            return self.Beam.theta
        elif self.slicing_coord is 'z':
            return self.Beam.z


def gauss(x, *p):
    '''
    *Defined as:*
    
    .. math:: A \, e^{\\frac{\\left(x-x_0\\right)^2}{2\\sigma_x^2}}
    
    '''
    
    A, x0, sx = p
    return A*np.exp(-(x-x0)**2/2./sx**2) 
    
        
