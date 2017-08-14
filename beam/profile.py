
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute beam slicing**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, 
          **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import object
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy import ndimage
import ctypes
from setup_cpp import libblond
import toolbox.filters_and_fitting as ffroutines


class CutOptions(object):
    
    
    def __init__(self, cut_left=None, cut_right=None, n_slices=100, n_sigma=None, 
                cuts_unit = 's', omega_RF=None):
        
        self.cut_left = cut_left
        self.cut_right = cut_right
        self.n_slices = n_slices
        self.n_sigma = n_sigma
        self.cuts_unit = cuts_unit
        self.omega_RF = omega_RF
        self.edges = np.zeros(n_slices + 1)
        self.bin_centers = np.zeros(n_slices)
        
    
    def set_cuts(self, Beam=None):
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
                dt_min = Beam.dt.min()
                dt_max = Beam.dt.max()
                self.cut_left = dt_min - 0.05 * (dt_max - dt_min)
                self.cut_right = dt_max + 0.05 * (dt_max - dt_min)
            else:
                mean_coords = np.mean(Beam.dt)
                sigma_coords = np.std(Beam.dt)
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
    
    
    def track_cuts(self, Beam):
        '''
        *Track the slice frame (limits and slice position) as the mean of the 
        bunch moves.
        Requires Beam statistics!
        Method to be refined!*
        '''

        delta = Beam.mean_dt - 0.5*(self.cut_left + self.cut_right)
        
        self.cut_left += delta
        self.cut_right += delta
        self.edges += delta
        self.bin_centers += delta
    
    
    def convert_coordinates(self, value, input_unit_type):
        '''
        *Method to convert a value from one input_unit_type to 's'.*
        '''
        
        if input_unit_type is 's':
            return value
        
        elif input_unit_type is 'rad':
            return value /\
                self.omega_RF
                

class FitOptions(object):
    
    def __init__(self, fitMethod=None, fitExtraOptions=None):
        
        self.fitMethod = fitMethod
        self.fitExtraOptions = fitExtraOptions


class FilterOptions(object):
    
    def __init__(self, filterMethod=None, filterExtraOptions=None):
        
        self.filterMethod = filterMethod
        self.filterExtraOptions = filterExtraOptions


class OtherSlicesOptions(object):
    
    def __init__(self, smooth=False, direct_slicing = False):
        
        self.smooth = smooth
        self.direct_slicing = direct_slicing
        


class Profile(object):
    '''
    *Contains the beam profile and related quantities including beam spectrum,
    profile derivative, and profile fitting.*
    '''

    def __init__(self, Beam,
                 CutOptions = CutOptions(),
                 FitOptions= FitOptions(),
                 FilterOptions=FilterOptions(), 
                 OtherSlicesOptions = OtherSlicesOptions()):
        
        #: *Import (reference) Beam*
        self.Beam = Beam
        
        # Pre-processing the slicing edges
        CutOptions.set_cuts(self.Beam)
        
        #: *Number of slices*
        self.n_slices = CutOptions.n_slices
        
        #: *Left edge of the slicing (optional). A default value will be set if
        #: no value is given.*
        self.cut_left = CutOptions.cut_left

        #: *Right edge of the slicing (optional). A default value will be set
        #: if no value is given.*
        self.cut_right = CutOptions.cut_right

        #: *Optional input parameters, corresponding to the number of*
        #: :math:`\sigma_{RMS}` *of the Beam to slice (this will overwrite
        #: any input of cut_left and cut_right).*
        self.n_sigma = CutOptions.n_sigma

        self.edges = CutOptions.edges
        self.bin_centers = CutOptions.bin_centers
        
        # Bin size
        self.bin_size = (self.cut_right - self.cut_left) / self.n_slices
        
        #: *Number of macro-particles per slice (~profile).*
        self.n_macroparticles = np.zeros(self.n_slices)
        
        #: *Beam spectrum (arbitrary units)*
        self.beam_spectrum = 0

        #: *Frequency array corresponding to the beam spectrum in Hz*
        self.beam_spectrum_freq = 0

        #: *Smooth option produces smoother profiles at the expenses of a 
        #: slower computation time*
        if OtherSlicesOptions.smooth:
            self.operations = [self._slice_smooth]
        else:
            self.operations = [self._slice]
        
        #: *Fit option allows to fit the Beam profile
        if FitOptions.fitMethod!=None:
            self.bunchPosition = 0
            self.bunchLength = 0
            if FitOptions.fitMethod == 'gaussian':
                self.operations.append(self.apply_fit)
            elif FitOptions.fitMethod == 'rms':
                self.operations.append(self.rms)
            elif FitOptions.fitMethod == 'fwhm':
                self.operations.append(self.fwhm)    
                
        if FilterOptions.filterMethod=='chebishev':
            self.filterExtraOptions = FilterOptions.filterExtraOptions
            self.operations.append(self.apply_filter)
        
        # Track at initialisation
        if OtherSlicesOptions.direct_slicing:
            self.track()
    
    
    def track(self):
        '''
        *Track method in order to update the slicing along with the tracker.
        This will update the beam properties (bunch length obtained from the
        fit, etc.).*
        '''
        
        for op in self.operations:
            op()
            

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
    
    
    def apply_fit(self):
        
        
        if self.bunchLength is 0:
            p0 = [max(self.n_macroparticles), np.mean(self.Beam.dt), 
                  np.std(self.Beam.dt)]
        else:
            p0 = [max(self.n_macroparticles), self.bunchPosition, self.bunchLength/4]
            
        self.fitExtraOptions = ffroutines.gaussian_fit(self.n_macroparticles, self.bin_centers, 
                            p0)
        
        self.bunchPosition = self.fitExtraOptions[1]
        self.bunchLength = 4*self.fitExtraOptions[2]
        
    
    def apply_filter(self):
        
        self.n_macroparticles = ffroutines.beam_profile_filter_chebyshev(self.n_macroparticles, 
                            self.bin_centers, self.filterExtraOptions)
        
    
    def rms(self):
        '''
        *Computation of the RMS bunch length and position from the line
        density (bunch length = 4sigma).*
        '''

        self.bunchPosition, self.bunchLength = ffroutines.rms(self.n_macroparticles, self.bin_centers)
    
    
    def fwhm(self, shift=0):
        '''
        *Computation of the bunch length and position from the FWHM
        assuming Gaussian line density.*
        '''

        self.bunchPosition, self.bunchLength = ffroutines.fwhm(self.n_macroparticles, self.bin_centers, shift)
    
    
    def fwhm_multibunch(self, n_bunches, n_slices_per_bunch,
                        bunch_spacing_buckets, bucket_size_tau,
                        bucket_tolerance=0.40, shift=0):
        '''
        *Computation of the bunch length and position from the FWHM
        assuming Gaussian line density for multibunch case.*
        '''
        
        self.bunchPosition, self.bunchLength = ffroutines.fwhm_multibunch(self.n_macroparticles, 
                        self.bin_centers, n_bunches, n_slices_per_bunch,
                        bunch_spacing_buckets, bucket_size_tau,
                        bucket_tolerance, shift)
    
    
    def beam_spectrum_freq_generation(self, n_sampling_fft):
        '''
        *Frequency array of the beam spectrum*
        '''
        
        self.beam_spectrum_freq = rfftfreq(n_sampling_fft, self.bin_size)
        
    
    def beam_spectrum_generation(self, n_sampling_fft):
        '''
        *Beam spectrum calculation*
        '''
        
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
            derivative = ndimage.gaussian_filter1d(self.n_macroparticles, 
                                sigma=1, order=1, mode='wrap') / dist_centers
        elif mode is 'gradient':
            derivative = np.gradient(self.n_macroparticles, dist_centers)
        elif mode is 'diff':
            derivative = np.diff(self.n_macroparticles) / dist_centers
            diffCenters = x[0:-1] + dist_centers/2
            derivative = np.interp(x, diffCenters, derivative)
        else:
            raise RuntimeError('Option for derivative is not recognized.')

        return x, derivative
