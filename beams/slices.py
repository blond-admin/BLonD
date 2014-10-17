'''
**Module to compute longitudinal beam slicing**

:Authors: **Hannes Bartosik**, **Kevin Li**, **Michael Schenk**, **Danilo Quartullo**, **Alexandre Lasheen**
'''

from __future__ import division
import numpy as np
from random import sample
from scipy.constants import c
from numpy.fft import rfft, rfftfreq
from scipy import ndimage
from scipy.optimize import curve_fit
import warnings
import ctypes
from setup_cpp import libfib

class Slices(object):
    '''
    *Slices class that controls longitudinal discretisation of a Beam. This
    include the Beam profiling (including computation of Beam spectrum,
    derivative, and profile fitting) and the computation of statistics per
    slice.*
    '''
    
    def __init__(self, Beam, n_slices, n_sigma = None, cut_left = None, 
                 cut_right = None, cuts_coord = 'tau', slicing_coord = 'tau', 
                 mode = 'const_space', statistics_option = 'off', fit_option = 'off', slice_immediately = 'off'):
        
        #: *Copy (reference) of the beam to be sliced (from Beam)*
        self.Beam = Beam
        
        #: *Number of slices*
        self.n_slices = n_slices
        
        #: | *Slicing computation mode*
        #: | *The options are: 'const_space' (default), 'const_space_hist' and 
        #: 'const_charge'.*
        self.mode = mode
        
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
        self.n_macroparticles = np.zeros(n_slices, dtype='uint32')
        
        #: *Edges positions of the slicing*
        self.edges = np.zeros(n_slices + 1)
        
        #: *Center of the bins*
        self.bins_centers = np.zeros(n_slices)
        
        # Pre-processing the slicing edges
        self.set_longitudinal_cuts()
        
        #: *Compute statistics option allows to compute mean_theta, mean_dE, 
        #: sigma_theta and sigma_dE properties each turn.*
        self.statistics_option = statistics_option
        
        if self.statistics_option is 'on' and self.mode is 'const_space_hist':
            raise RuntimeError('Slice compute statistics does not work with \
                                the const_space_hist mode !')
        elif self.statistics_option is 'on':
            #: *Average theta position of the particles in each slice (needs 
            #: the compute_statistics_option to be 'on').*
            self.mean_theta = np.zeros(n_slices)
            #: *Average dE position of the particles in each slice (needs 
            #: the compute_statistics_option to be 'on').*
            self.mean_dE = np.zeros(n_slices)
            #: *RMS theta position of the particles in each slice (needs 
            #: the compute_statistics_option to be 'on').*
            self.sigma_theta = np.zeros(n_slices)
            #: *RMS dE position of the particles in each slice (needs 
            #: the compute_statistics_option to be 'on').*
            self.sigma_dE = np.zeros(n_slices)
            #: *RMS dE position of the particles in each slice (needs 
            #: the compute_statistics_option to be 'on').*
            self.eps_rms_l = np.zeros(n_slices)
        
        #: *Fit option allows to fit the Beam profile, with the options
        #: 'off' (default), 'gaussian'.*
        self.fit_option = fit_option
        
        #: *Beam spectrum (arbitrary units)*
        self.beam_spectrum = 0
        
        #: *Frequency array corresponding to the beam spectrum in [Hz]*
        self.beam_spectrum_freq = 0
        
        if self.mode is not 'const_charge' and self.fit_option is 'gaussian':    
            #: *Beam length with a gaussian fit (needs fit_option to be 
            #: 'gaussian' defined as* :math:`\tau_{gauss} = 4\sigma`)
            self.bl_gauss = 0
            #: *Beam position with a gaussian fit (needs fit_option to be 
            #: 'gaussian')*
            self.bp_gauss = 0
            #: *Gaussian parameters list obtained from fit*
            self.pfit_gauss = 0
                  
        # Use of track in order to pre-process the slicing at injection
        if slice_immediately == 'on':
            self.track(self.Beam)
          
        
    def sort_particles(self):
        '''
        *Sort the particles with respect to their longitudinal position.*
        '''
        
        if (self.slicing_coord is 'tau') or (self.slicing_coord is 'theta'):
            argsorted = np.argsort(self.Beam.theta)
        elif self.slicing_coord is 'z':
            argsorted = np.argsort(self.Beam.z)
                    
        self.Beam.theta = self.Beam.theta.take(argsorted)
        self.Beam.dE = self.Beam.dE.take(argsorted)
        self.Beam.id = self.Beam.id.take(argsorted)
        

    def set_longitudinal_cuts(self):
        '''
        *Method to set the self.cut_left and self.cut_right properties. This is
        done as a pre-processing if the mode is set to 'const_space', for
        'const_charge' this is calculated each turn.*
        
        *The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the 
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.*
        '''

        
        if self.mode is not 'const_charge':
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
            
        else:
                mean_beam_coordinates = np.mean(self.beam_coordinates)
                sigma_beam_coordinates = np.std(self.beam_coordinates)
                self.cut_left = mean_beam_coordinates - self.n_sigma * sigma_beam_coordinates / 2
                self.cut_right = mean_beam_coordinates + self.n_sigma * sigma_beam_coordinates / 2


    def slice_constant_space(self):
        '''
        *Constant space slicing. This method consist in slicing a fixed frame
        (which length is determined in the beginning of the simulation) with
        bins of constant size. Each turn, the particles are sorted with respect
        to their longitudinal position and counted in each bin. This allows
        also to calculate the statistics of the particles for each bin (if 
        statistics_option is 'on') and fit the profile (e.g. Gaussian).*
        
        *Be careful that because the frame is not changing, a bunch with
        increasing bunch length might not be sliced properly as part of it
        might be out of the frame.*
        '''

        self.sort_particles()
        
        first_index_in_bin = np.searchsorted(self.beam_coordinates, self.edges)
            
        self.n_macroparticles = np.diff(first_index_in_bin)
        

    def slice_constant_space_histogram(self):
        '''
        *Constant space slicing with the built-in numpy histogram function,
        with a constant frame. This gives the same profile as the 
        slice_constant_space method, but no compute statistics possibilities
        (the index of the particles is needed).*
        
        *This method is faster than the classic slice_constant_space method 
        for high number of particles (~1e6).*
        '''
        w = self.beam_coordinates
        v = self.n_macroparticles
        libfib.histogram(w.ctypes.data_as(ctypes.c_void_p), 
                      v.ctypes.data_as(ctypes.c_void_p), 
               ctypes.c_double(self.cut_left), ctypes.c_double(self.cut_right), 
               ctypes.c_uint(self.n_slices), ctypes.c_uint(self.Beam.n_macroparticles))

        
    def slice_constant_charge(self):
        '''
        *Constant charge slicing. This method consist in slicing with varying
        bin sizes that adapts in order to have the same number of particles
        in each bin*
         
        *Must be updated in order to take into account potential losses (in order
        for the frame size not to diverge).*
        '''
         
        self.set_longitudinal_cuts()
         
        # 1. n_macroparticles - distribute macroparticles uniformly along slices.
        # Must be integer. Distribute remaining particles randomly among slices with indices 'ix'.
        n_cut_left = 0 # number of particles cut left, to be adapted for losses
        n_cut_right = 0 # number of particles cut right, to be adapted for losses
           
        q0 = self.Beam.n_macroparticles - n_cut_right - n_cut_left
          
        ix = sample(range(self.n_slices), q0 % self.n_slices)
        self.n_macroparticles = (q0 // self.n_slices) * np.ones(self.n_slices)
        self.n_macroparticles[ix] += 1
         
        # 2. edges
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack((n_cut_left, self.n_macroparticles, n_cut_right))
        first_index_in_bin = np.cumsum(n_macroparticles_all)
        first_particle_index_in_slice = first_index_in_bin[:-1]
        first_particle_index_in_slice = (first_particle_index_in_slice).astype(int)
          
        self.edges[1:-1] = (self.beam_coordinates[(first_particle_index_in_slice - 1)[1:-1]] + 
                            self.beam_coordinates[first_particle_index_in_slice[1:-1]]) / 2
        self.edges[0], self.edges[-1] = self.cut_left, self.cut_right
        self.bins_centers = (self.edges[:-1] + self.edges[1:]) / 2
        
        
    def track(self, Beam):
        '''
        *Track method in order to update the slicing along with the tracker.
        This will update the beam properties (bunch length obtained from the
        fit, etc.).*
        '''
        
        if self.mode == 'const_charge':
            self.slice_constant_charge()
        elif self.mode == 'const_space':
            self.slice_constant_space()
        elif self.mode == 'const_space_hist':
            self.slice_constant_space_histogram()
        else:
            raise RuntimeError('Choose one proper slicing mode!')
        
        if self.fit_option is 'gaussian':
            self.gaussian_fit()
            self.Beam.bl_gauss = self.convert_coordinates(self.bl_gauss, self.slicing_coord, 'theta')
            self.Beam.bp_gauss = self.convert_coordinates(self.bp_gauss, self.slicing_coord, 'theta')
           
        if self.statistics_option is 'on':
            self.compute_statistics()
        
        
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
        else:
            raise RuntimeError('Option for derivative is not recognized.')
            
#         elif mode is 'butterworth_filter':
#             pass
             
        return x, derivative
    
    
    def compute_statistics(self):
        '''
        *Compute statistics of each slice (average position of the particles
        in a slice and sigma_rms. Notice that empty slices will
        result with NaN values for the statistics but that doesn't cause any
        problem.*
        
        *Improvement is needed in order to include losses, and link with 
        transverse statistics calculation.*
        '''
        warnings.filterwarnings("ignore")
        index = np.cumsum(np.append(0, self.n_macroparticles))

        for i in xrange(self.n_slices):
           
            theta  = self.Beam.theta[index[i]:index[i + 1]]
            dE = self.Beam.dE[index[i]:index[i + 1]]

            self.mean_theta[i] = np.mean(theta)
            self.mean_dE[i] = np.mean(dE)

            self.sigma_theta[i] = np.std(theta)
            self.sigma_dE[i] = np.std(dE)

            self.eps_rms_l[i] = np.pi * self.sigma_dE[i] * self.sigma_theta[i] \
                                * self.Beam.ring_radius / (self.Beam.beta_r * c)
        warnings.filterwarnings("always")
                                
                                
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
    
        
