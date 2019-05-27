# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute the beam profile through slices**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, 
          **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import object
import numpy as np
# from numpy.fft import rfft, rfftfreq
from scipy import ndimage
import ctypes
from ..toolbox import filters_and_fitting as ffroutines
from ..utils import bmath as bm

class CutOptions(object):
    r"""
    This class groups all the parameters necessary to slice the phase space
    distribution according to the time axis, apart from the array collecting
    the profile which is defined in the constructor of the class Profile below.

    Parameters
    ----------
    cut_left : float
        Left edge of the slicing (optional). A default value will be set if
        no value is given.
    cut_right : float
        Right edge of the slicing (optional). A default value will be set
        if no value is given.
    n_slices : int
        Optional input parameters, corresponding to the number of
        :math:`\sigma_{RMS}` of the Beam to slice (this will overwrite
        any input of cut_left and cut_right).
    n_sigma : float
        defines the left and right extremes of the profile in case those are
        not given explicitly
    cuts_unit : str
        the unit of cut_left and cut_right, it can be seconds 's' or radians
        'rad'
    RFSectionParameters : object
        RFSectionParameters[0][0] is necessary for the conversion from radians
        to seconds if cuts_unit = 'rad'. RFSectionParameters[0][0] is the value
        of omega_rf of the main harmonic at turn number 0

    Attributes
    ----------
    cut_left : float
    cut_right : float
    n_slices : int
    n_sigma : float
    cuts_unit : str
    RFSectionParameters : object
    edges : float array
        contains the edges of the slices
    bin_centers : float array
        contains the centres of the slices

    Examples
    --------
    >>> from input_parameters.ring import Ring
    >>> from input_parameters.rf_parameters import RFStation
    >>> self.ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> self.rf_params = RFStation(Ring=self.ring, n_rf=1, harmonic=[4620],
    >>>                  voltage=[7e6], phi_rf_d=[0.])
    >>> CutOptions = profileModule.CutOptions(cut_left=0, cut_right=2*np.pi,
    >>> n_slices = 100, cuts_unit='rad', RFSectionParameters=self.rf_params)

    """

    def __init__(self, cut_left=None, cut_right=None, n_slices=100,
                 n_sigma=None, cuts_unit='s', RFSectionParameters=None):
        """
        Constructor
        """

        if cut_left is not None:
            self.cut_left = float(cut_left)
        else:
            self.cut_left = cut_left

        if cut_right is not None:
            self.cut_right = float(cut_right)
        else:
            self.cut_right = cut_right

        self.n_slices = int(n_slices)

        if n_sigma is not None:
            self.n_sigma = float(n_sigma)
        else:
            self.n_sigma = n_sigma

        self.cuts_unit = str(cuts_unit)

        self.RFParams = RFSectionParameters

        if self.cuts_unit == 'rad' and self.RFParams is None:
            #CutError
            raise RuntimeError('You should pass an RFParams object to ' +
                               'convert from radians to seconds')
        if self.cuts_unit != 'rad' and self.cuts_unit != 's':
            #CutError
            raise RuntimeError('cuts_unit should be "s" or "rad"')

        self.edges = np.zeros(n_slices + 1, dtype=float)
        self.bin_centers = np.zeros(n_slices, dtype=float)

    def set_cuts(self, Beam=None):
        """
        Method to set self.cut_left, self.cut_right, self.edges and
        self.bin_centers attributes.
        The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.
        """

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

            self.cut_left = float(self.convert_coordinates(self.cut_left,
                                                     self.cuts_unit))
            self.cut_right = float(self.convert_coordinates(self.cut_right,
                                                      self.cuts_unit))

        self.edges = np.linspace(self.cut_left, self.cut_right,
                                 self.n_slices + 1)
        self.bin_centers = (self.edges[:-1] + self.edges[1:])/2
        self.bin_size = (self.cut_right - self.cut_left) / self.n_slices

    def track_cuts(self, Beam):
        """
        Track the slice frame (limits and slice position) as the mean of the
        bunch moves.
        Requires Beam statistics!
        Method to be refined!
        """

        delta = Beam.mean_dt - 0.5*(self.cut_left + self.cut_right)

        self.cut_left += delta
        self.cut_right += delta
        self.edges += delta
        self.bin_centers += delta

    def convert_coordinates(self, value, input_unit_type):
        """
        Method to convert a value from 'rad' to 's'.
        """

        if input_unit_type is 's':
            return value

        elif input_unit_type is 'rad':
            return value /\
                self.RFParams.omega_rf[0, self.RFParams.counter[0]]

    def get_slices_parameters(self):
        """
        Reuturn all the computed parameters.
        """
        return self.n_slices, self.cut_left, self.cut_right, self.n_sigma, \
            self.edges, self.bin_centers, self.bin_size


class FitOptions(object):
    """
    This class defines the method to be used turn after turn to obtain the
    position and length of the bunch profile.

    Parameters
    ----------

    fit_method : string
        Current options are 'gaussian',
        'fwhm' (full-width-half-maximum converted to 4 sigma gaussian bunch)
        and 'rms'. The methods 'gaussian' and 'rms' give both 4 sigma.
    fitExtraOptions : unknown
        For the moment no options can be passed into fitExtraOptions

    Attributes
    ----------

    fit_method : string
    fitExtraOptions : unknown
    """

    def __init__(self, fit_option=None, fitExtraOptions=None):

        """
        Constructor
        """

        self.fit_option = str(fit_option)
        self.fitExtraOptions = fitExtraOptions


class FilterOptions(object):

    """
    This class defines the filter to be used turn after turn to smooth
    the bunch profile.

    Parameters
    ----------

    filterMethod : string
        The only option available is 'chebishev'
    filterExtraOptions : dictionary
        Parameters for the Chebishev filter (see the method
        beam_profile_filter_chebyshev in filters_and_fitting.py in the toolbox
        package)

    Attributes
    ----------

    filterMethod : string
    filterExtraOptions : dictionary

    """

    def __init__(self, filterMethod=None, filterExtraOptions=None):

        """
        Constructor
        """

        self.filterMethod = str(filterMethod)
        self.filterExtraOptions = filterExtraOptions


class OtherSlicesOptions(object):

    """
    This class groups all the remaining options for the Profile class.

    Parameters
    ----------

    smooth : boolean
        If set True, this method slices the bunch not in the
        standard way (fixed one slice all the macroparticles contribute
        with +1 or 0 depending if they are inside or not). The method assigns
        to each macroparticle a real value between 0 and +1 depending on its
        time coordinate. This method can be considered a filter able to smooth
        the profile.
    direct_slicing : boolean
        If set True, the profile is calculated when the Profile class below
        is created. If False the user has to manually track the Profile object
        in the main file after its creation

    Attributes
    ----------

    smooth : boolean
    direct_slicing : boolean

    """

    def __init__(self, smooth=False, direct_slicing=False):

        """
        Constructor
        """

        self.smooth = smooth
        self.direct_slicing = direct_slicing


class Profile(object):
    """
    Contains the beam profile and related quantities including beam spectrum,
    profile derivative.

    Parameters
    ----------

    Beam : object
        Beam from which the profile has to be calculated
    CutOptions : object
        Options for profile cutting (see above)
    FitOptions : object
        Options to get profile position and length (see above)
    FilterOptions : object
        Options to set a filter (see above)
    OtherSlicesOptions : object
        All remaining options, like smooth histogram and direct
        slicing (see above)

    Attributes
    ----------

    Beam : object
    n_slices : int
        number of slices to be used
    cut_left : float
        left extreme of the profile
    cut_right : float
        right extreme of the profile
    n_sigma : float
        defines the left and right extremes of the profile in case those are
        not given explicitly
    edges : float array
        contains the edges of the slices
    bin_centers : float array
        contains the centres of the slices
    bin_size : float
        lenght of one bin (or slice)
    n_macroparticles : float array
        contains the histogram (or profile); its elements are real if the
        smooth histogram tracking is used
    beam_spectrum : float array
        contains the spectrum of the beam (arb. units)
    beam_spectrum_freq : float array
        contains the frequencies on which the spectrum is computed [Hz]
    operations : list
        contains all the methods to be called every turn, like slice track,
        fitting, filtering etc.
    bunchPosition : float
        profile position [s]
    bunchLength : float
        profile length [s]
    filterExtraOptions : unknown (see above)

    Examples
    --------

    >>> n_slices = 100
    >>> CutOptions = profileModule.CutOptions(cut_left=0,
    >>>       cut_right=self.ring.t_rev[0], n_slices = n_slices, cuts_unit='s')
    >>> FitOptions = profileModule.FitOptions(fit_option='gaussian',
    >>>                                        fitExtraOptions=None)
    >>> filter_option = {'pass_frequency':1e7,
    >>>    'stop_frequency':1e8, 'gain_pass':1, 'gain_stop':2,
    >>>    'transfer_function_plot':False}
    >>> FilterOptions = profileModule.FilterOptions(filterMethod='chebishev',
    >>>         filterExtraOptions=filter_option)
    >>> OtherSlicesOptions = profileModule.OtherSlicesOptions(smooth=False,
    >>>                             direct_slicing = True)
    >>> self.profile4 = profileModule.Profile(my_beam, CutOptions = CutOptions,
    >>>                     FitOptions= FitOptions,
    >>>                     FilterOptions=FilterOptions,
    >>>                     OtherSlicesOptions = OtherSlicesOptions)

    """

    def __init__(self, Beam,
                 CutOptions=CutOptions(),
                 FitOptions=FitOptions(),
                 FilterOptions=FilterOptions(),
                 OtherSlicesOptions=OtherSlicesOptions()):
        """
        Constructor
        """

        # Copy of CutOptions object to be usef for reslicing
        self.cut_options = CutOptions

        # Define bins
        CutOptions.set_cuts(Beam)

        # Import (reference) Beam
        self.Beam = Beam

        # Get all computed parameters from CutOptions
        self.set_slices_parameters()

        # Initialize profile array as zero array
        self.n_macroparticles = np.zeros(self.n_slices, dtype=float)

        # Initialize beam_spectrum and beam_spectrum_freq as empty arrays
        self.beam_spectrum = np.array([], dtype=float)
        self.beam_spectrum_freq = np.array([], dtype=float)

        if OtherSlicesOptions.smooth:
            self.operations = [self._slice_smooth]
        else:
            self.operations = [self._slice]

        if FitOptions.fit_option is not None:
            self.fit_option = FitOptions.fit_option
            self.bunchPosition = 0.0
            self.bunchLength = 0.0
            if FitOptions.fit_option == 'gaussian':
                self.operations.append(self.apply_fit)
            elif FitOptions.fit_option == 'rms':
                self.operations.append(self.rms)
            elif FitOptions.fit_option == 'fwhm':
                self.operations.append(self.fwhm)

        if FilterOptions.filterMethod == 'chebishev':
            self.filterExtraOptions = FilterOptions.filterExtraOptions
            self.operations.append(self.apply_filter)

        if OtherSlicesOptions.direct_slicing:
            self.track()

    def set_slices_parameters(self):
        self.n_slices, self.cut_left, self.cut_right, self.n_sigma, \
                self.edges, self.bin_centers, self.bin_size = \
                self.cut_options.get_slices_parameters()

    def track(self):
        """
        Track method in order to update the slicing along with the tracker.
        """

        for op in self.operations:
            op()

    def _slice(self):
        """
        Constant space slicing with a constant frame. 
        """
        bm.slice(self)
        # libblond.histogram(self.Beam.dt.ctypes.data_as(ctypes.c_void_p), 
        #                  self.n_macroparticles.ctypes.data_as(ctypes.c_void_p), 
        #                  ctypes.c_double(self.cut_left), 
        #                  ctypes.c_double(self.cut_right), 
        #                  ctypes.c_int(self.n_slices), 
        #                  ctypes.c_int(self.Beam.n_macroparticles))

    def _slice_smooth(self):
        """
        At the moment 4x slower than _slice but smoother (filtered).
        """
        bm.slice_smooth(self)
        # libblond.smooth_histogram(self.Beam.dt.ctypes.data_as(ctypes.c_void_p), 
        #                  self.n_macroparticles.ctypes.data_as(ctypes.c_void_p), 
        #                  ctypes.c_double(self.cut_left), 
        #                  ctypes.c_double(self.cut_right), 
        #                  ctypes.c_uint(self.n_slices), 
        #                  ctypes.c_uint(self.Beam.n_macroparticles))

    
    def apply_fit(self):
        """
        It applies Gaussian fit to the profile.
        """

        if self.bunchLength == 0:
            p0 = [max(self.n_macroparticles), np.mean(self.Beam.dt),
                  np.std(self.Beam.dt)]
        else:
            p0 = [max(self.n_macroparticles), self.bunchPosition,
                  self.bunchLength/4]

        self.fitExtraOptions = ffroutines.gaussian_fit(self.n_macroparticles,
                                                       self.bin_centers, p0)
        self.bunchPosition = self.fitExtraOptions[1]
        self.bunchLength = 4*self.fitExtraOptions[2]

    def apply_filter(self):
        """
        It applies Chebishev filter to the profile.
        """
        self.n_macroparticles = ffroutines.beam_profile_filter_chebyshev(
            self.n_macroparticles, self.bin_centers, self.filterExtraOptions)

    def rms(self):
        """
        Computation of the RMS bunch length and position from the line
        density (bunch length = 4sigma).
        """

        self.bunchPosition, self.bunchLength = ffroutines.rms(
            self.n_macroparticles, self.bin_centers)

    def rms_multibunch(self, n_bunches, bunch_spacing_buckets, bucket_size_tau,
                       bucket_tolerance=0.40):
        """
        Computation of the bunch length (4sigma) and position from RMS.
        """

        self.bunchPosition, self.bunchLength = ffroutines.rms_multibunch(
            self.n_macroparticles, self.bin_centers, n_bunches,
            bunch_spacing_buckets, bucket_size_tau, bucket_tolerance)

    def fwhm(self, shift=0):
        """
        Computation of the bunch length and position from the FWHM
        assuming Gaussian line density.
        """

        self.bunchPosition, self.bunchLength = ffroutines.fwhm(
            self.n_macroparticles, self.bin_centers, shift)

    def fwhm_multibunch(self, n_bunches, bunch_spacing_buckets,
                        bucket_size_tau, bucket_tolerance=0.40, shift=0):
        """
        Computation of the bunch length and position from the FWHM
        assuming Gaussian line density for multibunch case.
        """

        self.bunchPosition, self.bunchLength = ffroutines.fwhm_multibunch(
            self.n_macroparticles, self.bin_centers, n_bunches,
            bunch_spacing_buckets, bucket_size_tau, bucket_tolerance, shift)

    def beam_spectrum_freq_generation(self, n_sampling_fft):
        """
        Frequency array of the beam spectrum
        """

        self.beam_spectrum_freq = bm.rfftfreq(n_sampling_fft, self.bin_size)

    def beam_spectrum_generation(self, n_sampling_fft):
        """
        Beam spectrum calculation
        """

        self.beam_spectrum = bm.rfft(self.n_macroparticles, n_sampling_fft)

    def beam_profile_derivative(self, mode='gradient'):
        """
        The input is one of the three available methods for differentiating
        a function. The two outputs are the bin centres and the discrete
        derivative of the Beam profile respectively.*
        """

        x = self.bin_centers
        dist_centers = x[1] - x[0]

        if mode is 'filter1d':
            derivative = ndimage.gaussian_filter1d(
                self.n_macroparticles, sigma=1, order=1, mode='wrap') / \
                dist_centers
        elif mode is 'gradient':
            derivative = np.gradient(self.n_macroparticles, dist_centers)
        elif mode is 'diff':
            derivative = np.diff(self.n_macroparticles) / dist_centers
            diffCenters = x[0:-1] + dist_centers/2
            derivative = np.interp(x, diffCenters, derivative)
        else:
            #ProfileDerivativeError
            raise RuntimeError('Option for derivative is not recognized.')

        return x, derivative
