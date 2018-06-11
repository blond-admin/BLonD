
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute intensity effects**

:Authors: **Juan F. Esteban Mueller**, **Danilo Quartullo**,
          **Alexandre Lasheen**, **Markus Schwarz**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from numpy.fft import  rfft, irfft, rfftfreq
from ctypes import c_uint, c_double, c_void_p
from scipy.constants import e
# from ..setup_cpp import libblond
from .. import libblond

from ..toolbox.next_regular import next_regular

linear_interp_kick = libblond.linear_interp_kick


class TotalInducedVoltage(object):
    r"""
    Object gathering all the induced voltage contributions. The input is a
    list of objects able to compute induced voltages (InducedVoltageTime,
    InducedVoltageFreq, InductiveImpedance). All the induced voltages will
    be summed in order to reduce the computing time. All the induced
    voltages should have the same slicing resolution.
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profiles : object
        Profiles object
    induced_voltage_list : object list
        List of objects for which induced voltages have to be calculated
        
    Attributes
    ----------
    beam : object
        Copy of the Beam object in order to access the beam info
    profile : object
        Copy of the Profiles object in order to access the profile info
    induced_voltage_list : object list
        List of objects for which induced voltages have to be calculated
    induced_voltage : float array
        Array to store the computed induced voltage [V]
    time_array : float array
        Time array corresponding to induced_voltage [s]
    """
    
    def __init__(self, Beam, Profiles, induced_voltage_list):
        """
        Constructor.
        """
        # Copy of the Beam object in order to access the beam info.
        self.beam = Beam
        
        # Copy of the Profiles object in order to access the profile info.
        self.profile = Profiles
        
        # Induced voltage list.
        self.induced_voltage_list = induced_voltage_list
        
        # Induced voltage from the sum of the wake sources in V
        self.induced_voltage = np.zeros(int(self.profile.n_slices))
        
        # Time array of the wake in s
        self.time_array = self.profile.bin_centers


    def reprocess(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """
        
        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.process()
        

    def induced_voltage_sum(self):
        """
        Method to sum all the induced voltages in one single array.
        """
        
        temp_induced_voltage = 0

        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.induced_voltage_generation()
            temp_induced_voltage += \
                  induced_voltage_object.induced_voltage[:self.profile.n_slices]
            
        self.induced_voltage = temp_induced_voltage
        
        
    def track(self):
        """
        Track method to apply the induced voltage kick on the beam.
        """
        
        self.induced_voltage_sum()
        
        linear_interp_kick(self.beam.dt.ctypes.data_as(c_void_p),
                           self.beam.dE.ctypes.data_as(c_void_p),
                           self.induced_voltage.ctypes.data_as(c_void_p), 
                           self.profile.bin_centers.ctypes.data_as(c_void_p),
                           c_double(self.beam.Particle.charge),
                           c_uint(self.profile.n_slices),
                           c_uint(self.beam.n_macroparticles),
                           c_double(0.))


    def track_ghosts_particles(self, ghostBeam):
        
        linear_interp_kick(ghostBeam.dt.ctypes.data_as(c_void_p),
                           ghostBeam.dE.ctypes.data_as(c_void_p), 
                           self.induced_voltage.ctypes.data_as(c_void_p), 
                           self.profile.bin_centers.ctypes.data_as(c_void_p),
                           c_double(self.beam.Particle.charge),
                           c_uint(self.profile.n_slices),
                           c_uint(ghostBeam.n_macroparticles),
                           c_double(0.))


class _InducedVoltage(object):
    r"""
    Induced voltage parent class. Only for internal use (inheritance), not to
    be directly instanciated.
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profiles : object
        Profiles object
    frequency_resolution : float, optional
        Frequency resolution of the impedance [Hz]
    wake_length : float, optional
        Wake length [s]
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    mtw_mode : boolean, optional
        Multi-turn wake mode can be 'freq' or 'time' (default)
    RFParams : object, optional
        RFStation object for turn counter and revolution period
        
    Attributes
    ----------
    beam : object
        Copy of the Beam object in order to access the beam info
    profile : object
        Copy of the Profiles object in order to access the profile info
    induced_voltage : float array
        Induced voltage from the sum of the wake sources in V
    wake_length_input : float
        Wake length [s]
    frequency_resolution_input : float
        Frequency resolution of the impedance [Hz]
    RFParams : object
        RFStation object for turn counter and revolution period
    multi_turn_wake : boolean
        Multi-turn wake enable flag
    mtw_mode : boolean
        Multi-turn wake mode can be 'freq' or 'time' (default)
    """
    
    def __init__(self, Beam, Profile, frequency_resolution=None,
                 wake_length=None, multi_turn_wake=False, mtw_mode='time', 
                 RFParams=None):

        # Beam object in order to access the beam info
        self.beam = Beam
        
        # Profile object in order to access the profile info
        self.profile = Profile
        
        # Induced voltage from the sum of the wake sources in V
        self.induced_voltage = 0
        
        # Wake length in s (optional)
        self.wake_length_input = wake_length
        
        # Frequency resolution of the impedance (optional)
        self.frequency_resolution_input = frequency_resolution
            
        # RFStation object for turn counter and revolution period
        self.RFParams = RFParams
        
        # Multi-turn wake enable flag
        self.multi_turn_wake = multi_turn_wake
        
        # Multi-turn wake mode can be 'freq' or 'time' (default). If 'freq'
        # is used, each turn the induced voltage of previous turns is shifted 
        # in the frequency domain. For 'time', a linear interpolation is used.
        self.mtw_mode = mtw_mode
        
        self.process()


    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """

        if (self.wake_length_input != None
                and self.frequency_resolution_input == None):
            # Number of points of the induced voltage array
            self.n_induced_voltage = int(np.ceil(self.wake_length_input/
                                                 self.profile.bin_size))
            if self.n_induced_voltage < self.profile.n_slices:
                raise RuntimeError('Error: too short wake length. '+
                'Increase it above {0:1.2e} s.'.format(self.profile.n_slices *
                                                       self.profile.bin_size))
            # Wake length in s, rounded up to the next multiple of bin size
            self.wake_length = self.n_induced_voltage * self.profile.bin_size
            self.frequency_resolution = 1 / self.wake_length
        elif (self.frequency_resolution_input != None
                and self.wake_length_input == None):
            self.n_induced_voltage = int(np.ceil(1/ (self.profile.bin_size *
                                             self.frequency_resolution_input)))
            if self.n_induced_voltage < self.profile.n_slices:
                raise RuntimeError('Error: too large frequency_resolution. '+
                'Reduce it below {0:1.2e} Hz.'.format(1 / 
                               (self.profile.cut_right - self.profile.cut_left)))
            self.wake_length = self.n_induced_voltage * self.profile.bin_size
            # Frequency resolution in Hz
            self.frequency_resolution = 1 / self.wake_length
        elif (self.wake_length_input == None
                and self.frequency_resolution_input == None):
            # By default the wake_length is the slicing frame length
            self.wake_length = (self.profile.cut_right -
                               self.profile.cut_left)
            self.frequency_resolution = 1 / self.wake_length
            self.n_induced_voltage = self.profile.n_slices
        else:
            raise RuntimeError('Error: only one of wake_length or '+
                'frequency_resolution can be specified.')
                
        if self.multi_turn_wake:            
            # Number of points of the memory array for multi-turn wake
            self.n_mtw_memory = self.n_induced_voltage
            
            self.front_wake_buffer = 0
            
            if self.mtw_mode == 'freq':
                # In frequency domain, an extra buffer for a revolution turn is
                # needed due to the circular time shift in frequency domain
                self.buffer_size = \
                    np.ceil(np.max(self.RFParams.t_rev) / self.profile.bin_size)
                # Extending the buffer to reduce the effect of the front wake
                self.buffer_size += \
                    np.ceil(np.max(self.buffer_extra) / self.profile.bin_size)
                self.n_mtw_memory += int(self.buffer_size)
                # Using next regular for FFTs speedup
                self.n_mtw_fft = next_regular(self.n_mtw_memory)
                # Frequency and omega arrays
                self.freq_mtw = \
                    rfftfreq(self.n_mtw_fft, d=self.profile.bin_size)
                self.omegaj_mtw = 2.0j * np.pi * self.freq_mtw
                # Selecting time-shift method
                self.shift_trev = self.shift_trev_freq
            else:
                # Selecting time-shift method
                self.shift_trev = self.shift_trev_time
                # Time array
                self.time_mtw = np.linspace(0, self.wake_length,
                                            self.n_mtw_memory, endpoint=False)
            
            # Array to add and shift in time the multi-turn wake over the turns
            self.mtw_memory = np.zeros(self.n_mtw_memory)
            
            # Select induced voltage generation method to be used
            self.induced_voltage_generation = self.induced_voltage_mtw
        else:
            self.induced_voltage_generation = self.induced_voltage_1turn


    def induced_voltage_1turn(self):
        """
        Method to calculate the induced voltage at the current turn. DFTs are 
        used for calculations in time and frequency domain (see classes below)
        """
        
        self.profile.beam_spectrum_generation(self.n_fft)
        
        induced_voltage = - (self.beam.Particle.charge * e * self.beam.ratio *
            irfft(self.total_impedance * self.profile.beam_spectrum))
        
        self.induced_voltage = induced_voltage[:self.n_induced_voltage]


    def induced_voltage_mtw(self):
        """
        Method to calculate the induced voltage taking into account the effect
        from previous passages (multi-turn wake)
        """

        # Shift of the memory wake field by the current revolution period
        self.shift_trev()
        
        # Induced voltage of the current turn calculation
        self.induced_voltage_1turn()
        
        # Setting to zero to the last part to remove the contribution from the
        # front wake
        self.induced_voltage[self.n_induced_voltage -
                             self.front_wake_buffer:] = 0
        
        # Add the induced voltage of the current turn to the memory from
        # previous turns
        self.mtw_memory[:self.n_induced_voltage] += self.induced_voltage
        
        self.induced_voltage = self.mtw_memory[:self.n_induced_voltage]


    def shift_trev_freq(self):
        """
        Method to shift the induced voltage by a revolution period in the
        frequency domain
        """
        
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        # Shift in frequency domain    
        induced_voltage_f = rfft(self.mtw_memory, self.n_mtw_fft)
        induced_voltage_f *= np.exp(self.omegaj_mtw * t_rev)
        self.mtw_memory = irfft(induced_voltage_f)[:self.n_mtw_memory]
        # Setting to zero to the last part to remove the contribution from the
        # circular convolution
        self.mtw_memory[-int(self.buffer_size):] = 0
    
    
    def shift_trev_time(self):
        """
        Method to shift the induced voltage by a revolution period in the
        time domain (linear interpolation)
        """
        
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        self.mtw_memory = np.interp(self.time_mtw + t_rev, self.time_mtw,
                                    self.mtw_memory, left=0, right=0)


    def _track(self):
        """
        Tracking method
        """
        
        self.induced_voltage_generation()

        linear_interp_kick(self.beam.dt.ctypes.data_as(c_void_p),
                           self.beam.dE.ctypes.data_as(c_void_p),
                           self.induced_voltage.ctypes.data_as(c_void_p),
                           self.profile.bin_centers.ctypes.data_as(c_void_p),
                           c_double(self.beam.Particle.charge),
                           c_uint(self.profile.n_slices),
                           c_uint(self.beam.n_macroparticles),
                           c_double(0.))



class InducedVoltageTime(_InducedVoltage):
    r"""
    Induced voltage derived from the sum of several wake fields (time domain)
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profile : object
        Profile object
    wake_source_list : list
        Wake sources list (e.g. list of Resonator objects)
    wake_length : float, optional
        Wake length [s]
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    RFParams : object, optional
        RFStation object for turn counter and revolution period
    mtw_mode : boolean, optional
        Multi-turn wake mode can be 'freq' or 'time' (default)
    
    Attributes
    ----------
    wake_source_list : list
        Wake sources list (e.g. list of Resonator objects)
    total_wake : float array
        Total wake array of all sources in :math:`\Omega / s`
    """
    
    def __init__(self, Beam, Profile, wake_source_list, wake_length=None,
                 multi_turn_wake=False, RFParams=None, mtw_mode=None):
                 
        # Wake sources list (e.g. list of Resonator objects)
        self.wake_source_list = wake_source_list
                
        # Total wake array of all sources in :math:`\Omega / s`
        self.total_wake = 0
        
        # Call the __init__ method of the parent class [calls process()]
        _InducedVoltage.__init__(self, Beam, Profile, frequency_resolution=None,
                 wake_length=wake_length, multi_turn_wake=multi_turn_wake,
                 RFParams=RFParams, mtw_mode=mtw_mode)


    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """

        _InducedVoltage.process(self)
        
        # Number of points for the FFT, equal to the length of the induced
        # voltage array + number of profile -1 to calculate a linear convolution
        # in the frequency domain. The next regular number is used for speed,
        # therefore the frequency resolution is always equal or finer than
        # the input value
        self.n_fft = next_regular(int(self.n_induced_voltage) +
                                  int(self.profile.n_slices) - 1)
        
        # Time array of the wake in s
        self.time = np.arange(0, self.wake_length, self.wake_length /
                              self.n_induced_voltage)
        
        # Processing the wakes
        self.sum_wakes(self.time)    


    def sum_wakes(self, time_array):
        """
        Summing all the wake contributions in one total wake.
        """
        
        self.total_wake = np.zeros(time_array.shape)
        for wake_object in self.wake_source_list:
            wake_object.wake_calc(time_array)
            self.total_wake += wake_object.wake
            
        # Pseudo-impedance used to calculate linear convolution in the 
        # frequency domain (padding zeros)
        self.total_impedance = rfft(self.total_wake, self.n_fft)


    
class InducedVoltageFreq(_InducedVoltage):
    r"""
    Induced voltage derived from the sum of several impedances
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profile : object
        Profile object
    impedance_source_list : list
        Impedance sources list (e.g. list of Resonator objects)
    frequency_resolution : float, optional
        Frequency resolution of the impedance [Hz]
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    front_wake_length : float, optional
        Lenght [s] of the front wake (if any) for multi-turn wake mode
    RFParams : object, optional
        RFStation object for turn counter and revolution period
    mtw_mode : boolean, optional
        Multi-turn wake mode can be 'freq' or 'time' (default)
    
    Attributes
    ----------
    impedance_source_list : list
        Impedance sources list (e.g. list of Resonator objects)
    total_impedance : float array
        Total impedance array of all sources in* :math:`\Omega`
    front_wake_length : float
        Lenght [s] of the front wake (if any) for multi-turn wake mode
    """
    
        
    def __init__(self, Beam, Profile, impedance_source_list, 
                 frequency_resolution=None, multi_turn_wake=False, 
                 front_wake_length=0, RFParams=None, mtw_mode=None):
        
        # Impedance sources list (e.g. list of Resonator objects)
        self.impedance_source_list = impedance_source_list
        
        # Total impedance array of all sources in* :math:`\Omega`
        self.total_impedance = 0
        
        # Lenght in s of the front wake (if any) for multi-turn wake mode.
        # If the impedance calculation is performed in frequency domain, an
        # artificial front wake may appear. With this option, it is possible to
        # set to zero a portion at the end of the induced voltage array.*
        self.front_wake_length = front_wake_length
        
        # Call the __init__ method of the parent class
        _InducedVoltage.__init__(self, Beam, Profile, wake_length=None,
                 frequency_resolution=frequency_resolution,
                 multi_turn_wake=multi_turn_wake, RFParams=RFParams,
                 mtw_mode=mtw_mode)


    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile change
        """

        _InducedVoltage.process(self)
        
        # Number of points for the FFT. The next regular number is used for 
        # speed, therefore the frequency resolution is always equal or finer
        # than the input value
        self.n_fft = next_regular(self.n_induced_voltage)
                
        self.profile.beam_spectrum_freq_generation(self.n_fft)
        
        # Frequency array of the impedance in Hz
        self.freq = self.profile.beam_spectrum_freq
            
        # Length of the front wake in frequency domain calculations 
        if self.front_wake_length:            
            self.front_wake_buffer = int(np.ceil(
                    np.max(self.front_wake_length) / self.profile.bin_size))
        
        # Processing the impedances
        self.sum_impedances(self.freq)
        
    
    def sum_impedances(self, freq):
        """
        Summing all the wake contributions in one total impedance.
        """
        
        self.total_impedance = np.zeros(freq.shape, complex)
       
        for i in range(len(self.impedance_source_list)):
            self.impedance_source_list[i].imped_calc(freq)
            self.total_impedance += self.impedance_source_list[i].impedance

        # Factor relating Fourier transform and DFT            
        self.total_impedance /= self.profile.bin_size



class InductiveImpedance(_InducedVoltage):
    """
    Constant imaginary Z/n impedance
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profile : object
        Profile object
    Z_over_n : float array
        Constant imaginary Z/n program in* :math:`\Omega`.
    RFParams : object
        RFStation object for turn counter and revolution period
    deriv_mode : string, optional
        Derivation method to compute induced voltage
    
    Attributes
    ----------
    Z_over_n : float array
        Constant imaginary Z/n program in* :math:`\Omega`.
    deriv_mode : string, optional
        Derivation method to compute induced voltage
    """
    
    def __init__(self, Beam, Profile, Z_over_n, RFParams, 
                 deriv_mode='gradient'):

        # Constant imaginary Z/n program in* :math:`\Omega`.
        self.Z_over_n = Z_over_n
        
        # Derivation method to compute induced voltage
        self.deriv_mode = deriv_mode
        
        # Call the __init__ method of the parent class
        _InducedVoltage.__init__(self, Beam, Profile, RFParams=RFParams)


    def induced_voltage_1turn(self):
        """
        Method to calculate the induced voltage through the derivative of the
        profile. The impedance must be a constant Z/n.
        """
        
        index = self.RFParams.counter[0]
        
        induced_voltage = - (self.beam.Particle.charge * e / (2 * np.pi) *
                self.beam.ratio * self.Z_over_n[index] *
                self.RFParams.t_rev[index] / self.profile.bin_size *
                self.profile.beam_profile_derivative(self.deriv_mode)[1])

        self.induced_voltage = induced_voltage[:self.n_induced_voltage]


class InducedVoltageResonator(_InducedVoltage):
    r"""
    *Calculates the induced voltage of several resonators for arbitrary 
    line density. It does so by linearily interpolating the line density and 
    solving the convolution integral with the resonator impedance analytically.
    The line density need NOT be sampled at equidistant points. The times where
    the induced voltage is calculated need to be the same where the line 
    density is sampled. If no timeArray is passed, the induced voltage is 
    evaluated at the points of the line density. This is nececassry of 
    compatability with other functions that calculate the induced voltage.
    Currently, it requires the all quality factors :math:`Q>0.5`
    Currently, only works for single turn.*
    
    Parameters
    ----------
    Beam : object
        Beam object
    Profile : object
        Profile object
    Resonators : object
        Resonators object
    timeArray : float array, optional
        Array of time values where the induced voltage is calculated. 
        If left out, the induced voltage is calculated at the times of the line
        density.
    
    Attributes
    ----------
    beam : object
        Copy of the Beam object in order to access the beam info.
    profile : object
        Copy of the Profile object in order to access the line density.
    tArray : float array
        array of time values where the induced voltage is calculated. 
        If left out, the induced voltage is calculated at the times of the 
        line density
    atLineDensityTimes : boolean
        flag indicating if the induced voltage has to be computed for timeArray
        or for the line density
    n_time : int
        length of tArray
    R, omega_r, Q : lists of float
        Resonators parameters
    n_resonators : int
        Number of resonators
    induced_voltage : float array
        Computed induced voltage [V]
    """
    
    def __init__ (self, Beam, Profile, Resonators, timeArray=None):
        
        # Test if one or more quality factors is smaller than 0.5.
        if sum(Resonators.Q<0.5)>0:
            raise RuntimeError('All quality factors Q must be larger than 0.5')
        
        # Copy of the Beam object in order to access the beam info.
        self.beam = Beam
        # Copy of the Profile object in order to access the line density.
        self.profile = Profile
        
        # Optional array of time values where the induced voltage is calculated.
        # If left out, the induced voltage is calculated at the times of the
        # line density.
        if timeArray is None:
            self.tArray =  self.profile.bin_centers
            self.atLineDensityTimes = True
        else:
            self.tArray = timeArray
            self.atLineDensityTimes = False
        
        # Length of timeArray
        self.n_time = len(self.tArray)
        
        # Copy of the shunt impedances of the Resonators in* :math:`\Omega`
        self.R = Resonators.R_S
        # Copy of the resonant frequencies of the Resonators in in 1/s
        self.omega_r = Resonators.omega_R #resonant frequencies [1/s]
        # Copy of the quality factors of the Resonators
        self.Q = Resonators.Q
        # Number of resonators
        self.n_resonators = len(self.R)

        # For internal use
        self._Qtilde = self.Q * np.sqrt(1. - 1./(4.*self.Q**2.))
        self._reOmegaP = self.omega_r * self._Qtilde / self.Q
        self._imOmegaP = self.omega_r / (2.*self.Q)
        
        # Each the 'n_resonator' rows of the matrix holds the induced voltage
        # at the 'n_time' time-values of one cavity. For internal use.
        self._tmp_matrix = np.ones((self.n_resonators, self.n_time))
        
        # Slopes of the line segments. For internal use.
        self._kappa1 = np.zeros(int(self.profile.n_slices-1))

        # Matrix to hold n_times many tArray[t]-bin_centers arrays.        
        self._deltaT = np.zeros((self.n_time,self.profile.n_slices))
        
        # Call the __init__ method of the parent class [calls process()]
        _InducedVoltage.__init__(self, Beam, Profile, wake_length=None,
             frequency_resolution=None,
             multi_turn_wake=False, RFParams=None, mtw_mode=None)
    
    
    def process(self):
        r"""
        Reprocess the impedance contributions. To be run when slicing changes
        """

        _InducedVoltage.process(self)
        
        # Since profile object changed, need to assign the proper dimensions to 
        # _kappa1 and _deltaT
        self._kappa1 = np.zeros(int(self.profile.n_slices-1))
        self._deltaT = np.zeros((self.n_time,self.profile.n_slices))
    
    
    def induced_voltage_1turn(self):
        r"""
        Method to calculate the induced voltage through linearily 
        interpolating the line density and applying the analytic equation
        to the result.
        """
        
        # Compute the slopes of the line sections of the linearily interpolated
        # (normalized) line density.
        self._kappa1[:] = np.diff(self.profile.n_macroparticles) \
                        / np.diff(self.profile.bin_centers) \
                        / (self.beam.n_macroparticles*self.profile.bin_size)
                        #[:] makes kappa pass by reference
        
        for t in range(self.n_time):
            self._deltaT[t] = self.tArray[t]-self.profile.bin_centers
        
        # For each cavity compute the induced voltage and store in the r-th row
        for r in range(self.n_resonators):
            tmp_sum=((((2 
                     *np.cos(self._reOmegaP[r] * self._deltaT)
                     +np.sin(self._reOmegaP[r] * self._deltaT)/self._Qtilde[r]) 
                     * np.exp(-self._imOmegaP[r] * self._deltaT)) 
                     * self.Heaviside(self._deltaT)) 
                     - np.sign(self._deltaT))
            # np.sum performs the sum over the points of the line density
            self._tmp_matrix[r] = self.R[r]/(2*self.omega_r[r]*self.Q[r]) \
                                * np.sum(self._kappa1 * np.diff(tmp_sum),axis=1)
        
        # To obtain the voltage, sum the contribution of each cavity...
        self.induced_voltage = self._tmp_matrix.sum(axis=0)
        # ... and multiply with bunch charge
        self.induced_voltage *= -self.beam.Particle.charge*e*self.beam.intensity
    
    # Implementation of Heaviside function
    def Heaviside(self,x):
        r"""
        Heaviside function, which returns 1 if x>1, 0 if x<0, and 1/2 if x=0
        """
        
        return 0.5*(np.sign(x) + 1.)

