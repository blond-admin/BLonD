# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to describe classes for the calculation of wakes and impedances to 
be used either alone or by InducedVoltage objects described in impedance.py.
The module consists of a parent class called _ImpedanceObject and several child
classes, as for example InputTable, Resonators and TravelingWaveCavity.**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, 
**Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from scipy.constants import c, physical_constants
import ctypes
# from ..setup_cpp import libblond
from .. import libblond


class _ImpedanceObject(object):
    
    """ 
    Parent impedance object to implement required methods and attributes 
    common to all the child classes. The attributes are initialised to 0 but
    they are overwritten by float arrays when the child classes are used.
    """
    
    def __init__(self):        
        # Time array of the wake in s
        self.time_array = 0
        
        # Wake array in :math:`\Omega / s`
        self.wake = 0
        
        # Frequency array of the impedance in Hz
        self.frequency_array = 0
        
        # Impedance array in :math:`\Omega`
        self.impedance = 0
        
    
    def wake_calc(self, *args):
        """
        Method required to compute the wake function. Returns an error if
        called from an object which does not implement this method.
        """
        
        raise RuntimeError('wake_calc() method not implemented in this class'+
                           '. This object is probably meant to be used in the'+
                           ' frequency domain')
    
    
    def imped_calc(self, *args):    
        """
        Method required to compute the impedance. Returns an error if called
        from an object which does not implement this method.
        """
        
        raise RuntimeError('imped_calc() method not implemented in this class'+
                           '. This object is probably meant to be used in the'+
                           ' time domain')
    

class InputTable(_ImpedanceObject):
    r"""
    Intensity effects from impedance and wake tables.
    If the constructor takes just two arguments, then a wake table is passed;
    if it takes three arguments, then an impedance table is passed. Be careful
    that if you input a wake, the input wake for W(t=0) should be already 
    divided by two (beam loading theorem) ; and that if you input impedance, 
    only the positive  frequencies of the impedance is needed (the impedance
    will be assumed to be Hermitian (Real part symmetric and Imaginary part
    antisymmetric).Note that we add the point (f, Z(f)) = (0, 0) to the 
    frequency and impedance arrays derived from the table.
    
    Parameters
    ----------
    input_1 : float array
        Time array of the wake in s or frequency array of the impedance in Hz
    input_2 : float array
        Wake array in :math:`\Omega / s` or real part of impedance 
        in :math:`\Omega`
    input_3 : float array
        Imaginary part of impedance in :math:`\Omega`
    
        
    Attributes
    ----------
    time_array : float array
        Input time array of the wake in s
    wake_array : float array
        Input wake array in :math:`\Omega / s`
    frequency_array_loaded : float array
        Input frequency array of the impedance in Hz
    Re_Z_array_loaded : float array
        Input real part of impedance in :math:`\Omega`
    Im_Z_array_loaded : float array
        Input imaginary part of impedance in :math:`\Omega`
    impedance_loaded : complex array
        Input impedance array in :math:`\Omega + j \Omega`
    
    Examples
    --------
    >>> ##### WAKEFIELD TABLE
    >>> time = np.array([1.1,2.2,3.3])
    >>> wake = np.array([4.4,5.5,6.6])
    >>> table = InputTable(time, wake)
    >>> new_time_array = np.array([7.7,8.8,9.9])
    >>> table.wake_calc(new_time_array)
    >>> 
    >>> ##### IMPEDANCE TABLE
    >>> frequency = np.array([1.1,2.2,3.3])
    >>> real_part = np.array([4.4,5.5,6.6])
    >>> imaginary_part = np.array([7.7,8.8,9.9])
    >>> table2 = InputTable(frequency, real_part, imaginary_part)
    >>> new_freq_array = np.array([7.7,8.8,9.9])
    >>> table2.imped_calc(new_freq_array)
    
    """
    
    def __init__(self, input_1, input_2, input_3 = None):    
        
        _ImpedanceObject.__init__(self)   

        if input_3 is None:
            # Time array of the wake in s
            self.time_array = input_1
            # Wake array in :math:`\Omega / s`
            self.wake_array = input_2
        else:
            # Frequency array of the impedance in Hz
            self.frequency_array_loaded = input_1
            # Real part of impedance in :math:`\Omega`
            self.Re_Z_array_loaded = input_2
            # Imaginary part of impedance in :math:`\Omega`
            self.Im_Z_array_loaded = input_3
            # Impedance array in :math:`\Omega`
            self.impedance_loaded = (self.Re_Z_array_loaded + 1j * 
                                     self.Im_Z_array_loaded)
            
            if self.frequency_array_loaded[0] != 0:
                self.frequency_array_loaded = np.hstack((0, 
                                                  self.frequency_array_loaded))
                self.Re_Z_array_loaded = np.hstack((0, self.Re_Z_array_loaded))
                self.Im_Z_array_loaded = np.hstack((0, self.Im_Z_array_loaded))
    
    
    def wake_calc(self, new_time_array):
        r"""
        The wake from the table is interpolated using the new time array.
        
        Parameters
        ----------
        new_time_array : float array
            Input time array in s
            
        Attributes
        ----------
        new_time_array : float array
            Input time array in s
        wake: float array
            Output interpolated wake in :math:`\Omega / s`
        """
        
        self.new_time_array = new_time_array
        self.wake = np.interp(self.new_time_array, self.time_array, 
                              self.wake_array, right=0)
                           
    
    def imped_calc(self, new_frequency_array):
        r"""
        The impedance from the table is interpolated using the new frequency
        array.
        
        Parameters
        ----------
        new_frequency_array : float array
            frequency array in :math:`\Omega`
            
        Attributes
        ----------
        frequency_array : float array
            frequency array in :math:`\Omega`
        Re_Z_array : float array
            Real part of the interpolated impedance in :math:`\Omega`
        Im_Z_array : float array
            Imaginary part of the interpolated impedance in :math:`\Omega`
        impedance : complex array
            Output interpolated impedance array in :math:`\Omega + j \Omega`
        """

        Re_Z = np.interp(new_frequency_array, self.frequency_array_loaded, 
                         self.Re_Z_array_loaded, right = 0)
        Im_Z = np.interp(new_frequency_array, self.frequency_array_loaded, 
                         self.Im_Z_array_loaded, right = 0)
        self.frequency_array = new_frequency_array
        self.Re_Z_array = Re_Z
        self.Im_Z_array = Im_Z
        self.impedance = Re_Z + 1j * Im_Z
        
    
    
class Resonators(_ImpedanceObject):
    r"""
    Impedance contribution from resonators, analytic formulas for both wake 
    and impedance. The resonant modes (and the corresponding R and Q) 
    can be inputed as a list in case of several modes.
    The model is the following:
    
    .. math::
    
        Z(f) = \frac{R}{1 + j Q \left(\frac{f}{f_r}-\frac{f_r}{f}\right)}
        
    .. math::
        
        W(t>0) = 2\alpha R e^{-\alpha t}\left(\cos{\bar{\omega}t} - \frac{\alpha}{\bar{\omega}}\sin{\bar{\omega}t}\right)

        W(0) = \alpha R
        
    .. math::
        
        \omega_r = 2 \pi f_r
        
        \alpha = \frac{\omega_r}{2Q}
        
        \bar{\omega} = \sqrt{\omega_r^2 - \alpha^2}
    
    Parameters
    ----------
    R_S : float list
        Shunt impepdance in :math:`\Omega`
    frequency_R : float list
        Resonant frequency in Hz
    Q : float list
        Quality factor
    method: string
        It defines which algorithm to use to calculate the impedance (C++ or
        Python)
    
    Attributes
    ----------
    R_S : float array
        Shunt impepdance in :math:`\Omega`
    frequency_R : float array
        Resonant frequency in Hz
    Q : float array
        Quality factor
    n_resonators : int
        number of resonators
        
    Examples
    ----------
    >>> R_S = [1, 2, 3]
    >>> frequency_R = [1, 2, 3]
    >>> Q = [1, 2, 3]
    >>> resonators = Resonators(R_S, frequency_R, Q)
    >>> time = np.array(1,2,3)
    >>> resonators.wake_calc(time)
    >>> frequency = np.array(1,2,3)
    >>> resonators.imped_calc(frequency)
    """
    
    def __init__(self, R_S, frequency_R, Q, method='c++'):
        
        _ImpedanceObject.__init__(self)

        # Shunt impepdance in :math:`\Omega`
        self.R_S = np.array([R_S], dtype=float).flatten()
        
        # Resonant frequency in Hz
        self.frequency_R = np.array([frequency_R], dtype=float).flatten()
                
        # Quality factor
        self.Q = np.array([Q], dtype=float).flatten()
        
        # Number of resonant modes
        self.n_resonators = len(self.R_S)    
    
        if method == 'c++':
            self.imped_calc = self._imped_calc_cpp
        elif method == 'python':
            self.imped_calc = self._imped_calc_python
        else:
            raise RuntimeError('method for impedance calculation in Resonator object not recognized')


    @property
    def frequency_R(self):
        return self.__frequency_R
        
    @frequency_R.setter
    def frequency_R(self, frequency_R):
        self.__frequency_R = frequency_R
        self.__omega_R = 2 *np.pi * frequency_R

    # Resonant angular frequency in rad/s
    @property
    def omega_R(self):
        return self.__omega_R
        
    @omega_R.setter
    def omega_R(self, omega_R):
        self.__frequency_R = omega_R / 2 / np.pi
        self.__omega_R = omega_R


    def wake_calc(self, time_array):
        r"""
        Wake calculation method as a function of time.
        
        Parameters
        ----------
        time_array : float array
            Input time array in s
        
        Attributes
        ----------
        time_array : float array
            Input time array in s
        wake : float array
            Output wake in :math:`\Omega / s`    
        """
        
        self.time_array = time_array
        self.wake = np.zeros(self.time_array.shape)
        
        for i in range(0, self.n_resonators):
       
            alpha = self.omega_R[i] / (2 * self.Q[i])
            omega_bar = np.sqrt(self.omega_R[i] ** 2 - alpha ** 2)
            
            self.wake += ((np.sign(self.time_array) + 1) * self.R_S[i] *
                         alpha * np.exp(-alpha * self.time_array) *
                         (np.cos(omega_bar * self.time_array) - alpha /
                          omega_bar * np.sin(omega_bar * self.time_array)))
    
    

    def _imped_calc_python(self, frequency_array):
        r"""
        Impedance calculation method as a function of frequency using Python.
        
        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz
        
        Attributes
        ----------
        frequency_array : float array
            nput frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array), complex)
        
        for i in range(0, self.n_resonators):
            
            self.impedance[1:] += self.R_S[i] / (1 + 1j * self.Q[i] *
                             (self.frequency_array[1:] / self.frequency_R[i] - 
                              self.frequency_R[i] / self.frequency_array[1:]))
 
    def _imped_calc_cpp(self, frequency_array):
        r"""
        Impedance calculation method as a function of frequency optimised in C++
        
        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz
        
        Attributes
        ----------
        frequency_array : float array
            nput frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array), complex)
        realImp = np.zeros(len(self.frequency_array))
        imagImp = np.zeros(len(self.frequency_array))

        libblond.fast_resonator_real_imag(realImp.ctypes.data_as(ctypes.c_void_p), imagImp.ctypes.data_as(ctypes.c_void_p),
               self.frequency_array.ctypes.data_as(ctypes.c_void_p), self.R_S.ctypes.data_as(ctypes.c_void_p),
               self.Q.ctypes.data_as(ctypes.c_void_p), self.frequency_R.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_uint(self.n_resonators), ctypes.c_uint(len(self.frequency_array)))
 
        self.impedance.real = realImp
        self.impedance.imag = imagImp



class TravelingWaveCavity(_ImpedanceObject):
    r"""
    Impedance contribution from travelling wave cavities, analytic formulas for 
    both wake and impedance. The resonance modes (and the corresponding R and a) 
    can be inputed as a list in case of several modes.
    
    The model is the following:
    
    .. math:: 
        Z &= Z_+ + Z_- \\
        Z_-(f) &= R \left[\left(\frac{\sin{\frac{a(f-f_r)}{2}}}{\frac{a(f-f_r)}{2}}\right)^2 - 2i \frac{a(f-f_r) - \sin{a(f-f_r)}}{\left(a(f-f_r)\right)^2}\right] \\
        Z_+(f) &= R \left[\left(\frac{\sin{\frac{a(f+f_r)}{2}}}{\frac{a(f+f_r)}{2}}\right)^2 - 2i \frac{a(f+f_r) - \sin{a(f+f_r)}}{\left(a(f+f_r)\right)^2}\right]
        
    .. math::
        W(0<t<\tilde{a}) &= \frac{4R}{\tilde{a}}\left(1-\frac{t}{\tilde{a}}\right)\cos{\omega_r t} \\
        W(0) &= \frac{2R}{\tilde{a}}
        
    .. math::
        a = 2 \pi \tilde{a}
    
    Parameters
    ----------
    R_S : float list
        Shunt impepdance in :math:`\Omega`
    frequency_R : float list
        Resonant frequency in Hz
    a_factor : float list
        Damping time a in s
    
    Attributes
    ----------
    R_S : float array
        Shunt impepdance in :math:`\Omega`
    frequency_R : float array
        Resonant frequency in Hz
    a_factor : float array
        Damping time a in s
    n_twc : int
        number of resonant modes
    
    Examples
    ----------
    >>> R_S = [1, 2, 3]
    >>> frequency_R = [1, 2, 3]
    >>> a_factor = [1, 2, 3]
    >>> twc = TravelingWaveCavity(R_S, frequency_R, a_factor)
    >>> time = np.array(1,2,3)
    >>> twc.wake_calc(time)
    >>> frequency = np.array(1,2,3)
    >>> twc.imped_calc(frequency)
    """
    
    
    def __init__(self, R_S, frequency_R, a_factor):
        
        _ImpedanceObject.__init__(self)
        
        # Shunt impepdance in :math:`\Omega`
        self.R_S = np.array([R_S], dtype=float).flatten()
        
        # Resonant frequency in Hz
        self.frequency_R = np.array([frequency_R], dtype=float).flatten()
        
        # Damping time a in s
        self.a_factor = np.array([a_factor], dtype=float).flatten()
        
        # Number of resonant modes
        self.n_twc = len(self.R_S)
        
    
    def wake_calc(self, time_array):
        r"""
        Wake calculation method as a function of time.
        
        Parameters
        ----------
        time_array : float array
            Input time array in s
        
        Attributes
        ----------
        time_array : float array
            Input time array in s
        wake : float array
            Output wake in :math:`\Omega / s`    
        """
        
        self.time_array = time_array
        self.wake = np.zeros(self.time_array.shape)
        
        for i in range(0, self.n_twc):
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = np.where(self.time_array <= a_tilde)
            self.wake[indexes] += ((np.sign(self.time_array[indexes]) + 1) * 2
                                  * self.R_S[i] / a_tilde * 
                                  (1 - self.time_array[indexes] / a_tilde) *
                                  np.cos(2 * np.pi * self.frequency_R[i] *
                                  self.time_array[indexes]))
    
    
    def imped_calc(self, frequency_array):
        r"""
        Impedance calculation method as a function of frequency.
        
        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz
        
        Attributes
        ----------
        frequency_array : float array
            nput frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array), complex)
        
        for i in range(0, self.n_twc):
            
            Zplus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 *
                    (self.frequency_array - self.frequency_R[i])) / 
                    (self.a_factor[i] / 2 * (self.frequency_array -
                    self.frequency_R[i])))**2 - 2j*(self.a_factor[i] *
                    (self.frequency_array - self.frequency_R[i]) -
                    np.sin(self.a_factor[i] * (self.frequency_array - 
                    self.frequency_R[i]))) / (self.a_factor[i] *
                    (self.frequency_array - self.frequency_R[i]))**2)
            
            Zminus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * 
                     (self.frequency_array + self.frequency_R[i])) /
                     (self.a_factor[i] / 2 * (self.frequency_array + 
                     self.frequency_R[i])))**2 - 2j*(self.a_factor[i] *
                     (self.frequency_array + self.frequency_R[i]) - 
                     np.sin(self.a_factor[i] * (self.frequency_array +
                     self.frequency_R[i]))) / (self.a_factor[i] *
                     (self.frequency_array + self.frequency_R[i]))**2)
            
            self.impedance += Zplus + Zminus   

 

class ResistiveWall(_ImpedanceObject):
    r"""
    Impedance contribution from resistive wall for a cilindrical beam pipe
    
    The model is the following:
    
    .. math::
    
        Z(f) = \frac{Z_0 c L}{ \pi } \frac{ 1 }{ \left[1 - i \sign{f}\right] 2  b  c \sqrt{ \frac{\sigma_c Z_0 c }{ 4 \pi |f| } + i 2 \pi b^2 f }
     
    Parameters
    ----------
    pipe_radius : float
        Beam pipe radius in m
    pipe_length : float
        Beam pipe length in m
    resistivity : float
        Beam pipe resistivity in :math:`m / s`
    conductivity : float
        Beam pipe conductivity in :math:`s / m`
    
    Attributes
    ----------
    pipe_radius : float
        Beam pipe radius in m
    pipe_length : float
        Beam pipe length in m
    conductivity : float
        Beam pipe conductivity in :math:`s / m`   
    Z0 : float
        Characteristic impedance of vacuum in* :math:`\Omega`
        
    Examples
    ----------
    >>> pipe_radius = 1
    >>> pipe_length = 2
    >>> resistivity = 3
    >>> rw = ResistiveWall(pipe_radius, pipe_length, resistivity)
    >>> frequency = np.array(1,2,3)
    >>> rw.imped_calc(frequency)
    """
    
    def __init__(self, pipe_radius, pipe_length, resistivity = None, 
                 conductivity = None):
        
        _ImpedanceObject.__init__(self)
        
        # Beam pipe radius in m
        self.pipe_radius = float(pipe_radius)
        
        # Beam pipe length in m
        self.pipe_length = float(pipe_length)
        
        # Beam pipe conductivity in :math:`s / m`
        if resistivity != None:
            self.conductivity = 1/resistivity
        elif conductivity != None:
            self.conductivity = conductivity
        else:
            raise RuntimeError('At least one of the following parameters ' + 
                             'should be provided: resistivity or conductivity')
                
        # Characteristic impedance of vacuum in* :math:`\Omega`
        self.Z0 = physical_constants['characteristic impedance of vacuum'][0]
            
    @property
    def resistivity(self):
        return self.__resistivity
        
    @resistivity.setter
    def resistivity(self, resistivity):
        self.__resistivity = resistivity
        self.__conductivity = 1 / resistivity

    @property
    def conductivity(self):
        return self.__conductivity
        
    @conductivity.setter
    def conductivity(self, conductivity):
        self.__resistivity = 1 / conductivity
        self.__conductivity = conductivity
    
    def imped_calc(self, frequency_array):
        r"""
        Impedance calculation method as a function of frequency.
        
        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz
        
        Attributes
        ----------
        frequency_array : float array
            Input frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        
        self.frequency_array = frequency_array
                
        self.impedance = (self.Z0 * c * self.pipe_length /
            (np.pi * (1.0 - 1j*np.sign(self.frequency_array)) * 2 *
            self.pipe_radius * c * np.sqrt(self.conductivity * self.Z0 * c /
            (4.0 * np.pi * np.abs(self.frequency_array)) )
            + 1j * self.pipe_radius**2.0 * 2.0 * np.pi * self.frequency_array))

        self.impedance[np.isnan(self.impedance)]= 0.0
