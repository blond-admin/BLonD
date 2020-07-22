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
**Juan F. Esteban Mueller**, **Markus Schwarz**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from scipy.constants import c, physical_constants
from scipy.special import gamma as gamma_func
from scipy.special import kv, airy, polygamma
from scipy import integrate
import mpmath as mp
from ..utils import bmath as bm

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
        # WrongCalcError
        raise NotImplementedError('wake_calc() method not implemented in this class' +
                                  '. This object is probably meant to be used in the' +
                                  ' frequency domain')

    def imped_calc(self, *args):
        """
        Method required to compute the impedance. Returns an error if called
        from an object which does not implement this method.
        """
        # WrongCalcError
        raise NotImplementedError('imped_calc() method not implemented in this class' +
                                  '. This object is probably meant to be used in the' +
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

    def __init__(self, input_1, input_2, input_3=None):

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
                         self.Re_Z_array_loaded, right=0)
        Im_Z = np.interp(new_frequency_array, self.frequency_array_loaded,
                         self.Im_Z_array_loaded, right=0)
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
            # WrongCalcError
            raise RuntimeError(
                'method for impedance calculation in Resonator object not recognized')

    @property
    def frequency_R(self):
        return self.__frequency_R

    @frequency_R.setter
    def frequency_R(self, frequency_R):
        self.__frequency_R = frequency_R
        self.__omega_R = 2 * np.pi * frequency_R

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

            self.wake += ((np.sign(self.time_array) + 1) * self.R_S[i]
                         * alpha * np.exp(-alpha * self.time_array)
                          * (np.cos(omega_bar * self.time_array) - alpha /
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

            self.impedance[1:] += self.R_S[i] / (1 + 1j * self.Q[i]
                                                 * (self.frequency_array[1:] / self.frequency_R[i] -
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
        self.impedance = bm.fast_resonator(self.R_S, self.Q,
                                           self.frequency_array,
                                           self.frequency_R)


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
                                   * self.R_S[i] / a_tilde
                                   * (1 - self.time_array[indexes] / a_tilde)
                                   * np.cos(2 * np.pi * self.frequency_R[i] *
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
                                     (self.a_factor[i] / 2 * (self.frequency_array
                                                              + self.frequency_R[i])))**2 - 2j*(self.a_factor[i]
                                                                              * (self.frequency_array + self.frequency_R[i])
                - np.sin(self.a_factor[i] * (self.frequency_array
                                           + self.frequency_R[i]))) / (self.a_factor[i] *
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

    def __init__(self, pipe_radius, pipe_length, resistivity=None,
                 conductivity=None):

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
            # MissingParameterError
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

        self.impedance = (self.Z0 * c * self.pipe_length
                          / (np.pi * (1.0 - 1j*np.sign(self.frequency_array)) * 2 *
                             self.pipe_radius * c * np.sqrt(self.conductivity * self.Z0 * c
                                                            / (4.0 * np.pi * np.abs(self.frequency_array)) ) +
                           1j * self.pipe_radius**2.0 * 2.0 * np.pi * self.frequency_array))

        self.impedance[np.isnan(self.impedance)] = 0.0

class CoherentSynchrotronRadiation(_ImpedanceObject):


    def __init__(self, r_bend, gamma=None, chamber_height=np.inf):
        r"""
        

        Parameters
        ----------
        r_bend : float
            Bending radius in m
        gamma : float, optional
            The Lorentz factor is only used when the impedance is computed The default is None.
        chamber_height : float, optional
            The height of the vacuum chamber, i.e. the separation between the parallel plates. 
            `chamber_height` is required for the 
            The default is None.
        low_frequency_approximation : bool, optional
            If true, the low-frequency approximation for either the free-space or parallel 
            plates impedances are used. If false, the Lorentz factor `gamma` must be specified.
            The default is True.
        parallel_plates : TYPE, optional
            If ture, the parallel plates impedance is computed. In this case, `chamber_height`
            must be specified. If false, the free-space impedance is computed. The default is False.

        Raises
        ------
        ValueError
            Raised when input parameters are inconsistent, e.g. using the parallel plates impedance,
            but not specifying the `chamber_height`.
        RuntimeError
            Raised when the method for computing the impedance could not be deterimined

        Returns
        -------
        None.

        """
        _ImpedanceObject.__init__(self)
        
        # Characteristic impedance of vacuum in* :math:`\Omega`
        self.Z0 = physical_constants['characteristic impedance of vacuum'][0]
        
        self.r_bend = r_bend
        self.gamma = gamma
        self.chamber_height = chamber_height

        #TODO use f_0 = f_rev (from ring length) or f_0 = c/r_bend/2pi (only path in dipoles)?
        self.f_0 = 0.5 * c / self.r_bend / np.pi  # assumes beta == 1
        
        # test for input consistency
        if self.chamber_height <= 0.0:
            raise ValueError('chamber_height must be greater 0')
        if self.gamma is not None and self.gamma <= 1.0:
            raise ValueError('gamma must be greater 1 when using the full spectrum')
        
        #TODO check factor 2 in chamber_height
        if not np.isinf(self.chamber_height):
            self.Delta = self.chamber_height / self.r_bend
        
        if self.gamma is not None:
            # critical frequency in Hz
            self.f_crit = 0.75 * self.gamma**3 * c / r_bend / np.pi
        
        # chose proper impedance method based on input
        if self.gamma is None and np.isinf(self.chamber_height):
            self.imped_calc = self._fs_low_frequency_wrapper
        
        elif self.gamma is not None and np.isinf(self.chamber_height):
            self.imped_calc = self._fs_exact_spectrum
        
        elif self.gamma is None and not np.isinf(self.chamber_height):
            self.imped_calc = self._pp_low_frequency
            
        elif self.gamma is not None and not np.isinf(self.chamber_height):
            self.imped_calc = self._pp_exact_spectrum

        else:
            # WrongCalcError
            raise RuntimeError(
                'method for impedance calculation in CoherentSynchrotronRadiation object '
                +'not recognized')
            
        self.hyper_vec = np.vectorize(mp.hyper, excluded=[0,1], otypes=[float])


    def _pp_low_frequency(self, frequency_array, u_max=10, high_frequency_transition=np.inf):
        # based on eq. 8 of [Y. Cai]
        
        n_array = frequency_array / self.f_0
        
        # using high_frequency_transition factor of 1 definetely yields unphysical results
        if high_frequency_transition <= 1.0:
            raise ValueError('high frequency transition must be greater than 1')

        self.impedance = np.zeros_like(n_array, dtype=complex)
        
        # parallel plates cut-off frequency in units of f_0
        n_cut = np.sqrt(2/3) * (np.pi / self.Delta)**1.5
        
        # use approximate equation for frequencies above high_frequency_transition * n_cut
        approx_indexes = bm.where(n_array, more_than=high_frequency_transition * n_cut)

        self.impedance[approx_indexes] = self._fs_low_frequency(frequency_array[approx_indexes])

        # use exact result for all other frequencies
        exact_indexes = np.invert(approx_indexes)
        
        n_array = n_array[exact_indexes]  # override for convenience
        
        # maximum p(n) to have u(p,n)<u_max(n)
        pMax_array = np.array(np.ceil(self.Delta * n_array**(2/3) * np.sqrt(u_max) / (2**(2/3)*np.pi)
                                      - 0.5), dtype=int) 
        
        pMax = pMax_array[-1]  # maximum p; assumes largest frequency is at last array element
        
        p_matrix = np.zeros(shape=(len(n_array),pMax), dtype=int)

        # matrix to store the summands
        Z_matrix = np.zeros_like(p_matrix, dtype=complex)
        
        for nit, n in enumerate(n_array):
            # first element of p_matrix is 1 to ensure evaluation at u_min...
            #... if n is large enough so that u_min < 100, (i.e. airy(u_min) does not yield np.nan)
            if pMax_array[nit] == 0 and n > (np.pi/self.Delta)**1.5 / np.sqrt(2) / 100**0.75:
                p_matrix[nit,0] = 1
            else:
                p_matrix[nit,:pMax_array[nit]] = (2*np.arange(pMax_array[nit])+1)**2

        # evaluate Airy functions only at these values of p
        indexes = p_matrix > 0
        
        # argument of the Airy functions
        u_matrix = ((np.pi/2**(1/3)/self.Delta)**2 / n_array**(4/3) * p_matrix.T).T
        
        # evaluate Airy function only at relevant indexes
        airy_matrix = airy(u_matrix[indexes])  # returns Ai(u), Ai'(u), Bi(u), Bi'(u)
        
        Ci_matrix = airy_matrix[0] - 1j * airy_matrix[2]
        Ci_prime_matrix = airy_matrix[1] - 1j * airy_matrix[3]

        Z_matrix[indexes] = airy_matrix[1] * Ci_prime_matrix \
                                        + u_matrix[indexes] * airy_matrix[0] * Ci_matrix
        # sum over p
        self.impedance[exact_indexes] = np.sum(Z_matrix, axis=1)

        # the sum in eq. 8 is up to infinity, but we stop at p_max; the real part is exponentially
        # surpressed for large u, but the imaginary part is not; using the approximate expression
        # of the imaginary summands for large u, the remaining sum from p_max to infinity can be 
        # performed analytically by Mathematica 12.1.0.0.
        self.impedance[exact_indexes] += 1j * 2**(5/3) / (4096*np.pi**6) * (self.Delta * n_array**(2/3))**5 \
            * polygamma(4, pMax_array+1)

        self.impedance[exact_indexes] *= self.Z0 * 4*np.pi**2 * 2**(1/3) * 1/self.Delta / n_array**(1/3)


    # def _pp_exact_spectrum(self, frequency_array, zeta_max=9,
    #                        fs_low_frequency_transition=0, fs_high_frequency_transition=np.inf):
    def _pp_exact_spectrum(self, frequency_array, zeta_max=9, **kwargs):

        """
        Implements on eq. B13 of [Murphy et al.]
        """
        
        ns = frequency_array / self.f_0
        
        # Murphy et al. convention has plates at +/-h, wheras we use h for the full chamber height
        # => we need to use 0.5*self.Delta in their equations
        
        # based an eq. B8
        alphas = np.sqrt(2) * np.exp(-1j*np.pi/6) * ns**(2/3) * 0.5*self.Delta / 3**(1/6)
        
        # sets self.impedance to the full free-space impedance
        # self._fs_exact_spectrum(frequency_array,
        #                         low_frequency_transition=fs_low_frequency_transition,
        #                         high_frequency_transition=fs_high_frequency_transition)
        self._fs_exact_spectrum(frequency_array, **kwargs)


        # subtract low-frequency free-space impedance
        self.impedance -= self._fs_low_frequency(frequency_array)
            
        # parallel plates part
        Z_pp = np.zeros_like(ns, dtype=complex)

        # for it, n in enumerate(ns):
        #     pMax = int(np.ceil(np.sqrt(9 *2 / 3**(1/3))/np.pi  * 0.5*self.Delta * n**(2/3) - 0.5))
        #     # print(pMax)
        #     if pMax == 0:
        #         continue
        #     # maximum p ensures that Re(z_array)<9; see documentation of _hFun
        #     p_array = np.arange(pMax)
        # #     # z_array = 0.25 * 3**(1/3) * np.pi**2 * np.exp(1j*np.pi/3) \
        # #     #     * ( (2*p_array+1) * self.r_bend / self.chamber_height / n**(2/3) )**2
        # #     print(it, len(p_array))
        #     z_array = 0.5 * ( np.pi * (2*p_array+1) / alphas[it] )**2
        #     # print(pMax, z_array[-1])
        #     Z_pp[it] = np.sum( self._hFunNum(z_array) )
        #     # Z_pp[it] = np.sum( self._hFun(z_array) )
        #     Z_pp[it] += np.sqrt(np.pi) / (32*3**(5/6)) * np.exp(1j*np.pi/6) * (0.5*self.Delta * n**(2/3) / np.pi)**5\
        #         * polygamma(4, pMax+0.5)
        
        # maximum p(n) to have zeta(p,n)<zeta_max(n)
        pMax_array = np.array(np.ceil(np.sqrt(zeta_max / 3**(1/3))/np.pi * 0.5*self.Delta * ns**(2/3)
                                      - 0.5), dtype=int)
        # print(pMax_array[-1])

        for it, n in enumerate(ns):
            # # maximum p ensures that Re(z_array)<15; see documentation of _hFun
            # pMax = int(np.ceil(np.sqrt(9 / 3**(1/3))/np.pi  * 0.5*self.Delta * n**(2/3) - 0.5))
            pMax = pMax_array[it]
            # print(pMax)
            if pMax <= 0:
                continue
            p_array = np.arange(pMax)
        #     # z_array = 0.25 * 3**(1/3) * np.pi**2 * np.exp(1j*np.pi/3) \
        #     #     * ( (2*p_array+1) * self.r_bend / self.chamber_height / n**(2/3) )**2
        #     print(it, len(p_array))
            z_array = 0.5 * ( np.pi * (2*p_array+1) / alphas[it] )**2
            # print(z_array[0] * np.exp(-1j*np.pi/3), z_array[-1] * np.exp(-1j*np.pi/3))
            Z_pp[it] = np.sum( self._hFun(z_array) )
        
        Z_pp += np.sqrt(np.pi) / (32*3**(5/6)) * np.exp(1j*np.pi/6) * (0.5*self.Delta * ns**(2/3) / np.pi)**5\
            * polygamma(4, pMax_array+0.5)
        
        Z_pp *= self.Z0 * 2*(2*np.pi)**0.5 * 3**(2/3) * np.exp(1j*np.pi/6) * ns**(1/3) / alphas

        self.impedance += Z_pp

    def _pp_exact_spectrum2(self, frequency_array, zeta_max=9, **kwargs):
        """
        Implements on eq. B13 of [Murphy et al.]
        """
        
        n_array = frequency_array / self.f_0
        
        # Murphy et al. convention has plates at +/-h, wheras we use h for the full chamber height
        # => we need to use 0.5*self.Delta in their equations
        
        # based an eq. B8
        alphas = np.sqrt(2) * np.exp(-1j*np.pi/6) * n_array**(2/3) * 0.5*self.Delta / 3**(1/6)
        
        # sets self.impedance to the full free-space impedance
        self._fs_exact_spectrum2(frequency_array, **kwargs)

        # subtract low-frequency free-space impedance
        self.impedance -= self._fs_low_frequency(frequency_array)
            
        # parallel plates part
        
        # maximum p(n) to have zeta(p,n)<zeta_max(n)
        pMax_array = np.array(np.ceil(np.sqrt(zeta_max / 3**(1/3))/np.pi * 0.5*self.Delta
                                      * n_array**(2/3) - 0.5), dtype=int)

        pMax = pMax_array[-1]  # maximum p; assumes largest frequency is at last array element
        
        p_matrix = np.zeros(shape=(len(n_array),pMax), dtype=int)

        # matrix to store the summands
        Z_matrix = np.zeros_like(p_matrix, dtype=complex)

        for nit, n in enumerate(n_array):
            # first element of p_matrix is 1 to ensure evaluation at zeta_min...
            #... if n is large enough so that zeta_min < zeta_max
            if pMax_array[nit] == 0 and n > 3**0.25 * (np.pi/(0.5*self.Delta))**1.5 / zeta_max**0.75:
                p_matrix[nit,0] = 1
            else:
                p_matrix[nit,:pMax_array[nit]] = (2*np.arange(pMax_array[nit])+1)**2

        # evaluate h function only at these values of p
        indexes = p_matrix > 0

        z_matrix = (0.5 * ( np.pi / alphas )**2 * p_matrix.T).T
        
        Z_matrix[indexes] = self._hFun(z_matrix[indexes])
        
        Z_pp = np.sum(Z_matrix, axis=1)
        
        Z_pp += np.sqrt(np.pi) / (32*3**(5/6)) * np.exp(1j*np.pi/6)\
                * (0.5*self.Delta * n_array**(2/3) / np.pi)**5\
                * polygamma(4, pMax_array+0.5)
        
        Z_pp *= self.Z0 * 2*(2*np.pi)**0.5 * 3**(2/3) * np.exp(1j*np.pi/6) * n_array**(1/3) / alphas

        self.impedance += Z_pp    


    def _hFun(self, z):
        r"""
        Implements eq. B14 of [Murphy et al.]. 
        
        The integral in eq. B14 was solved analytically by Mathematica 12.1.0.0. However,
        the analytic solution in terms of Airy functions ist numerically unstable for $\zeta>15$,
        with $z=\exp(\i\pi/3)\zeta$. Numerically solving eq. B14, gives a negligable contribution 
        for $zeta > 15.0$, and the analytic solution is, thus, sufficient for our purposes.

        Parameters
        ----------
        z : complex array
            Argument of the function. Numerically unstable for Re z > 10.0.
            This condition is not checked.

        Returns
        -------
        complex array
            h(z)

        """

        airy_array = airy( -z / 12**(1/3) )  # returns Ai(), Ai'(), Bi(), Bi'()
        return - np.pi**1.5 / (2**(2/3) * 3**(5/6))\
            * (z * (airy_array[0]**2 + airy_array[2]**2) / 12**(1/3)
               - airy_array[1]**2 - airy_array[3]**2)
    
    def _hFunNum(self, z):
        re = integrate.quad_vec(lambda nu: np.real(nu**1.5 * np.exp(-nu**3 - nu*z)), 0, np.inf)[0]
        im = integrate.quad_vec(lambda nu: np.imag(nu**1.5 * np.exp(-nu**3 - nu*z)), 0, np.inf)[0]
        
        return re + 1j*im


    def _fs_exact_spectrum(self, frequency_array,
                           low_frequency_transition=0, high_frequency_transition=np.inf):
        """
        Computes the exact free-space synchrotron radiation impedance, based on eqs. A4 and A5 of 
        [Murphy et. al]. For computation speed and numerical stability, the approximate expressions
        are used for frequencies lower / higher than the critical frequency f_c. 

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to evaluate the impedance
        high_frequency_transition : float, optional
            Ratio of f/f_c above which the high frequency approximation is used. If it is smaller
            than 1, a `ValueError` is raised. The default is `np.inf`, i.e. the approximation is not 
            used.
        low_frequency_transition : float, optional
            Ratio of f/f_c below which the low frequency approximation is used. If it is greater
            than 1, a `ValueError` is raised. The default is 0, i.e. the approximation is not used.

        Raises
        ------
        ValueError
            Raised if high_frequency_transition is smaller than the low_frequency_transition

        Returns
        -------
        None.

        """
        
        # check for input consistency        
        if high_frequency_transition < 1.0:
            raise ValueError('high_frequency_transition ratio must be greater than 1')

        if low_frequency_transition > 1.0:
            raise ValueError('low_frequency_transition ratio must be smaller than 1')

        if high_frequency_transition <= low_frequency_transition:
            raise ValueError('high_frequency_transition ratio must be larger than the'
                             +'low_frequency_transition rato')

        
        self.impedance = np.zeros_like(frequency_array, dtype=complex)
        
        l_array = frequency_array / self.f_crit
        
        # use the low frequency approximation where f < LFT * f_c
        low_indexes = bm.where(l_array, less_than=low_frequency_transition)
        self.impedance[low_indexes] = self._fs_low_frequency(frequency_array[low_indexes])
        # index from which to start compute the  exact spectrum
        exact_start_index = np.sum(low_indexes)

        # use the high frequency approximation where f > HFT * f_c
        high_indexes = bm.where(l_array, more_than=high_frequency_transition)
        self.impedance[high_indexes] = self._fs_high_frequency(frequency_array[high_indexes])

        # use full integration for frequencies inbetween
        exact_indexes = np.invert(low_indexes + high_indexes)
        
        # need to loop over frequencies because integrade.quad does not accept arrays :(
        for it, l in enumerate(l_array[exact_indexes]):
            self.impedance[exact_start_index + it] = \
                0.25 * np.sqrt(3) * integrate.quad(self._fs_integrandReZ, l, np.inf)[0]\
                + 1j * (integrate.quad(self._fs_integrandImZ1, 0, 1, args=l)[0] 
                        - integrate.quad(self._fs_integrandImZ2, 1, np.inf, args=l)[0] ) 
        
        self.impedance[exact_indexes] *= self.Z0 * self.gamma * l_array[exact_indexes]

    def _fs_exact_spectrum2(self, frequency_array, epsilon=1e-6,
                           low_frequency_transition=0, high_frequency_transition=np.inf):
        """
        Computes the exact free-space synchrotron radiation impedance, based on eqs. A4 and A5 of 
        [Murphy et. al]. For computation speed and numerical stability, the approximate expressions
        are used for frequencies lower / higher than the critical frequency f_c. 

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to evaluate the impedance
        high_frequency_transition : float, optional
            Ratio of f/f_c above which the high frequency approximation is used. If it is smaller
            than 1, a `ValueError` is raised. The default is `np.inf`, i.e. the approximation is not 
            used.
        low_frequency_transition : float, optional
            Ratio of f/f_c below which the low frequency approximation is used. If it is greater
            than 1, a `ValueError` is raised. The default is 0, i.e. the approximation is not used.

        Raises
        ------
        ValueError
            Raised if high_frequency_transition is smaller than the low_frequency_transition

        Returns
        -------
        None.

        """
        
        # check for input consistency        
        if high_frequency_transition < 1.0:
            raise ValueError('high_frequency_transition ratio must be greater than 1')

        if low_frequency_transition > 1.0:
            raise ValueError('low_frequency_transition ratio must be smaller than 1')

        if high_frequency_transition <= low_frequency_transition:
            raise ValueError('high_frequency_transition ratio must be larger than the'
                             +'low_frequency_transition rato')
        
        if epsilon < 0 or epsilon > 1:
            raise ValueError('epsilon must be a small positive value')

        
        self.impedance = np.zeros_like(frequency_array, dtype=complex)
        
        l_array = frequency_array / self.f_crit
        
        # use the low frequency approximation where f < LFT * f_c
        low_indexes = bm.where(l_array, less_than=low_frequency_transition)
        self.impedance[low_indexes] = self._fs_low_frequency(frequency_array[low_indexes])
        # # index from which to start compute the  exact spectrum
        # exact_start_index = np.sum(low_indexes)

        # use the high frequency approximation where f > HFT * f_c
        high_indexes = bm.where(l_array, more_than=high_frequency_transition)
        self.impedance[high_indexes] = self._fs_high_frequency(frequency_array[high_indexes])

        # use full integration for frequencies inbetween
        exact_indexes = np.invert(low_indexes + high_indexes)
        
        # Real part: eq. A4, is solved analytically with Mathematica 12.0 in terms of 
        # generalized hypergeometric functions
        # Imaginary part: quad_vec can't handle the integrable singularity at y=1 for y<1, we need
        # to integrate up to 1-epsilon
        self.impedance[exact_indexes] =\
            np.sqrt(3) * gamma_func(2/3) / l_array[exact_indexes]**(2/3) / 2**(4/3)\
                * self.hyper_vec([-1/3], [-2/3,2/3], 0.25*l_array[exact_indexes]**2) \
            + 81*np.pi * l_array[exact_indexes]**(8/3) / (640*2**(2/3)*gamma_func(-1/3))\
                * self.hyper_vec([4/3], [7/3,8/3], 0.25*l_array[exact_indexes]**2)\
            - 0.25 * np.pi\
            + 1j* (integrate.quad_vec(lambda y: self._fs_integrandImZ1(y, l_array[exact_indexes]),
                                      0, 1-epsilon)[0]
                   - integrate.quad_vec(lambda y: self._fs_integrandImZ2(y, l_array[exact_indexes]),
                                        1, np.inf)[0])
        
        self.impedance[exact_indexes] *= self.Z0 * self.gamma * l_array[exact_indexes]
    
    def _fs_low_frequency_wrapper(self, frequency_array):
        """
        Wrapper to compute the free-space low-frequency approximation of the synchrotron
        radiation impedance, based on eq. 6.18 of [Murphy et al.]

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        None.

        """
        
        self.impedance = self._fs_low_frequency(frequency_array)
            
    def _fs_low_frequency(self, frequency_array):
        """
        Computes the free-space low-frequency approximation of the synchrotron radiation impedance,
        based on eq. 6.18 of [Murphy et al.]. This is a helper function.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        complex array
            impedance
        """
        
        return self.Z0 * gamma_func(2/3) / 3**(1/3) * np.exp(1j*np.pi/6) \
            * (frequency_array / self.f_0)**(1/3)
        
    def _fs_high_frequency(self, frequency_array):
        """
        Computes the free-space high-frequency approximation of the synchrotron radiation impedance,
        based on eq. 6.20 of [Murphy et al.]. This function is a helper function for 
        _fs_full_spectrum.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        complex array
            impedance
        """
        
        return self.Z0 * self.gamma * (np.sqrt(3*np.pi/32)\
             * np.sqrt(frequency_array/self.f_crit) * np.exp(-frequency_array/self.f_crit)\
             - 4j/9 * self.f_crit / frequency_array )
    
    
    def _fs_integrandReZ(self,x):
        # integrand of real part of free-space impedance
        return kv(5/3, x)


    def _fs_integrandImZ1(self,y,l):
        # integrand of imaginary part of free-space impedance for y<1
        return np.real( np.exp(-l*y) * (
            (1j*y+np.sqrt(1-y**2))**(5/3) + 1/(1j*y+np.sqrt(1-y**2))**(5/3) -2 * np.sqrt(1-y**2))\
            / (4*y*np.sqrt(1-y**2)) )


    def _fs_integrandImZ2(self,y, l):
        # integrand of imaginary part of free-space impedance for y>1
        return np.exp(-l*y) * (
            8*np.sqrt(y**2-1) - (y+np.sqrt(y**2-1))**(5/3) + 1/(y+np.sqrt(y**2-1))**(5/3))\
            / (8*y*np.sqrt(y**2-1) )


