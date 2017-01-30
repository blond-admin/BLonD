
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to describe impedance classes to be used by InducedVoltage* objects**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from scipy.constants import c, physical_constants


class _ImpedanceObject(object):
    '''
    Parent impedance object to implement required methods and attributes
    '''
    
    def __init__(self):        
        #: *Time array of the wake in s*
        self.time_array = 0
        
        #: *Wake array in* :math:`\Omega / s`
        self.wake = 0
        
        #: *Frequency array of the impedance in Hz*
        self.frequency_array = 0
        
        #: *Impedance array in* :math:`\Omega`
        self.impedance = 0
        
    
    def wake_calc(self, *args):
        '''
        *Method required to compute the wake function. Returns an error if
        called from an object which does not implement this method.*
        '''
        
        raise RuntimeError('wake_calc() method not implemented in this class'+
                           '. This object is probably meant to be used in the'+
                           ' frequency domain')
    
    
    def imped_calc(self, *args):    
        '''
        *Method required to compute the impedance. Returns an error if called
        from an object which does not implement this method.*
        '''
        
        raise RuntimeError('imped_calc() method not implemented in this class'+
                           '. This object is probably meant to be used in the'+
                           ' time domain')
    

class InputTable(_ImpedanceObject):
    '''
    *Intensity effects from impedance and wake tables.
    If this constructor takes just two arguments, then a wake table is passed;
    if it takes three arguments, then an impedance table is passed. Be careful
    that if you input a wake, the input wake for W(t=0) should be already 
    divided by two (beam loading theorem) ; and that if you input impedance, 
    only the positive  frequencies of the impedance is needed (the impedance
    will be assumed to be Hermitian (Real part symmetric and Imaginary part
    antisymmetric).Note that we add the point (f, Z(f)) = (0, 0) to the 
    frequency and impedance arrays derived from the table.*
    '''
    
    def __init__(self, input_1, input_2, input_3 = None):    
        
        _ImpedanceObject.__init__(self)   

        if input_3 is None:
            #: *Time array of the wake in s*
            self.time_array = input_1
            #: *Wake array in* :math:`\Omega / s`
            self.wake_array = input_2
        else:
            #: *Frequency array of the impedance in Hz*
            self.frequency_array_loaded = input_1
            #: *Real part of impedance in* :math:`\Omega`
            self.Re_Z_array_loaded = input_2
            #: *Imaginary part of impedance in* :math:`\Omega`
            self.Im_Z_array_loaded = input_3
            #: *Impedance array in* :math:`\Omega`
            self.impedance_loaded = (self.Re_Z_array_loaded + 1j * 
                                     self.Im_Z_array_loaded)
            
            if self.frequency_array_loaded[0] != 0:
                self.frequency_array_loaded = np.hstack((0, 
                                                  self.frequency_array_loaded))
                self.Re_Z_array_loaded = np.hstack((0, self.Re_Z_array_loaded))
                self.Im_Z_array_loaded = np.hstack((0, self.Im_Z_array_loaded))
    
    
    def wake_calc(self, new_time_array):
        '''
        *The wake is interpolated in order to scale with the new time array.*
        '''
        
        self.wake = np.interp(new_time_array, self.time_array, self.wake_array, 
                           right=0)
                           
    
    def imped_calc(self, new_frequency_array):
        '''
        *The impedance is interpolated in order to scale with the new frequency
        array.*
        '''

        Re_Z = np.interp(new_frequency_array, self.frequency_array_loaded, 
                         self.Re_Z_array_loaded, right = 0)
        Im_Z = np.interp(new_frequency_array, self.frequency_array_loaded, 
                         self.Im_Z_array_loaded, right = 0)
        self.frequency_array = new_frequency_array
        self.Re_Z_array = Re_Z
        self.Im_Z_array = Im_Z
        self.impedance = Re_Z + 1j * Im_Z
        
    
    
class Resonators(_ImpedanceObject):
    '''
    *Impedance contribution from resonators, analytic formulas for both wake 
    and impedance. The resonant modes (and the corresponding R and Q) 
    can be inputed as a list in case of several modes.*
    
    *The model is the following:*
    
    .. math::
    
        Z(f) = \\frac{R}{1 + j Q \\left(\\frac{f}{f_r}-\\frac{f_r}{f}\\right)}
        
    .. math::
        
        W(t>0) = 2\\alpha R e^{-\\alpha t}\\left(\\cos{\\bar{\\omega}t} - \\frac{\\alpha}{\\bar{\\omega}}\\sin{\\bar{\\omega}t}\\right)

        W(0) = \\alpha R
        
    .. math::
        
        \\omega_r = 2 \\pi f_r
        
        \\alpha = \\frac{\\omega_r}{2Q}
        
        \\bar{\\omega} = \\sqrt{\\omega_r^2 - \\alpha^2}
        
    '''
    
    def __init__(self, R_S, frequency_R, Q):
        
        _ImpedanceObject.__init__(self)

        #: *Shunt impepdance in* :math:`\Omega`
        self.R_S = np.array([R_S]).flatten()
        
        #: *Resonant frequency in Hz*
        self.frequency_R = np.array([frequency_R]).flatten()
                
        #: *Quality factor*
        self.Q = np.array([Q]).flatten()
        
        #: *Number of resonant modes*
        self.n_resonators = len(self.R_S)    
    
    @property
    def frequency_R(self):
        return self.__frequency_R
        
    @frequency_R.setter
    def frequency_R(self, frequency_R):
        self.__frequency_R = frequency_R
        self.__omega_R = 2 *np.pi * frequency_R

    #: *Resonant angular frequency in rad/s*
    @property
    def omega_R(self):
        return self.__omega_R
        
    @omega_R.setter
    def omega_R(self, omega_R):
        self.__frequency_R = omega_R / 2 / np.pi
        self.__omega_R = omega_R
        
    def wake_calc(self, time_array):
        '''
        *Wake calculation method as a function of time.*
        '''
        
        self.time_array = time_array
        self.wake = np.zeros(self.time_array.shape)
        
        for i in range(0, self.n_resonators):
       
            alpha = self.omega_R[i] / (2 * self.Q[i])
            omega_bar = np.sqrt(self.omega_R[i] ** 2 - alpha ** 2)
            
            self.wake += ((np.sign(self.time_array) + 1) * self.R_S[i] *
                         alpha * np.exp(-alpha * self.time_array) *
                         (np.cos(omega_bar * self.time_array) - alpha /
                          omega_bar * np.sin(omega_bar * self.time_array)))
    
    
    def imped_calc(self, frequency_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array), complex)
        
        for i in range(0, self.n_resonators):
            
            self.impedance[1:] += self.R_S[i] / (1 + 1j * self.Q[i] *
                             (self.frequency_array[1:] / self.frequency_R[i] - 
                              self.frequency_R[i] / self.frequency_array[1:]))
 
 

class TravelingWaveCavity(_ImpedanceObject):
    '''
    *Impedance contribution from traveling wave cavities, analytic formulas for 
    both wake and impedance. The resonance modes (and the corresponding R and a) 
    can be inputed as a list in case of several modes.*
    
    *The model is the following:*
    
    .. math::
    
        Z_+(f) = R \\left[\\left(\\frac{\\sin{\\frac{a\\left(f-f_r\\right)}{2}}}{\\frac{a\\left(f-f_r\\right)}{2}}\\right)^2 - 2i \\frac{a\\left(f-f_r\\right) - \\sin{a\\left(f-f_r\\right)}}{\\left(a\\left(f-f_r\\right)\\right)^2}\\right]
        
        Z_-(f) = R \\left[\\left(\\frac{\\sin{\\frac{a\\left(f+f_r\\right)}{2}}}{\\frac{a\\left(f+f_r\\right)}{2}}\\right)^2 - 2i \\frac{a\\left(f+f_r\\right) - \\sin{a\\left(f+f_r\\right)}}{\\left(a\\left(f+f_r\\right)\\right)^2}\\right]
        
        Z = Z_+ + Z_-
        
    .. math::
        
        W(0<t<\\tilde{a}) = \\frac{4R}{\\tilde{a}}\\left(1-\\frac{t}{\\tilde{a}}\\right)\\cos{\\omega_r t} 

        W(0) = \\frac{2R}{\\tilde{a}}
        
    .. math::
        
        a = 2 \\pi \\tilde{a}
        
    '''
    
    def __init__(self, R_S, frequency_R, a_factor):
        
        _ImpedanceObject.__init__(self)
        
        #: *Shunt impepdance in* :math:`\Omega`
        self.R_S = np.array([R_S]).flatten()
        
        #: *Resonant frequency in Hz*
        self.frequency_R = np.array([frequency_R]).flatten()
        
        #: *Damping time a in s*
        self.a_factor = np.array([a_factor]).flatten()
        
        #: *Number of resonant modes*
        self.n_twc = len(self.R_S)
        
    
    def wake_calc(self, time_array):
        '''
        *Wake calculation method as a function of time.*
        '''
        
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
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
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
    '''
    *Impedance contribution from resistive wall for a cilindrical beam pipe*
    
    *The model is the following:*
    
    .. math::
    
        Z(f) = \\frac{Z_0 c L}{ \\pi } \\frac{ 1 }{ \\left[1 - i \\sign{f}\\right] 2  b  c \\sqrt{ \\frac{\\sigma_c Z_0 c }{ 4 \\pi |f| } + i 2 \\pi b^2 f }
        
    '''
    
    def __init__(self, pipe_radius, pipe_length, resistivity = None, 
                 conductivity = None):
        
        _ImpedanceObject.__init__(self)
        
        #: *Beam pipe radius in m*
        self.pipe_radius = pipe_radius
        
        #: *Beam pipe length in m*
        self.pipe_length = pipe_length
        
        #: *Beam pipe conductivity in * :math:`s / m`
        if resistivity != None:
            self.conductivity = 1/resistivity
        elif conductivity != None:
            self.conductivity = conductivity
        else:
            raise RuntimeError('At least one of the following parameters ' + 
                             'should be provided: resistivity or conductivity')
                
        #: *Characteristic impedance of vacuum in* :math:`\Omega`
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
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.frequency_array = frequency_array
                
        self.impedance = (self.Z0 * c * self.pipe_length /
            (np.pi * (1.0 - 1j*np.sign(self.frequency_array)) * 2 *
            self.pipe_radius * c * np.sqrt(self.conductivity * self.Z0 * c /
            (4.0 * np.pi * np.abs(self.frequency_array)) )
            + 1j * self.pipe_radius**2.0 * 2.0 * np.pi * self.frequency_array))

        self.impedance[np.isnan(self.impedance)]= 0.0
