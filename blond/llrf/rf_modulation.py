
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to generate RF phase modulation from given frequency, amplitude
and offset functions**

:Authors: **Simon Albright**
'''

#General imports
import numpy as np
import scipy.interpolate as interp

#BLonD imports
import blond.utils.data_check as dCheck
import blond.utils.exceptions as blExcept



class PhaseModulation:
    
    def __init__(self, timebase, frequency, amplitude, offset, \
                 harmonic, multiplier = 1, modulate_frequency = True):
        
        
        msg = "must be a single numerical value or have shape (2, n)"
        dCheck.check_input(timebase, "Timebase must have shape (n)", [-1])
        dCheck.check_input(frequency, "Frequency " + msg, 0, (2, -1))
        dCheck.check_input(amplitude, "Amplitude " + msg, 0, (2, -1))
        dCheck.check_input(offset, "Offset " + msg, 0, (2, -1))
        dCheck.check_input(multiplier, "Multiplier " + msg, 0, (2, -1))
        dCheck.check_input(harmonic, "Harmonic must be single valued number", 0)

        self.timebase = timebase        
        self.frequency = frequency
        self.amplitude = amplitude
        self.offset = offset        
        self.multiplier = multiplier
        self.harmonic = harmonic

        if not isinstance(modulate_frequency, bool):
            raise TypeError("modulate_frequency must be boolean")
        
        self._mod_freq = modulate_frequency


#Calculate the modulation with linear interpolation of functions
    def calc_modulation(self):
        
        amplitude = self._interp_param(self.amplitude)
        frequency = self._interp_param(self.frequency)
        offset = self._interp_param(self.offset)
        multiplier = self._interp_param(self.multiplier)

        frequency *= multiplier

        self.dphi = amplitude \
                    * np.sin(2*np.pi*(np.cumsum(frequency \
                                                *np.gradient(self.timebase))))\
                    + offset



    def calc_delta_omega(self, omegaProg):

        dCheck.check_input(omegaProg, "omegaProg must have shape (2, n)", \
                           (2, -1))
        
        if not self._mod_freq:
            self.domega = np.zeros(len(self.dphi))
        
        else:   
            omega = self._interp_param(omegaProg)
            self.domega = np.gradient(self.dphi) * omega \
                          / (2*np.pi * self.harmonic)
            


#Interpolate functions onto self.timebase
    def _interp_param(self, param):
        
        if dCheck.check_data_dimensions(param, 0)[0]:
            return np.array([param]*len(self.timebase))
        
        elif dCheck.check_data_dimensions(param, (2, -1))[0]:
            return np.interp(self.timebase, param[0], param[1])
        
        else:
            raise TypeError("Param must be number or have shape (2, n)")

         
#Extend passed parameter to requred n_rf if n_rf > 1 for treatment in
#rf_parameters
    def extend_to_n_rf(self, harmonics):

        try:
            n_rf = len(harmonics)
        except TypeError:
            n_rf = 1
            harmonics = [harmonics]

        if self.harmonic not in harmonics:
            raise AttributeError("self.harmonic not in harmonics")
        
        if not hasattr(self, 'domega'):
            raise AttributeError("""domega has not yet been calculated, 
                                 calc_delta_omega must be called first""")

        if n_rf == 1:
            return (self.timebase, self.dphi), (self.timebase, self.domega)

        else:
            extendTuple = ([self.timebase[0], self.timebase[-1]], [0, 0])
            return (tuple([self.timebase, self.dphi] \
                         if self.harmonic == harmonics[i] \
                         else extendTuple for i in range(n_rf)), 
                         
                    tuple([self.timebase, self.domega] \
                         if self.harmonic == harmonics[i] \
                         else extendTuple for i in range(n_rf)))
    
    
    
    
    
    
    
    
    
    




class OldPhaseModulation:

    def __init__(self, frequency, amplitude, offset, time, freqMultiplier = 1, \
                 interpType = 'cubic', smoothing = 1000, \
                 ontime = 0, offtime = 0):

        '''
        Generate phase modulation to be applied to phi_rf
        frequency, amplitude and offset.  All three will be
        interpolated onto the passed time array.  They can
        either be a single value, a list of 2 points or a
        2D numpy array with array[0] as time and array[1] as
        the function.  If a list of 2 points is passed it
        will be assumed that the first and last points are
        at the beginning and end of the time array
    
        freqMultiplier is a multiplier applied to the modulation
        frequency allowing e.g. a fixed multiple of synchrotron
        frequency to be used.
        '''

        if not isinstance(frequency, np.ndarray):
            if isinstance(frequency, list):
                frequency = np.array([[time[0], time[-1]], [frequency[0], frequency[-1]]])
            else:
                frequency = np.array([[time[0], time[-1]], [frequency, frequency]])

        else:
            early = np.where(frequency[0] < time[0])[0]
            late = np.where(frequency[0] > time[-1])[0]

            if len(early) == 0:
                early = 0
            else:
                early = early[-1]
            if len(late) == 0:
                late = -1
            else:
                late = late[0]

            newFreq = frequency[1][early:late]
            newTime = frequency[0][early:late]

            frequency = np.zeros([2, len(newFreq)])
            frequency[0] = newTime
            frequency[1] = newFreq

        frequency[1] *= freqMultiplier

        if not isinstance(amplitude, np.ndarray):
            if isinstance(amplitude, list):
                amplitude = np.array([[time[0], time[-1]], [amplitude[0], amplitude[-1]]])
            else:
                amplitude = np.array([time, [amplitude]*len(time)])

        if not isinstance(offset, np.ndarray):
            if isinstance(offset, list):
                offset = np.array([[time[0], time[-1]], [offset[0], offset[-1]]])
            else:
                offset = np.array([time, [offset]*len(time)])

        self.time = time

        if interpType == 'cubic':
            freqInterp = interp.splrep(frequency[0], frequency[1], s=smoothing)
            ampInterp = interp.splrep(amplitude[0], amplitude[1], s=smoothing)
            offInterp = interp.splrep(offset[0], offset[1], s=smoothing)

            self.frequency = interp.splev(self.time, freqInterp)
            self.amplitude = interp.splev(self.time, ampInterp)
            self.offset = interp.splev(self.time, offInterp)

        elif interpType == 'linear':
            self.frequency = np.interp(self.time, frequency[0], frequency[1])
            self.amplitude = np.interp(self.time, amplitude[0], amplitude[1])
            self.offset = np.interp(self.time, offset[0], offset[1])

        else:
            print("Interpolation type not recognised, defaulting to cubic spline")
            freqInterp = interp.splrep(frequency[0], frequency[1], s=smoothing)
            ampInterp = interp.splrep(amplitude[0], amplitude[1], s=smoothing)
            offInterp = interp.splrep(offset[0], offset[1], s=smoothing)

            self.frequency = interp.splev(self.time, freqInterp)
            self.amplitude = interp.splev(self.time, ampInterp)
            self.offset = interp.splev(self.time, offInterp)

        self.ontime = ontime
        self.offtime = offtime



def ModulateHarmonic(Ring, RFStation, modulationList, harmonic, includeFreq = True, preshift = 0):


    '''
    Method to apply modulation(s) to RFStation
    modulationList is a list of modulation objects applied
    sequentially to RFStation.

    harmonicList specifies which harmonic of RFStation
    will be modulated

    includeFreq flag determines if effect of modulation on
    RF frequency should be included.  False will treat
    modulation as pure phase shift, True will include the
    small frequency shift needed to adjust the phase.
    '''


    if isinstance(preshift, np.ndarray):
        if len(preshift.shape) == 2:
            preshift = prep.preprocess_rf_params(Ring, [preshift[0]], [preshift[1]], plot=False)[0]
        elif len(preshift) == 2:
            preshift = prep.preprocess_rf_params(Ring, [[Ring.cumulative_times[0], Ring.cumulative_times[-1]]], [[preshift[0], preshift[1]]], plot=False)[0]
        else:
            print("ERROR: preshift array is incorrectly shaped")
    elif isinstance(preshift, list):
        if len(preshift) == 2:
            preshift = prep.preprocess_rf_params(Ring, [[Ring.cumulative_times[0], Ring.cumulative_times[-1]]], [[preshift[0], preshift[1]]], plot=False)[0]
        else:
            print("ERROR: preshift list is the wrong length")

    fullPhiAddition = np.zeros(len(Ring.cycle_time)) + preshift
    
    startPoints = []
    stopPoints = []

    for i in range(len(modulationList)):
        
        modulation = modulationList[i]

        modOff = prep.preprocess_rf_params(Ring, [[modulation.time[0] - modulation.ontime] + modulation.time.tolist() + [modulation.time[-1] + modulation.offtime]], [[0] + modulation.offset.tolist() + [0]], plot=False)
        modDepth = prep.preprocess_rf_params(Ring, [modulation.time], [modulation.amplitude], plot=False)
        modFreq = prep.preprocess_rf_params(Ring, [modulation.time], [modulation.frequency], plot=False)

        start = np.where(Ring.cumulative_times < modulation.time[0])[0]
        if len(start) == 0:
            start = [0, 1]
        stop = np.where(Ring.cumulative_times > modulation.time[-1])[0]
        if len(stop)==0:
            stop = [-1, -2]

        modDepth[0][:start[-2]] = 0
        modDepth[0][stop[1]:] = 0

        phiAddition = (modDepth*np.sin(2*np.pi*(np.cumsum(modFreq*np.gradient(Ring.cumulative_times)))) + modOff)[0]

        fullPhiAddition[start[-2]:stop[1]] += phiAddition[start[-2]:stop[1]]

        startPoints.append(start[-1])
        stopPoints.append(stop[0])

    RFStation.phi_RF[harmonic] += fullPhiAddition

    if includeFreq:

        freqAddition = (np.gradient(fullPhiAddition)*RFStation.omega_RF[harmonic])/(2*np.pi*RFStation.harmonic[harmonic])
        RFStation.omega_RF[harmonic] += freqAddition
