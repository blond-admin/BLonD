
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

# General imports
import numpy as np

# BLonD imports
import blond.utils.data_check as dCheck


class PhaseModulation:

    def __init__(self, timebase, frequency, amplitude, offset,
                 harmonic, multiplier=1, modulate_frequency=True):

        msg = "must be a single numerical value or have shape (2, n)"
        dCheck.check_input(timebase, "Timebase must have shape (n)", [-1])
        dCheck.check_input(frequency, "Frequency " + msg, 0, (2, -1))
        dCheck.check_input(amplitude, "Amplitude " + msg, 0, (2, -1))
        dCheck.check_input(offset, "Offset " + msg, 0, (2, -1))
        dCheck.check_input(multiplier, "Multiplier " + msg, 0, (2, -1))
        dCheck.check_input(
            harmonic, "Harmonic must be single valued number", 0)

        self.timebase = timebase
        self.frequency = frequency
        self.amplitude = amplitude
        self.offset = offset
        self.multiplier = multiplier
        self.harmonic = harmonic

        if not isinstance(modulate_frequency, bool):
            raise TypeError("modulate_frequency must be boolean")

        self._mod_freq = modulate_frequency

    # Calculate the modulation with linear interpolation of functions
    def calc_modulation(self):

        amplitude = self._interp_param(self.amplitude)
        frequency = self._interp_param(self.frequency)
        offset = self._interp_param(self.offset)
        multiplier = self._interp_param(self.multiplier)

        frequency *= multiplier

        self.dphi = amplitude \
            * np.sin(2 * np.pi * (np.cumsum(frequency
                                            * np.gradient(self.timebase))))\
            + offset

    def calc_delta_omega(self, omegaProg):

        dCheck.check_input(omegaProg, "omegaProg must have shape (2, n)",
                           (2, -1))

        if not self._mod_freq:
            self.domega = np.zeros(len(self.dphi))

        else:
            omega = self._interp_param(omegaProg)
            self.domega = np.gradient(self.dphi) * omega \
                / (2 * np.pi * self.harmonic)

    # Interpolate functions onto self.timebase
    def _interp_param(self, param):

        if dCheck.check_data_dimensions(param, 0)[0]:
            return np.array([param] * len(self.timebase))

        elif dCheck.check_data_dimensions(param, (2, -1))[0]:
            return np.interp(self.timebase, param[0], param[1])

        else:
            raise TypeError("Param must be number or have shape (2, n)")

    # Extend passed parameter to requred n_rf if n_rf > 1 for treatment in
    # rf_parameters
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
            return (tuple([self.timebase, self.dphi]
                          if self.harmonic == harmonics[i]
                          else extendTuple for i in range(n_rf)),

                    tuple([self.timebase, self.domega]
                          if self.harmonic == harmonics[i]
                          else extendTuple for i in range(n_rf)))
