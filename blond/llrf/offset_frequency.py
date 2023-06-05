# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Frequency corrections to design frequency to allow fixed injection frequency
and frequency offsets**

:Authors: **Simon Albright**
'''

import numpy as np


class _FrequencyOffset:
    '''
    Compute effect of having a different RF and design frequency
    '''

    def __init__(self, Ring, RFStation, System=None, MainH=None):

        #: | *Import Ring*
        self.ring = Ring

        #: | *Import RFStation*
        self.rf_station = RFStation

        #: | *Set system number(s) to modify, if None all are modified*
        if isinstance(System, int):
            self.system = [System]
        elif hasattr(System, '__iter__'):
            self.system = []
            for s in System:
                self.system.append(s)
        elif System is None:
            self.system = System
        else:
            raise TypeError("System must be int, iterable of ints or None")

        if self.system and not all((isinstance(s, int) for s in self.system)):
            raise TypeError("System must be int, iterable of ints or None")

        #: | *Main harmonic the delta F is taken as being in reference to,
        #: |  if None RFStation.harmonic[0][0] is taken as the main*
        if MainH is not None:
            self.mainH = MainH
        else:
            self.mainH = RFStation.harmonic[0][0]

    def set_frequency(self, NewFrequencyProgram):
        '''
        Set new frequency program
        '''

        #: | *Check of frequency is passed as array of [time, freq]*
        if isinstance(NewFrequencyProgram, np.ndarray):
            if NewFrequencyProgram.shape[0] == 2:
                end_turn = np.where(self.ring.cycle_time >=
                                    NewFrequencyProgram[0][-1])[0][0]
                NewFrequencyProgram = np.interp(self.ring.cycle_time[:end_turn],
                                                NewFrequencyProgram[0], NewFrequencyProgram[1])

        #: | *Store new frequency as numpy array relative to the main harmonic*
        self.new_frequency = np.array(NewFrequencyProgram) / self.mainH

        self.end_turn = len(self.new_frequency)

        #: | *Store design frequency during offset*
        self.design_frequency = self.rf_station.omega_rf_d[:, :self.end_turn]

    def calculate_phase_slip(self):
        '''
        Calculate the phase slippage resulting from the frequency offset for \
        each RF system
        '''

        delta_phi = (2 * np.pi * self.rf_station.harmonic[:, :self.end_turn]
                     * (self.rf_station.harmonic[:, :self.end_turn]
                         * self.new_frequency
                         - self.design_frequency)
                     / self.design_frequency)
        self.phase_slippage = np.cumsum(delta_phi, axis=1)

    def apply_new_frequency(self):
        '''
        Sets the RF frequency and phase
        '''

        if self.system is None:
            self.rf_station.omega_rf[:, :self.end_turn] = \
                (self.rf_station.harmonic[:, :self.end_turn]
                 * self.new_frequency)

            self.rf_station.phi_rf[:, :self.end_turn] += self.phase_slippage

            for n in range(self.rf_station.n_rf):
                self.rf_station.phi_rf[n, self.end_turn:] \
                    += self.phase_slippage[n, -1]

        else:
            for system in self.system:
                self.rf_station.omega_rf[system, :self.end_turn] \
                    = (self.rf_station.harmonic[system, :self.end_turn]
                        * self.new_frequency)
                self.rf_station.phi_rf[system, :self.end_turn] \
                    += self.phase_slippage[system]
                self.rf_station.phi_rf[system, self.end_turn:] \
                    += self.phase_slippage[system, -1]


class FixedFrequency(_FrequencyOffset):
    '''
    Compute effect of fixed RF frequency different to frequency from momentum
    program at the start of the cycle.
    '''

    def __init__(self, Ring, RFStation, FixedFrequency, FixedDuration,
                 TransitionDuration, transition=1):

        _FrequencyOffset.__init__(self, Ring, RFStation)

        #: | *Set value of fixed frequency*
        self.fixed_frequency = FixedFrequency

        #: | *Duration of fixed frequency*
        self.fixed_duration = FixedDuration

        #: | *Duration of transition to design frequency*
        self.transition_duration = TransitionDuration

        self.end_fixed_turn = np.where(self.ring.cycle_time >=
                                       self.fixed_duration)[0][0]
        self.end_transition_turn = np.where(self.ring.cycle_time >=
                                            (self.fixed_duration + self.transition_duration))[0][0]

        self.end_frequency = self.rf_station.omega_rf_d[0, self.end_transition_turn]

        if transition == 1:
            self.calculate_frequency_prog = self.transition_1

        self.compute()

    def compute(self):

        self.calculate_frequency_prog()
        self.set_frequency(self.frequency_prog)
        self.calculate_phase_slip()
        self.apply_new_frequency()

    def linear_calculate_frequency_prog(self):
        '''
        Calculate the fixed and transition frequency programs turn by turn
        '''

        fixed_frequency_prog = np.ones(self.end_fixed_turn) * self.fixed_frequency
        transition_frequency_prog = np.linspace(float(self.fixed_frequency),
                                                float(self.end_frequency),
                                                (self.end_transition_turn
                                                - self.end_fixed_turn))

        self.frequency_prog = np.concatenate((fixed_frequency_prog,
                                             transition_frequency_prog))

    def transition_1(self):

        t1 = (self.ring.cycle_time[self.end_transition_turn]
              - self.ring.cycle_time[self.end_fixed_turn])
        f1 = self.end_frequency
        f1Prime = (np.gradient(self.rf_station.omega_rf_d[0])
                   / np.gradient(self.ring.cycle_time))[self.end_transition_turn]

        constA = (t1 * f1Prime - 2 * (f1 - self.fixed_frequency)) / t1**3
        constB = - (t1 * f1Prime - 3 * (f1 - self.fixed_frequency)) / t1**2

        transTime = (self.ring.cycle_time[self.end_fixed_turn:self.end_transition_turn]
                     - self.ring.cycle_time[self.end_fixed_turn])

        transition_freq = (constA * transTime**3 + constB * transTime**2
                           + self.fixed_frequency)

        self.frequency_prog = np.concatenate((np.ones(self.end_fixed_turn)
                                              * self.fixed_frequency, transition_freq))
