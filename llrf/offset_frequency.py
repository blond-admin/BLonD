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
import matplotlib.pyplot as plt
import scipy.constants as cont


class _FrequencyOffset(object):
    '''
    Compute effect of having a different RF and design frequency
    '''

    def __init__(self, Ring, RFStation, System = None):

        #: | *Import Ring*
        self.ring = Ring

        #: | *Import RFStation*
        self.rf_station = RFStation

        #: | *Set system to modify, if None all are modified*
        self.system = System


    def set_frequency(self, NewFrequencyProgram):

        '''
        Set new frequency program
        '''

        #: | *Check of frequency is passed as array of [time, freq]*
        if isinstance(NewFrequencyProgram, np.ndarray):
            if NewFrequencyProgram.shape[0] == 2:
                end_turn = np.where(self.ring.cycle_time >= \
                                    NewFrequencyProgram[0][-1])[0][0]
                NewFrequencyProgram = np.interp(self.ring.cycle_time[:end_turn],\
                                 NewFrequencyProgram[0], NewFrequencyProgram[1])

        #: | *Store new frequency as numpy array*
        self.new_frequency = np.array(NewFrequencyProgram)

        self.end_turn = len(self.new_frequency)

        #: | *Store design frequency during offset*
        self.design_frequency = self.rf_station.omega_rf_d[:,:self.end_turn]


    def calculate_phase_slip(self):

        '''
        Calculate the phase slippage resulting from the frequency offset for \
        each RF system
        '''

        delta_phi = 2*np.pi*self.rf_station.harmonic[:,:self.end_turn]*(self.rf_station.harmonic[:,:self.end_turn]*self.new_frequency - self.design_frequency)/\
                       self.design_frequency

        self.phase_slippage = np.cumsum(delta_phi, axis=1)


    def apply_new_frequency(self):

        '''
        Sets the RF frequency and phase
        '''

        if self.system is None:
            self.rf_station.omega_rf[:, :self.end_turn] = self.rf_station.harmonic[:, :self.end_turn]*self.new_frequency
            self.rf_station.phi_rf[:, :self.end_turn] += self.phase_slippage

            for n in range(self.rf_station.n_rf):
                self.rf_station.phi_rf[n, self.end_turn:] += self.phase_slippage[n,-1]

        else:
            self.rf_station.omega_rf[self.system, :self.end_turn] = self.rf_station.harmonic[self.system, :self.end_turn]*self.new_frequency
            self.rf_station.phi_rf[self.system, :self.end_turn] += self.phase_slippage[self.system]
            self.rf_station.phi_rf[self.system, self.end_turn:] += self.phase_slippage[self.system,-1]



class FixedFrequency(_FrequencyOffset):
    '''
    Compute effect of fixed RF frequency different to frequency from momentum
    program at the start of the cycle.
    '''

    def __init__(self, Ring, RFStation, FixedFrequency, FixedDuration,
                 TransitionDuration):

        _FrequencyOffset.__init__(self, Ring, RFStation)

        #: | *Set value of fixed frequency*
        self.fixed_frequency = FixedFrequency

        #: | *Duration of fixed frequency*
        self.fixed_duration = FixedDuration

        #: | *Duration of transition to design frequency*
        self.transition_duration = TransitionDuration

        self.end_fixed_turn = np.where(self.ring.cycle_time >= \
                       self.fixed_duration)[0][0]
        self.end_transition_turn = np.where(self.ring.cycle_time >= \
                      (self.fixed_duration + self.transition_duration))[0][0]

        self.end_frequency = self.rf_station.omega_rf_d[0, self.end_transition_turn]

        self.calculate_frequency_prog()
        self.set_frequency(self.frequency_prog)
        self.calculate_phase_slip()
        self.apply_new_frequency()


    def calculate_frequency_prog(self):

        '''
        Calculate the fixed and transition frequency programs turn by turn
        '''

        fixed_frequency_prog = np.ones(self.end_fixed_turn)*self.fixed_frequency
        transition_frequency_prog = np.linspace(self.fixed_frequency, \
                                 self.end_frequency, \
                                 self.end_transition_turn - self.end_fixed_turn)

        self.frequency_prog = np.concatenate((fixed_frequency_prog, \
                                             transition_frequency_prog))



    def transition_1(self):

        def delta_f(a3, a2, a1, B, B0):
            return a3*(B-B0)**3 + a2*(B-B0)**2 + a1*(B-B0)

        def FofB(K1, K2, B):
            return K1/np.sqrt(1 + (K2/B)**2)

        def FofBP(K1, K2, B):
            return (K1*K2**2)/(B**3 * (1+(K2/B)**2 )**(3/2))

        fixed_frequency_prog = np.ones(self.end_fixed_turn)*self.fixed_frequency
        BField = self.ring.momentum[0][:self.end_transition_turn]
        eft = self.end_fixed_turn

        deltaFProg = []

        for t in range(self.end_fixed_turn, self.end_transition_turn):

            omega_t = self.rf_station.omega_rf_d[0][t]
            omega_in = 2*np.pi*self.fixed_frequency

            a1 = 0
            a2 = - (3*omega_in - 2*FofB(omega_in, omega_t, BField[-1]) - BField[eft]*FofBP(omega_in, omega_t, BField[-1]) + BField[-1]*FofBP(omega_in, omega_t, BField[-1])) / ((BField[eft] - BField[-1])**2)
            a3 = - (2*omega_in - 2*FofB(omega_in, omega_t, BField[-1]) - BField[eft]*FofBP(omega_in, omega_t, BField[-1]) + BField[-1]*FofBP(omega_in, omega_t, BField[-1])) / ((BField[eft] - BField[-1])**3)

            deltaFProg.append(delta_f(a3, a2, a1, BField[t], BField[eft]))

        plt.plot(np.array(deltaFProg))
        plt.plot(self.rf_station.omega_rf_d[0][eft:self.end_transition_turn])
        plt.show()





