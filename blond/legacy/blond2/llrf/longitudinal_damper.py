# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Parent class to develop cavity feedback models and various cavity loops for the CERN machines**

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
'''


import logging
from abc import ABC

import numpy as np

from .signal_processing import longitudinal_damper_fir_filter


class LongitudinalDamper(ABC):

    def __init__(self):
        pass


class LHCLongitudinalDamper:
    r'''Documentation...'''

    def __init__(self, RFStation, Profile, store_turns=200, n_bunches=1, int_thres=1e-2, gain=10, action_delay=0):

        self.rfstation = RFStation
        self.profile = Profile

        # Simulation parameter
        self.n_bunches = n_bunches
        self.store_turns = store_turns
        self.int_thres = int_thres
        self.ld_filter = None
        self.gain = gain
        self.action_delay = action_delay

        # Arrays for the damper
        self.V_FIR_FILTERED = np.zeros(self.n_bunches, dtype=complex)
        self.V_SET_CORR = np.zeros(self.n_bunches, dtype=complex)
        self.FILLED_BUCKETS = np.zeros(self.n_bunches)
        self.BUNCH_PHASES = np.zeros(self.n_bunches)
        self.PHASE_CORRECTIONS = np.zeros(self.n_bunches)
        self.I_SET_CORR = None

        self.PHASE_BUFFER = np.zeros((self.n_bunches, self.store_turns), dtype=complex)

        self.compute_fir_filter()

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("LHCLongitudinalDamper class initialized")

    def track(self, i_rf_b, v_set):
        r'''Track the feedback model'''

        self.PHASE_BUFFER = np.roll(self.PHASE_BUFFER, -1, 1)

        # Find beam phase
        self.get_bunch_phases(i_rf_b, v_set)
        self.PHASE_BUFFER[:, -1] = self.BUNCH_PHASES

        # Apply a FIR filter
        self.apply_fir_filter_old()

        # Additional filter
        # TODO: implement a filter

        # Apply a gain
        self.PHASE_CORRECTIONS *= self.gain

        # Generate correction array
        self.I_SET_CORR = np.ones(len(i_rf_b), dtype=complex)
        self.I_SET_CORR[self.FILLED_BUCKETS] = np.exp(-1j * self.PHASE_CORRECTIONS)

    def compute_fir_filter(self):
        f_rev = (self.rfstation.omega_rf[0, self.rfstation.counter[0]] /
                 (2 * np.pi * self.rfstation.harmonic[0, self.rfstation.counter[0]]))
        f_s = self.rfstation.omega_s0[self.rfstation.counter] / (2 * np.pi)

        self.ld_filter = longitudinal_damper_fir_filter(f_s, f_rev)

    def get_bunch_phases(self, i_rf_b, v_set):
        self.FILLED_BUCKETS = np.argwhere(np.abs(i_rf_b) > self.int_thres)[:, 0]
        self.BUNCH_PHASES = np.angle(i_rf_b[self.FILLED_BUCKETS]) + np.pi/2 - (np.angle(v_set[self.FILLED_BUCKETS]))

    def apply_fir_filter_old(self):

        n_taps = len(self.ld_filter)
        filtered_signal = np.zeros((self.PHASE_BUFFER.shape[0], self.PHASE_BUFFER.shape[1] - n_taps),
                                   dtype=complex)
        for i in range(n_taps, self.PHASE_BUFFER.shape[1]):
            for k in range(n_taps):
                filtered_signal[:, i - n_taps] += self.ld_filter[k] * self.PHASE_BUFFER[:, i - k]

        self.PHASE_CORRECTIONS = filtered_signal[:, -1]

    def apply_fir_filter(self):

        n_taps = len(self.ld_filter)
        filtered_signal = np.zeros(self.PHASE_BUFFER.shape[0], dtype=complex)

        for i in range(n_taps):
            filtered_signal += self.PHASE_BUFFER[:, -2 - i] * self.ld_filter[i]

        self.PHASE_CORRECTIONS = filtered_signal
