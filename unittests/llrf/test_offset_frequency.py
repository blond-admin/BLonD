# -*- coding: utf-8 -*-

from __future__ import division, print_function

import unittest

import numpy as np

import blond.llrf.offset_frequency as offFreq
from blond.beam.beam import Proton
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


class TestOffsetFrequency(unittest.TestCase):

    def setUp(self):

        # Machine and RF parameters
        radius = 25
        gamma_transition = 4.4  # [1]
        C = 2 * np.pi * radius  # [m]
        momentum_compaction = 1 / gamma_transition**2  # [1]

        self.ring = Ring(C, momentum_compaction,
                         ([0, 25E-3], [3.13E8, 3.5E8]), Proton())

        self.rf_params = RFStation(self.ring, [1, 2], [0, 0],
                                   [np.pi, np.pi * 0.95], 2)

    def test_fixed_freq(self):

        self.injection_frequency = offFreq.FixedFrequency(self.ring,
                                                          self.rf_params, self.rf_params.omega_rf_d[0][0], 2E-3, 1.5E-3)

        self.assertEqual(len(self.injection_frequency.frequency_prog),
                         self.injection_frequency.end_transition_turn,
                         msg='Fixed frequency + transition wrong length')

        self.assertEqual(len(self.injection_frequency.phase_slippage[0]),
                         self.injection_frequency.end_transition_turn,
                         msg='Phase slippage wrong length')

        self.assertEqual(self.rf_params.omega_rf_d[0][0],
                         self.injection_frequency.frequency_prog[0],
                         msg='Fixed frequency initial value wrong')

        self.assertEqual(self.rf_params.omega_rf_d[0][0],
                         self.injection_frequency.frequency_prog[self.injection_frequency.end_fixed_turn],
                         msg='Fixed frequency final value wrong')

        self.assertAlmostEqual(self.rf_params.omega_rf_d[0][self.injection_frequency.end_transition_turn - 1],
                               self.injection_frequency.frequency_prog[-1],
                               delta=1, msg='Fixed frequency end transition value wrong')

        self.assertEqual(self.injection_frequency.phase_slippage[0][0], 0,
                         msg='Phase slippage not starting at 0 for system 0')

        self.assertEqual(self.injection_frequency.phase_slippage[1][0], 0,
                         msg='Phase slippage not starting at 0 for system 1')

        self.assertSequenceEqual(self.injection_frequency.frequency_prog.tolist(),
                                 self.rf_params.omega_rf[0][:self.injection_frequency.end_transition_turn].tolist(
        ),
            msg='rf_params.omega_rf not equal to injection frequency')

        self.assertAlmostEqual(self.rf_params.phi_rf[0][-1], -56.304959088,
                               places=5, msg='System 1 end phase wrong')
        self.assertAlmostEqual(self.rf_params.phi_rf[1][-1], -115.908590462,
                               places=5, msg='System 1 end phase wrong')

    def test_exceptions(self):

        with self.assertRaises(TypeError, msg='Non-integer systems should raise TypeError'):
            offFreq._FrequencyOffset(self.ring, self.rf_params, 1.)
        with self.assertRaises(TypeError, msg='Non-integer memebered iterables should raise TypeError'):
            offFreq._FrequencyOffset(self.ring, self.rf_params, ['a'])


if __name__ == '__main__':

    unittest.main()
