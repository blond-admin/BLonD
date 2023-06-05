# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
"""

import unittest

import numpy as np

import blond.llrf.rf_modulation as rfMod
import blond.utils.exceptions as blExcept


class TestRFModulation(unittest.TestCase):

    def test_construct(self):

        timebase = np.linspace(0, 1, 100)

        stringMsg = "Integer input should raise an InputDataError exception"
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(1, 1, 1, 1, 1)

        stringMsg = "String input should raise an InputDataError exception"
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation('a', 1, 1, 1, 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(timebase, 'a', 1, 1, 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(timebase, 1, 'a', 1, 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(timebase, 1, 1, 'a', 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(timebase, 1, 1, 1, 'a')

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(timebase, 1, 1, 1, 1, 'a')

        stringMsg = "Wrong shape input should raise an InputDataError exception"
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(np.zeros([2, 2]), 1, 1, 1, 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(np.zeros(2), [1, 2, 3], 1, 1, 1)

        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(np.zeros(2), 1, np.zeros([3, 100]), 1, 1, 1)

        modulator = rfMod.PhaseModulation(timebase, 1, 1, 1, 1)
        self.assertEqual((modulator.timebase, modulator.frequency,
                          modulator.amplitude, modulator.offset,
                          modulator.multiplier, modulator.harmonic),
                         (timebase, 1, 1, 1, 1, 1),
                         msg="Input has not been applied correctly")

        with self.assertRaises(TypeError,
                               msg='Non-boolean input should raise TypeError'):

            rfMod.PhaseModulation(np.zeros(2), 1, 1, 1, 1, 1, "Not a bool")

    def test_interpolation(self):

        timebase = np.linspace(0, 1, 1000)

        modulator = rfMod.PhaseModulation(timebase, 1, 1, 1, 1)

        self.assertTrue(all(modulator._interp_param([[0, 1], [0, 1]])
                            == timebase), msg='Function interpolation incorrect')

        self.assertTrue(all(modulator._interp_param(1) == 1),
                        msg='Single valued interpolation incorrect')

        with self.assertRaises(TypeError,
                               msg='Error should be raised for wrong shape data'):
            modulator._interp_param([1, 0])

    def test_modulation(self):

        timebase = np.linspace(0, 1, 1000)
        testFreqProg = [[0, 1], [20, 5]]
        testAmpProg = [[0, 0.5, 1], [0, 2, 0]]
        testOffsetProg = [[0, 1], [0, np.pi]]
        testMultProg = 2
        harmonic = 8

        modulator = rfMod.PhaseModulation(timebase, testFreqProg,
                                          testAmpProg, testOffsetProg,
                                          harmonic, testMultProg)

        modulator.calc_modulation()

        self.assertEqual(modulator.dphi[0], 0,
                         msg='Start phase should be 0')

        self.assertEqual(modulator.dphi[-1], np.pi,
                         msg='Start phase should be np.pi')

        self.assertAlmostEqual(np.max(modulator.dphi), 3.55908288285, 5,
                               msg='Max dphi is incorrect')

    def test_delta_omega(self):

        timebase = np.linspace(0, 1, 1000)

        freqProg = np.array([np.linspace(0, 1, 10000),
                             np.linspace(1E6, 1E6, 10000)])

        modulator = rfMod.PhaseModulation(timebase, 1, 1, 1, 1)

        modulator.dphi = np.linspace(0, np.pi / 2, 250).tolist() \
            + [np.pi / 2] * 500 \
            + np.linspace(np.pi / 2, 0, 250).tolist()

        with self.assertRaises(blExcept.InputDataError,
                               msg='wrong shape frequency should raise Error'):
            modulator.calc_delta_omega(np.zeros([3, 100]))

        modulator.calc_delta_omega(freqProg)

        self.assertEqual(np.sum(modulator.domega), 0,
                         msg="Trapezoid dphi should give sum(domega) == 0")

        modulator.dphi = [np.pi / 2] * 1000
        modulator.calc_delta_omega(freqProg)

        self.assertEqual(modulator.domega.tolist(), [0] * len(timebase),
                         msg="Constant dphi should have domega == 0")

    def test_extender(self):

        timebase = np.linspace(0, 1, 1000)
        testFreqProg = [[0, 1], [1E3, 5E2]]
        testAmpProg = [[0, 0.5, 1], [0, 1, 0]]
        testOffsetProg = [[0, 1], [0, np.pi]]
        testMultProg = 2
        harmonic = 8
        freqProg = np.array([np.linspace(0, 1, 10000),
                             np.linspace(1E6, 2E6, 10000)])

        modulator = rfMod.PhaseModulation(timebase, testFreqProg,
                                          testAmpProg, testOffsetProg,
                                          harmonic, testMultProg,
                                          modulate_frequency=False)

        modulator.calc_modulation()

        with self.assertRaises(AttributeError,
                               msg="""Attribute error should be raised
                               before domega has been calculated"""):
            modulator.extend_to_n_rf(8)

        modulator.calc_delta_omega(freqProg)

        with self.assertRaises(AttributeError,
                               msg="""AttrubuteError should be raised if
                               modulator.harmonic not in passed harmonics"""):
            dPhi, dOmega = modulator.extend_to_n_rf([1, 3, 5])

        dPhi, dOmega = modulator.extend_to_n_rf([1, 3, 5, 7, 8])

        self.assertEqual(len(dPhi), 5,
                         msg="dPhi Not correctly extended to n_rf")

        self.assertEqual(len(dPhi), 5,
                         msg="dOmega not correctly extended to n_rf")

        for i in range(5):
            self.assertEqual(len(dPhi[i]), 2,
                             msg="All dPhi members should have length 2")
            self.assertEqual(len(dOmega[i]), 2,
                             msg="All dOmega members should have length 2")

            if i != 4:

                self.assertEqual(dPhi[i][1], [0, 0],
                                 msg="Unused system dPhi should be [0, 0]")
                self.assertEqual(dOmega[i][1], [0, 0],
                                 msg="Unused system dOmega should be [0, 0]")

            else:
                self.assertEqual(dPhi[i][1].tolist(),
                                 modulator.dphi.tolist(),
                                 msg="Used dPhi should match dPhi")
                self.assertEqual(dOmega[i][1].tolist(),
                                 [0] * len(timebase),
                                 msg="""Used dOmega should be 0 with
                                 modulate_frequency = False""")


if __name__ == '__main__':

    unittest.main()
