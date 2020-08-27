# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for impedances.impedance_sources

:Authors: **Markus Schwarz**
"""

import unittest
import numpy as np

from blond.impedances.impedance_sources import _ImpedanceObject, Resonators, ResistiveWall,\
    CoherentSynchrotronRadiation


class Test_ImpedanceObject(unittest.TestCase):

    def setUp(self):
        self.test_object = _ImpedanceObject()

    def test_notImplemented(self):

        self.assertRaises(NotImplementedError, self.test_object.imped_calc)
        self.assertRaises(NotImplementedError, self.test_object.wake_calc)


class TestResonators(unittest.TestCase):

    def test_smallQError(self):
        with self.assertRaises(RuntimeError):
            Resonators(1, 2, 0.2)

    def test_wrongMethodError(self):
        with self.assertRaises(RuntimeError):
            Resonators(1, 2, 3, method='something')


class TestResistiveWall(unittest.TestCase):

    def test_noNecessaryKwargs(self):
        with self.assertRaises(RuntimeError):
            ResistiveWall(1, 2)


class TestCoherentSynchrotronRadiation(unittest.TestCase):

    def test_wrongBendingRadius(self):
        with self.assertRaises(ValueError):
            CoherentSynchrotronRadiation(-1)
        with self.assertRaises(ValueError):
            CoherentSynchrotronRadiation(0)

    def test_wrongChamberHeight(self):
        with self.assertRaises(ValueError):
            CoherentSynchrotronRadiation(1, chamber_height=0)
        with self.assertRaises(ValueError):
            CoherentSynchrotronRadiation(1, chamber_height=-1)

    def test_wrongGamma(self):
        with self.assertRaises(ValueError):
            CoherentSynchrotronRadiation(1, gamma=0.1)

    def test_correctImpedanceFuncion1(self):
        csr_imped = CoherentSynchrotronRadiation(1)

        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._fs_low_frequency_wrapper)

    def test_correctImpedanceFuncion2(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42)

        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._fs_spectrum)

    def test_correctImpedanceFuncion3(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._pp_low_frequency)

    def test_correctImpedanceFuncion4(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42, chamber_height=4.2)

        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._pp_spectrum)

    def test_lowHighFrequencyTransitionFreeSpace(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42)

        self.assertRaises(ValueError, csr_imped.imped_calc, np.arange(5),
                          high_frequency_transition=0.2)
        self.assertRaises(ValueError, csr_imped.imped_calc, np.arange(5),
                          low_frequency_transition=2)
        self.assertRaises(ValueError, csr_imped.imped_calc, np.arange(5),
                          low_frequency_transition=1.1, high_frequency_transition=1)

    def test_lowHighFrequencyTransitionApproxPP(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertRaises(ValueError, csr_imped.imped_calc, np.arange(5),
                          high_frequency_transition=0.2)


if __name__ == '__main__':

    unittest.main()
