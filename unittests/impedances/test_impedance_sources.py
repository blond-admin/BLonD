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

try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz
from scipy.constants import e as e_charge

from blond.beam.beam import Electron
from blond.impedances.impedance_sources import (
    CoherentSynchrotronRadiation,
    ResistiveWall,
    Resonators,
    _ImpedanceObject,
)


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

        self.assertEqual(
            csr_imped.imped_calc.__func__,
            CoherentSynchrotronRadiation._fs_low_frequency_wrapper,
        )

    def test_correctImpedanceFuncion2(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42)

        self.assertTrue(hasattr(csr_imped, "f_crit"))
        self.assertEqual(
            csr_imped.imped_calc.__func__,
            CoherentSynchrotronRadiation._fs_spectrum,
        )

    def test_correctImpedanceFuncion3(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertTrue(hasattr(csr_imped, "f_cut"))
        self.assertEqual(
            csr_imped.imped_calc.__func__,
            CoherentSynchrotronRadiation._pp_low_frequency,
        )

    def test_correctImpedanceFuncion4(self):
        csr_imped = CoherentSynchrotronRadiation(
            1, gamma=42, chamber_height=4.2
        )

        self.assertTrue(hasattr(csr_imped, "f_crit"))
        self.assertTrue(hasattr(csr_imped, "f_cut"))
        self.assertEqual(
            csr_imped.imped_calc.__func__,
            CoherentSynchrotronRadiation._pp_spectrum,
        )

    def test_lowHighFrequencyTransitionFreeSpace(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42)

        self.assertRaises(
            ValueError,
            csr_imped.imped_calc,
            np.arange(5),
            high_frequency_transition=0.2,
        )
        self.assertRaises(
            ValueError,
            csr_imped.imped_calc,
            np.arange(5),
            low_frequency_transition=2,
        )
        self.assertRaises(
            ValueError,
            csr_imped.imped_calc,
            np.arange(5),
            low_frequency_transition=1.1,
            high_frequency_transition=1,
        )

    def test_lowHighFrequencyTransitionApproxPP(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertRaises(
            ValueError,
            csr_imped.imped_calc,
            np.arange(5),
            high_frequency_transition=0.2,
        )

    def test_energyLoss(self):
        # based on Example 22: Coherent Radiation

        r_bend, energy = (
            1.273,
            40e6,
        )  # bending radius [m], particle energy [eV]
        gamma = energy / Electron().mass  # Lorentz factor

        # frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
        frequencies = 10 ** np.linspace(8, 15, num=200)

        Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
        Z_fs.imped_calc(frequencies, low_frequency_transition=1e-4)

        energy_loss = (
            2 * np.trapezoid(Z_fs.impedance.real, frequencies) * e_charge
        )  # [eV]

        energy_loss_textbook = Electron().c_gamma * energy**4 / r_bend  # [eV]

        self.assertAlmostEqual(energy_loss, energy_loss_textbook, places=3)

    def test_wakeIfFrontWake(self):
        # based on Example 22: Coherent Radiation

        r_bend, energy = (
            1.273,
            40e6,
        )  # bending radius [m], particle energy [eV]
        gamma = energy / Electron().mass  # Lorentz factor

        # bunch intensity and length
        intensity, sigma_dt = 7e6, 3e-12  # 1, [s]

        # times where to compute wake potential
        times = np.linspace(-2e-11, 2e-11, num=11)

        # frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
        freqs = 10 ** np.linspace(8, 15, num=200)

        Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
        Z_fs.imped_calc(freqs, high_frequency_transition=10)

        # Fourier transform of Gaussian bunch profile
        Lambda = np.exp(-0.5 * (2 * np.pi * freqs * sigma_dt) ** 2)

        W_fs = np.zeros_like(times)
        for it, t in enumerate(times):
            W_fs[it] = (
                2
                * np.trapezoid(
                    Z_fs.impedance * Lambda * np.exp(2j * np.pi * freqs * t),
                    freqs,
                ).real
            )

        # convert to volt
        W_fs *= e_charge * intensity

        W_fs_test = np.array(
            [
                -2.99526459e01,
                -4.11849124e01,
                -6.39549552e01,
                -1.15710237e02,
                -2.71278552e01,
                +4.34850900e02,
                +2.48528445e02,
                +1.86894225e01,
                -2.58048424e-01,
                -2.86820828e-01,
                -2.03095038e-01,
            ]
        )

        np.testing.assert_allclose(W_fs, W_fs_test)


if __name__ == "__main__":
    unittest.main()
