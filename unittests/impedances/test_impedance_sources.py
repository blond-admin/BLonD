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

:Authors: **Simon Lauber**, **Markus Schwarz**
"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized
from scipy.constants import e as elCharge

from blond.beam.beam import Electron
from blond.impedances.impedance_sources import (
    CoherentSynchrotronRadiation,
    ResistiveWall,
    Resonators,
    _FftHandler,
    InputTableTimeDomain,
    InputTableFrequencyDomain,
    TravelingWaveCavity,
)

try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz

class Test_FftHandler(unittest.TestCase):
    def setUp(self):
        frequencies = np.linspace(0, 1e9, 500)
        index = len(frequencies) // 2
        amplitudes = np.zeros_like(frequencies, dtype=float)
        amplitudes[index] = 1
        self.sinus_freq = float(frequencies[index])
        self.fft_handler = _FftHandler(
            frequencies=frequencies,
            amplitudes=amplitudes,
        )

    def test___init__(self):
        pass  # tests only if `setUp` works

    def test_get_periodic_wake_by_time(self):
        time_array = np.linspace(0, 2 * 1 / self.sinus_freq, 500)
        wake = self.fft_handler.get_periodic_wake_by_time(time_array=time_array)
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(time_array, wake)
            plt.show()
        wake_expected = np.load(
            Path(__file__).parent.resolve()
            / Path("resources/test_get_periodic_wake_by_time.npy")
        )
        np.testing.assert_allclose(wake, wake_expected)

    def test_get_periodic_wake(self):
        ts_itp, wake_itp = self.fft_handler.get_periodic_wake(
            t_periodicity=1 / self.sinus_freq
        )
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(ts_itp, wake_itp)
            plt.show()

        wake_expected = np.load(
            Path(__file__).parent.resolve()
            / Path("resources/test_get_periodic_wake.npy")
        )
        np.testing.assert_allclose(wake_itp, wake_expected)

    def test_get_non_periodic_wake(self):
        time_array = np.linspace(0, 2 * 1 / self.sinus_freq, 500)
        wave = self.fft_handler.get_non_periodic_wake(time_array=time_array)

        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(time_array, wave)
            plt.show()

        wake_expected = np.load(
            Path(__file__).parent.resolve()
            / Path("resources/test_get_non_periodic_wake.npy")
        )
        np.testing.assert_allclose(wave, wake_expected)


class TestInputTableTimeDomain(unittest.TestCase):
    def setUp(self):
        time_array = np.linspace(0, 6, 100)
        wake = np.sin(time_array)
        self.input_table_time_domain = InputTableTimeDomain(
            time_array=time_array,
            wake=wake,
        )

    def test___init__(self):
        pass  # tests only if `setUp` works

    def test_wake_calc(self):
        self.input_table_time_domain.wake_calc(np.linspace(-1, 7, 100))

    def test_imped_calc(self):
        self.input_table_time_domain.imped_calc(
            frequency_array=np.linspace(0, 1 / (2 * 6 / 100))
        )


class TestInputTableFrequencyDomain(unittest.TestCase):
    def setUp(self):
        frequency_array = np.linspace(0, 1e9, 1000)
        Re_Z_array = np.random.rand(len(frequency_array))
        Im_Z_array = np.random.rand(len(frequency_array))
        self.input_table_frequency_domain = InputTableFrequencyDomain(
            frequency_array=frequency_array,
            Re_Z_array=Re_Z_array,
            Im_Z_array=Im_Z_array,
        )

    def test___init__(self):
        pass  # tests only if `setUp` works

    @parameterized.expand(["auto", 1.0, None])
    def test_wake_calc(self, t_periodicity):
        self.input_table_frequency_domain.t_periodicity = t_periodicity
        time_array = np.linspace(0, 1, 100)
        wake = self.input_table_frequency_domain.wake_calc(time_array=time_array)

    def test_imped_calc(self):
        frequency_array = np.linspace(0, 1, 100)
        self.input_table_frequency_domain.imped_calc(frequency_array=frequency_array)
        np.testing.assert_allclose(
            self.input_table_frequency_domain.frequency_array, frequency_array
        )


class TestTravelingWaveCavity(unittest.TestCase):
    def setUp(self):
        self.traveling_wave_cavity = TravelingWaveCavity(
            R_S=np.linspace(0, 1, 10),
            frequency_R=np.linspace(0, 1, 10),
            a_factor=np.linspace(0, 1, 10),
        )

    def test___init__(self):
        pass  # tests only if `setUp` works


class TestCoherentSynchrotronRadiation(unittest.TestCase):
    def setUp(self):
        self.coherent_synchrotron_ratiation = CoherentSynchrotronRadiation(
            r_bend=1.2,
            gamma=1.2,
            chamber_height=1.2,
        )

    def test___init__(self):
        pass  # tests only if `setUp` works


class TestResonators(unittest.TestCase):
    def test_smallQError(self):
        with self.assertRaises(RuntimeError):
            Resonators(1, 2, 0.2)


class TestResistiveWall(unittest.TestCase):
    def setUp(self):
        self.resistive_wall = ResistiveWall(
            pipe_radius=0.5, pipe_length=2, resistivity=1e-3
        )

    def test_noNecessaryKwargs(self):
        with self.assertRaises(RuntimeError):
            ResistiveWall(1, 2)

    def test___init__(self):
        pass  # tests only if `setUp` works

    @parameterized.expand(["auto", 1.0, None])
    def test_wake_calc(self, t_periodicity):
        self.resistive_wall.imped_calc(np.linspace(0, 5e9, 500))
        self.resistive_wall.t_periodicity = t_periodicity
        time_array = np.linspace(0, 1, 100)
        wake = self.resistive_wall.wake_calc(time_array=time_array)


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
            csr_imped.imped_calc.__func__, CoherentSynchrotronRadiation._fs_spectrum
        )

    def test_correctImpedanceFuncion3(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertTrue(hasattr(csr_imped, "f_cut"))
        self.assertEqual(
            csr_imped.imped_calc.__func__,
            CoherentSynchrotronRadiation._pp_low_frequency,
        )

    def test_correctImpedanceFuncion4(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42, chamber_height=4.2)

        self.assertTrue(hasattr(csr_imped, "f_crit"))
        self.assertTrue(hasattr(csr_imped, "f_cut"))
        self.assertEqual(
            csr_imped.imped_calc.__func__, CoherentSynchrotronRadiation._pp_spectrum
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
            ValueError, csr_imped.imped_calc, np.arange(5), low_frequency_transition=2
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

        r_bend, energy = 1.273, 40e6  # bending radius [m], particle energy [eV]
        gamma = energy / Electron().mass  # Lorentz factor

        # frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
        frequencies = 10 ** np.linspace(8, 15, num=200)

        Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
        Z_fs.imped_calc(frequencies, low_frequency_transition=1e-4)

        energy_loss = (
            2 * np.trapezoid(Z_fs.impedance.real, frequencies) * elCharge
        )  # [eV]

        energy_loss_textbook = Electron().c_gamma * energy**4 / r_bend  # [eV]

        self.assertAlmostEqual(energy_loss, energy_loss_textbook, places=3)

    def test_wakeIfFrontWake(self):
        # based on Example 22: Coherent Radiation

        r_bend, energy = 1.273, 40e6  # bending radius [m], particle energy [eV]
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
                    Z_fs.impedance * Lambda * np.exp(2j * np.pi * freqs * t), freqs
                ).real
            )

        # convert to volt
        W_fs *= elCharge * intensity

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
