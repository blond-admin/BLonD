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
from scipy.constants import e as elCharge

from blond.beam.beam import Electron
from blond.impedances.impedance_sources import (CoherentSynchrotronRadiation,
                                                ResistiveWall, Resonators,
                                                _ImpedanceObject,
                                                TravelingWaveCavity,
                                                InputTable)


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

        self.assertTrue(hasattr(csr_imped, 'f_crit'))
        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._fs_spectrum)

    def test_correctImpedanceFuncion3(self):
        csr_imped = CoherentSynchrotronRadiation(1, chamber_height=42)

        self.assertTrue(hasattr(csr_imped, 'f_cut'))
        self.assertEqual(csr_imped.imped_calc.__func__,
                         CoherentSynchrotronRadiation._pp_low_frequency)

    def test_correctImpedanceFuncion4(self):
        csr_imped = CoherentSynchrotronRadiation(1, gamma=42, chamber_height=4.2)

        self.assertTrue(hasattr(csr_imped, 'f_crit'))
        self.assertTrue(hasattr(csr_imped, 'f_cut'))
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

    def test_energyLoss(self):
        # based on Example 22: Coherent Radiation

        r_bend, energy = 1.273, 40e6  # bending radius [m], particle energy [eV]
        gamma = energy / Electron().mass  # Lorentz factor

        # frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
        frequencies = 10**np.linspace(8, 15, num=200)

        Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
        Z_fs.imped_calc(frequencies, low_frequency_transition=1e-4)

        energy_loss = 2 * np.trapz(Z_fs.impedance.real, frequencies) * elCharge  # [eV]

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
        freqs = 10**np.linspace(8, 15, num=200)

        Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
        Z_fs.imped_calc(freqs, high_frequency_transition=10)

        # Fourier transform of Gaussian bunch profile
        Lambda = np.exp(-0.5 * (2*np.pi*freqs*sigma_dt)**2)

        W_fs = np.zeros_like(times)
        for it, t in enumerate(times):
            W_fs[it] = 2 * np.trapz(
                Z_fs.impedance*Lambda*np.exp(2j*np.pi*freqs*t), freqs).real

        # convert to volt
        W_fs *= elCharge * intensity

        W_fs_test = np.array([-2.99526459e+01, -4.11849124e+01, -6.39549552e+01,
                              -1.15710237e+02, -2.71278552e+01,  +4.34850900e+02,
                              +2.48528445e+02, +1.86894225e+01, -2.58048424e-01,
                              -2.86820828e-01, -2.03095038e-01])

        np.testing.assert_allclose(W_fs, W_fs_test)


class TestCoherentSynchrotronRadiation2(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.coherent_synchrotron_radiation = CoherentSynchrotronRadiation(r_bend=None, gamma=None, chamber_height=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test__fs_integrandReZ(self):
        # TODO: implement test for `_fs_integrandReZ`
        self.coherent_synchrotron_radiation._fs_integrandReZ(x=None)

    @unittest.skip
    def test__fs_spectrum(self):
        # TODO: implement test for `_fs_spectrum`
        self.coherent_synchrotron_radiation._fs_spectrum(frequency_array=None, epsilon=None, low_frequency_transition=None, high_frequency_transition=None)

    @unittest.skip
    def test__pp_low_frequency(self):
        # TODO: implement test for `_pp_low_frequency`
        self.coherent_synchrotron_radiation._pp_low_frequency(frequency_array=None, u_max=None, high_frequency_transition=None)

    @unittest.skip
    def test__pp_spectrum(self):
        # TODO: implement test for `_pp_spectrum`
        self.coherent_synchrotron_radiation._pp_spectrum(frequency_array=None, zeta_max=None)


class TestInputTable(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.input_table = InputTable(input_1=None, input_2=None, input_3=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_wake_calc(self):
        # TODO: implement test for `wake_calc`
        self.input_table.wake_calc(new_time_array=None)

    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.resistive_wall = ResistiveWall(pipe_radius=None, pipe_length=None, resistivity=None, conductivity=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_conductivity(self):
        # TODO: implement test for `conductivity`
        self.resistive_wall.conductivity()

    @unittest.skip
    def test_conductivity(self):
        # TODO: implement test for `conductivity`
        self.resistive_wall.conductivity(conductivity=None)

    @unittest.skip
    def test_imped_calc(self):
        # TODO: implement test for `imped_calc`
        self.resistive_wall.imped_calc(frequency_array=None)

    @unittest.skip
    def test_resistivity(self):
        # TODO: implement test for `resistivity`
        self.resistive_wall.resistivity()

    @unittest.skip
    def test_resistivity(self):
        # TODO: implement test for `resistivity`
        self.resistive_wall.resistivity(resistivity=None)

    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.resonators = Resonators(R_S=None, frequency_R=None, Q=None, method=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test__imped_calc_python(self):
        # TODO: implement test for `_imped_calc_python`
        self.resonators._imped_calc_python(frequency_array=None)

    @unittest.skip
    def test_omega_R(self):
        # TODO: implement test for `omega_R`
        self.resonators.omega_R(omega_R=None)


class TestTravelingWaveCavity(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.traveling_wave_cavity = TravelingWaveCavity(R_S=None, frequency_R=None, a_factor=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_imped_calc(self):
        # TODO: implement test for `imped_calc`
        self.traveling_wave_cavity.imped_calc(frequency_array=None)

if __name__ == '__main__':

    unittest.main()