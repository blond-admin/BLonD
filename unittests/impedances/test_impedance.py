# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for impedances.impedance

:Authors: **Markus Schwarz**
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized
from scipy.constants import c, e, m_p

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageResonator
from blond.impedances.impedance import InducedVoltageTime
from blond.impedances.impedance import (
    InductiveImpedance,
)
from blond.impedances.impedance_sources import Resonators
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


class TestInducedVoltageFreq(unittest.TestCase):
    def setUp(self):
        # Nyquist frequency 1.6 GHz; frequency spacing 50 MHz
        self.profile = Profile(
            None, cut_options=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16)
        )
        self.impedance_source = Resonators([4.5e6], [200.222e6], [200])

    def test_default_wake_length(self):
        test_object = InducedVoltageFreq(None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.wake_length, self.profile.bin_size * self.profile.n_slices
        )

    def test_default_frequency_resolution(self):
        test_object = InducedVoltageFreq(None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.frequency_resolution,
            1 / (self.profile.bin_size * self.profile.n_slices),
        )

    def test_default_n_rfft(self):
        test_object = InducedVoltageFreq(None, self.profile, [self.impedance_source])

        np.testing.assert_equal(test_object.n_fft, 16)

    def test_frequency_resolution_input(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source], frequency_resolution=3e6
        )

        np.testing.assert_allclose(test_object.frequency_resolution_input, 3e6)

    def test_frequency_resolution_n_induced_voltage(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source], frequency_resolution=3e6
        )

        np.testing.assert_equal(test_object.n_induced_voltage, 1067)

    def test_frequency_resolution_n_fft(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source], frequency_resolution=3e6
        )

        np.testing.assert_equal(test_object.n_fft, 1080)

    def test_frequency_resolution_frequency_resolution(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source], frequency_resolution=3e6
        )

        np.testing.assert_allclose(
            test_object.frequency_resolution, test_object.freq[1] - test_object.freq[0]
        )



class TestInducedVoltageTime(unittest.TestCase):
    def setUp(self):
        # Nyquist frequency 1.6 GHz; frequency spacing 50 MHz
        self.profile = Profile(
            None, cut_options=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16)
        )
        self.impedance_source = Resonators([4.5e6], [200.222e6], [200])

    def test_default_wake_length(self):
        test_object = InducedVoltageTime(None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.wake_length, self.profile.bin_size * self.profile.n_slices
        )

    def test_default_frequency_resolution(self):
        test_object = InducedVoltageTime(None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.frequency_resolution,
            1 / (self.profile.bin_size * 2 * test_object.n_induced_voltage),
        )

    def test_default_n_rfft(self):
        test_object = InducedVoltageTime(None, self.profile, [self.impedance_source])

        np.testing.assert_equal(test_object.n_fft, 2 * 16)

    def test_wake_length_input(self):
        test_object = InducedVoltageTime(
            None, self.profile, [self.impedance_source], wake_length=11e-9
        )

        np.testing.assert_allclose(test_object.wake_length_input, 11e-9)



class TestInductiveImpedance(unittest.TestCase):
    def setUp(self):
        # SIMULATION PARAMETERS -------------------------------------------------------

        # Beam parameters
        sigma_dt = 180e-9 / 4  # [s]
        kin_beam_energy = 1.4e9  # [eV]

        E_0 = m_p * c**2 / e  # [eV]
        tot_beam_energy = E_0 + kin_beam_energy  # [eV]

        self.ring = Ring(
            2 * np.pi * 25,
            1 / 4.4**2,
            np.sqrt(tot_beam_energy**2 - E_0**2),
            Proton(),
            2,
        )

        self.rf_station = RFStation(self.ring, [1], [8e3], [np.pi], 1)

        dt = np.random.randn(1001) * sigma_dt
        dE = np.random.randn(len(dt)) / sigma_dt
        self.beam = Beam(self.ring, len(dt), 1e11, dt=dt, dE=dE)


        # DEFINE SLICES----------------------------------------------------------------
        self.profile1 = Profile(
            self.beam,
            CutOptions(cut_left=np.min(dt), cut_right=np.max(dt), n_slices=64),
        )
        self.profile1.track()

    @parameterized.expand(
        [
            ("gradient"),
            ("diff"),
            # ('filter1d')
        ]
    )
    def test_1(self, deriv_mode):
        for z_over_n in (
            [0.00606029, 0.00606029, 0.00606029],
            [-66.22486545, -66.22486545, -66.22486545],
        ):
            inductive_impedance = InductiveImpedance(
                beam=self.beam,
                profile=self.profile1,
                Z_over_n=z_over_n,
                rf_station=self.rf_station,
                deriv_mode=deriv_mode,  # "filter1d", "gradient", "diff"
            )
            wake_kernel = inductive_impedance.get_wake_kernel(
                t_start=self.profile1.cut_left,
                t_stop=self.profile1.cut_right,
                n=self.profile1.n_slices,
            )
            inductive_impedance.induced_voltage_1turn()
            result_expected = inductive_impedance.induced_voltage[1:-1]
            result_under_test = ((-self.beam.particle.charge * e *
                                 self.beam.ratio)*
                                 np.convolve(
                self.profile1.n_macroparticles,
                wake_kernel,
                "full",
            )[32 : 32 + 62])
            DEV_DEBUG = False

            if DEV_DEBUG:
                plt.plot(result_under_test, label="result_under_test")
                plt.plot(result_expected, label="result_expected")
                plt.legend()
                plt.show()

            np.testing.assert_allclose(
                result_under_test,
                result_expected,
                atol=1e-12,
            )


class TestInducedVoltageResonator(unittest.TestCase):
    def setUp(self):
        sigma_dt = 180e-9 / 4  # [s]
        kin_beam_energy = 1.4e9  # [eV]

        E_0 = m_p * c**2 / e  # [eV]
        tot_beam_energy = E_0 + kin_beam_energy  # [eV]

        self.ring = Ring(
            2 * np.pi * 25,
            1 / 4.4**2,
            np.sqrt(tot_beam_energy**2 - E_0**2),
            Proton(),
            2,
        )

        self.rf_station = RFStation(self.ring, [1], [8e3], [np.pi], 1)

        dt = np.random.randn(1001) * sigma_dt
        dE = np.random.randn(len(dt)) / sigma_dt
        self.beam = Beam(self.ring, len(dt), 1e11, dt=dt, dE=dE)
        self.profile1 = Profile(
            self.beam,
            CutOptions(cut_left=np.min(dt), cut_right=np.max(dt), n_slices=64),
        )
        self.resonators = Resonators([4.5e6], [200.222e6], [200])
        self.induced_voltage_resonator = InducedVoltageResonator(
            beam=self.beam,
            profile=self.profile1,
            resonators=self.resonators,
        )



if __name__ == "__main__":
    unittest.main()
