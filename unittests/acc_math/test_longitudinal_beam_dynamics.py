import unittest

import numpy as np
from scipy.constants import e

from blond import backend, electron
from blond.acc_math.analytic.longitudinal_beam_dynamics import (
    get_angular_synchrotron_frequency,
    get_angular_synchrotron_tune,
    get_small_amplitude_angular_synchrotron_frequency,
    get_small_amplitude_angular_synchrotron_tune,
)
from blond.generals.exceptions import UnevenArraySizes


class TestLongitudinalBeamDynamics_float_inputs(unittest.TestCase):
    def setUp(self):
        # FCC-ee high-energy booster at injection energy
        self.particle_type = electron
        self.beam_energy = 20e9
        self.voltage = 50e6
        self.revolution_frequency = 3300
        self.harmonic_number = 242400
        self.momentum_compaction_factor = 7.120435962 * 1e-6
        self.phase_slip_factor = 7.120435962 * 1e-6
        self.synchronous_phase = 3.10

        self.expected_small_amplitude_tune = np.sqrt(
            (
                self.harmonic_number
                * e
                * self.voltage
                * abs(self.phase_slip_factor * np.cos(self.synchronous_phase))
            )
            / (2 * np.pi * self.beam_energy)
        )
        self.expected_small_amplitude_frequency = (
            2
            * np.pi
            * self.revolution_frequency
            * self.expected_small_amplitude_tune
        )

        self.places = 6 if backend.float == np.float32 else 12

    def test_get_linear_angular_synchrotron_frequency(self):
        tune = get_small_amplitude_angular_synchrotron_tune(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
        )
        self.assertAlmostEqual(
            self.expected_small_amplitude_tune,
            tune,
            msg="Expected value = 1.1e-7 rad/s",
            places=self.places,
        )

    def test_get_small_amplitude_angular_synchrotron_frequency(self):
        frequency = get_small_amplitude_angular_synchrotron_frequency(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )
        self.assertAlmostEqual(
            self.expected_small_amplitude_frequency,
            frequency,
            msg="Expected value = 1.1e-7 rad/s",
            places=self.places,
        )

    def test_get_angular_synchrotron_tune_and_frequency(self):
        small_amplitude_tune = self.expected_small_amplitude_tune
        expected = (1 / (2 * np.pi)) * np.arccos(
            1 - 2 * (np.pi * small_amplitude_tune) ** 2
        )

        tune = get_angular_synchrotron_tune(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
        )

        self.assertAlmostEqual(
            expected,
            tune,
            places=self.places,
        )

        tune = get_angular_synchrotron_frequency(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )

        self.assertAlmostEqual(
            2 * np.pi * self.revolution_frequency * expected,
            tune,
            places=self.places,
        )


class TestLongitudinalBeamDynamics_array_inputs(unittest.TestCase):
    def setUp(self):
        # Example of the FCC-ee high-energy booster at injection
        self.energy = np.array([20.0, 20.0, 1.0, 10.0, 1.0])
        self.revolution_frequency = np.array([5.0, 2.0, 1.0, 10.0, 1.0])
        self.voltage = np.array([50e6, 40e6, 1e6, 10e6, 5e6])
        self.harmonic_number = 250
        self.phase_slip_factor = 7e-6
        self.synchronous_phase = 0.15

        self.expected_small_amplitude_tune = np.sqrt(
            (
                self.harmonic_number
                * e
                * self.voltage
                * abs(self.phase_slip_factor * np.cos(self.synchronous_phase))
            )
            / (2 * np.pi * self.energy)
        )
        self.expected_small_amplitude_frequency = (
            2
            * np.pi
            * self.revolution_frequency
            * self.expected_small_amplitude_tune
        )

        self.places = 6 if backend.float == np.float32 else 12

    def test_get_linear_angular_synchrotron_frequency(self):
        with self.assertRaises(UnevenArraySizes):
            get_small_amplitude_angular_synchrotron_frequency(
                energy=self.energy,
                voltage=self.voltage[0:3],
                harmonic_number=self.harmonic_number,
                synchronous_phase=self.synchronous_phase,
                phase_slip_factor=self.phase_slip_factor,
                revolution_frequency=self.revolution_frequency,
            )
        tune = get_small_amplitude_angular_synchrotron_tune(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
        )
        np.testing.assert_almost_equal(
            self.expected_small_amplitude_tune,
            tune,
            decimal=self.places,
        )

    def test_get_small_amplitude_angular_synchrotron_frequency(self):
        frequency = get_small_amplitude_angular_synchrotron_frequency(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )
        np.testing.assert_almost_equal(
            self.expected_small_amplitude_frequency,
            frequency,
            decimal=self.places,
        )

    def test_get_angular_synchrotron_tune_and_frequency(self):
        small_amplitude_tune = self.expected_small_amplitude_tune
        expected = (1 / (2 * np.pi)) * np.arccos(
            1 - 2 * (np.pi * small_amplitude_tune) ** 2
        )

        tune = get_angular_synchrotron_tune(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
        )

        np.testing.assert_almost_equal(
            expected,
            tune,
            decimal=self.places,
        )

        tune = get_angular_synchrotron_frequency(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )

        np.testing.assert_almost_equal(
            2 * np.pi * self.revolution_frequency * expected,
            tune,
            decimal=self.places,
        )
