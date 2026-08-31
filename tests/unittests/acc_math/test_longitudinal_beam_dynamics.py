import unittest

import numpy as np

from blond import backend, electron
from blond.acc_math.analytic.hamilton import (
    calc_synchrotron_tune_single_harmonic,
)
from blond.acc_math.analytic.longitudinal_beam_dynamics import (
    get_angular_synchrotron_frequency,
    get_angular_synchrotron_tune,
    get_small_amplitude_angular_synchrotron_frequency,
    get_small_amplitude_angular_synchrotron_tune,
)
from blond.generals.exceptions_ import UnevenArraySizes


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
        # 20 GeV electrons are ultra-relativistic; beta ~ 1.
        self.beta = 1.0
        self.charge = self.particle_type.charge

        # Independent ground truth: the single-harmonic synchrotron tune
        # from `hamilton`, which is the reference implementation.
        self.expected_small_amplitude_tune = (
            calc_synchrotron_tune_single_harmonic(
                charge=self.charge,
                voltage=self.voltage,
                beta=self.beta,
                energy=self.beam_energy,
                phi_s=self.synchronous_phase,
                harmonic=self.harmonic_number,
                eta_0=self.phase_slip_factor,
            )
        )
        self.expected_small_amplitude_frequency = (
            2
            * np.pi
            * self.revolution_frequency
            * self.expected_small_amplitude_tune
        )

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        self.places = 12

    def test_get_linear_angular_synchrotron_frequency(self):
        tune = get_small_amplitude_angular_synchrotron_tune(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            beta=self.beta,
            charge=self.charge,
        )
        self.assertAlmostEqual(
            self.expected_small_amplitude_tune,
            tune,
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
            beta=self.beta,
            charge=self.charge,
        )
        self.assertAlmostEqual(
            self.expected_small_amplitude_frequency,
            frequency,
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
            beta=self.beta,
            charge=self.charge,
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
            beta=self.beta,
            charge=self.charge,
        )

        self.assertAlmostEqual(
            2 * np.pi * self.revolution_frequency * expected,
            tune,
            places=self.places,
        )


class TestLongitudinalBeamDynamics_array_inputs(unittest.TestCase):
    def setUp(self):
        # Physically consistent electron beam (total energy > rest mass) so
        # that the reference beta can be derived from the energy.
        self.particle_type = electron
        rest_mass = electron.mass
        self.energy = np.array([20e9, 20e9, 5e9, 10e9, 1e9])
        self.beta = np.sqrt(1.0 - (rest_mass / self.energy) ** 2)
        self.revolution_frequency = np.array([5.0, 2.0, 1.0, 10.0, 1.0])
        self.voltage = np.array([50e6, 40e6, 1e6, 10e6, 5e6])
        self.harmonic_number = 250
        self.phase_slip_factor = 7e-6
        self.synchronous_phase = 0.15
        self.charge = electron.charge

        self.expected_small_amplitude_tune = (
            calc_synchrotron_tune_single_harmonic(
                charge=self.charge,
                voltage=self.voltage,
                beta=self.beta,
                energy=self.energy,
                phi_s=self.synchronous_phase,
                harmonic=self.harmonic_number,
                eta_0=self.phase_slip_factor,
            )
        )
        self.expected_small_amplitude_frequency = (
            2
            * np.pi
            * self.revolution_frequency
            * self.expected_small_amplitude_tune
        )

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        self.places = 12

    def test_get_linear_angular_synchrotron_frequency(self):
        with self.assertRaises(UnevenArraySizes):
            get_small_amplitude_angular_synchrotron_frequency(
                energy=self.energy,
                voltage=self.voltage[0:3],
                harmonic_number=self.harmonic_number,
                synchronous_phase=self.synchronous_phase,
                phase_slip_factor=self.phase_slip_factor,
                revolution_frequency=self.revolution_frequency,
                beta=self.beta,
                charge=self.charge,
            )
        tune = get_small_amplitude_angular_synchrotron_tune(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            beta=self.beta,
            charge=self.charge,
        )
        np.testing.assert_allclose(
            self.expected_small_amplitude_tune,
            tune,
            rtol=1e-12,
        )

    def test_get_small_amplitude_angular_synchrotron_frequency(self):
        frequency = get_small_amplitude_angular_synchrotron_frequency(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
            beta=self.beta,
            charge=self.charge,
        )
        np.testing.assert_allclose(
            self.expected_small_amplitude_frequency,
            frequency,
            rtol=1e-12,
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
            beta=self.beta,
            charge=self.charge,
        )

        np.testing.assert_allclose(
            expected,
            tune,
            rtol=1e-12,
        )

        tune = get_angular_synchrotron_frequency(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
            beta=self.beta,
            charge=self.charge,
        )

        np.testing.assert_allclose(
            2 * np.pi * self.revolution_frequency * expected,
            tune,
            rtol=1e-12,
        )
