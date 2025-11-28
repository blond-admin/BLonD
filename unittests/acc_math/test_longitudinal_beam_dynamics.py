import unittest

import numpy as np
from numpy.ma.testutils import assert_equal

from blond import backend, electron
from blond.acc_math.analytic.longitudinal_beam_dynamics import (
    get_small_amplitude_angular_synchrotron_frequency,
    get_small_amplitude_angular_synchrotron_tune,
)


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

    def test_get_linear_angular_synchrotron_frequency_single_value(self):
        angular_frequency = get_small_amplitude_angular_synchrotron_frequency(
            energy=self.beam_energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )
        self.assertAlmostEqual(
            2 * 1.0870031405066292e-07,
            angular_frequency,
            msg="Expected value = 1.1e-7 rad/s",
            places=6 if backend.float == np.float32 else 12,
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

    def test_get_linear_angular_synchrotron_frequency(self):
        with self.assertRaises(ValueError):
            get_small_amplitude_angular_synchrotron_frequency(
                energy=self.energy,
                voltage=self.voltage[0:3],
                harmonic_number=self.harmonic_number,
                synchronous_phase=self.synchronous_phase,
                phase_slip_factor=self.phase_slip_factor,
                revolution_frequency=self.revolution_frequency,
            )

        angular_frequency = get_small_amplitude_angular_synchrotron_frequency(
            energy=self.energy,
            voltage=self.voltage,
            harmonic_number=self.harmonic_number,
            synchronous_phase=self.synchronous_phase,
            phase_slip_factor=self.phase_slip_factor,
            revolution_frequency=self.revolution_frequency,
        )

        assert_equal(
            angular_frequency,
            2 * np.array(
                [
                    1.6497648478537963e-07,
                    5.902378154705098e-08,
                    2.0868058091596585e-08,
                    2.086805809159658e-07,
                    4.666239645122449e-08,
                ]
            ),
        )
