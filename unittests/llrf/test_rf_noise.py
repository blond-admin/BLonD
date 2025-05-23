import unittest
import numpy as np
from unittest.mock import MagicMock

from blond.beam.beam import Proton
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
# Dummy backend math module if the original uses a backend like cupy

from blond.llrf.rf_noise import LHCNoiseFB

# Dummy cfwhm if it’s used as a global constant
cfwhm = 1.0  # Adjust if there's a specific constant you're using


class DummyProfile:
    def __init__(self, bin_centers, n_macroparticles):
        self.bin_centers = bin_centers
        self.n_macroparticles = n_macroparticles


# -- Import class under test --
# from your_module import LHCNoiseFB  # adjust import path accordingly


class TestLHCNoiseFB(unittest.TestCase):
    def setUp(self):
        # Machine and RF parameters
        C = 26658.883  # Machine circumference [m]
        h = 35640  # Harmonic number
        dphi = 0.0  # Phase modulation/offset
        gamma_t = 53.8  # Transition gamma
        alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor
        n_turns = 20000
        p = 450e9 * np.ones(n_turns + 1)
        v = 5e6 * np.ones(n_turns + 1)

        self.ring = Ring(C, alpha, p, Particle=Proton(), n_turns=n_turns)
        self.rf = RFStation(self.ring, [h], v, [dphi])

        bin_centers = np.linspace(-0.5, 0.5, 100)
        n_macroparticles = np.ones(100)
        self.profile = DummyProfile(bin_centers, n_macroparticles)

        self.f_rev = 11245
        self.bl_target = 1.35e-9

    def test_initialization(self):
        fb = LHCNoiseFB(self.rf, self.profile, self.f_rev, self.bl_target)
        self.assertAlmostEqual(fb.x, 0.0)
        self.assertEqual(fb.bl_meas, self.bl_target)
        self.assertTrue(callable(fb.track))

    def test_fwhm_single_bunch(self):
        # create a peak at the center to test FWHM calculation
        self.profile.n_macroparticles[45:55] = 10
        fb = LHCNoiseFB(self.rf, self.profile, self.f_rev, self.bl_target)
        fb.fwhm_single_bunch()
        self.assertGreater(fb.bl_meas, 0)

    def test_track_no_delay(self):
        # Simulate turn update
        fb = LHCNoiseFB(
            self.rf, self.profile, self.f_rev, self.bl_target, no_delay=True
        )
        self.rf.counter[0] = fb.n_update  # Trigger condition
        fb.fwhm = MagicMock()  # Mock the fwhm method
        fb.track()
        self.assertTrue(0 <= fb.x <= 1)

    def test_update_bqm_measurement_buffering(self):
        fb = LHCNoiseFB(self.rf, self.profile, self.f_rev, self.bl_target)
        fb.fwhm = MagicMock()
        fb.timers[1].counter = fb.delay  # Simulate initial delay pass
        fb.bl_meas = 1.23e-9
        fb.update_bqm_measurement()
        np.testing.assert_array_equal(
            fb.last_bqm_measurements, np.full(5, 1.23e-9)
        )

    def test_update_noise_amplitude_before_delay(self):
        fb = LHCNoiseFB(self.rf, self.profile, self.f_rev, self.bl_target)
        fb.update_x = False
        fb.update_noise_amplitude()
        self.assertEqual(fb.x, 0)

    def test_update_noise_amplitude_after_delay(self):
        fb = LHCNoiseFB(self.rf, self.profile, self.f_rev, self.bl_target)
        fb.update_x = True
        fb.timers[0].counter = 3 * int(self.f_rev)  # timestamp
        fb.time_array = np.array([0, 1000, 2000, 3000, 4000])
        fb.last_bqm_measurements = np.array(
            [1.0e-9, 1.05e-9, 1.1e-9, 1.2e-9, 1.3e-9]
        )
        fb.g = np.ones(fb.rf_params.n_turns + 1)
        fb.rf_params.counter[0] = 0
        fb.x = 0.2
        fb.update_noise_amplitude()
        self.assertTrue(0 <= fb.x <= 1)


if __name__ == "__main__":
    unittest.main()
