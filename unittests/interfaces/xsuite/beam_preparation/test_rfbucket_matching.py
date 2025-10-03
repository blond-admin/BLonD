import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from blond.interefaces.xsuite.beam_preparation.rfbucket_matching import XsuiteRFBucketMatcher
from xpart.longitudinal.rfbucket_matching import (QGaussianDistribution,
                                                  ThermalDistribution,
                                                  ParabolicDistribution)
from blond import SingleHarmonicCavity, proton

class TestXsuiteRFBucketMatcher(unittest.TestCase):
    def setUp(self):
        self.mock_cavity = SingleHarmonicCavity()
        self.mock_cavity.harmonic = 35640
        self.mock_cavity.voltage = 6e6
        self.mock_cavity.phi_rf = 135 * (np.pi / 180)
        self.mock_cavity.apply_schedules = MagicMock()

        self.matcher = XsuiteRFBucketMatcher(
            n_macroparticles=1000,
            distribution_type=QGaussianDistribution,
            cavity=self.mock_cavity,
            sigma_z=0.1,
            energy_init=450e9,
            verbose_regeneration=False,
        )

    def test___init__(self):
        self.assertEqual(self.matcher.n_macroparticles, 1000)
        self.assertEqual(self.matcher.distribution_type, QGaussianDistribution)
        self.assertEqual(self.matcher.cavity, self.mock_cavity)
        self.assertEqual(self.matcher.sigma_z, 0.1)
        self.assertEqual(self.matcher.energy_init, 450e9)
        self.assertFalse(self.matcher.verbose_regeneration)


    @patch("blond.interefaces.xsuite.beam_preparation.rfbucket_matching.RFBucketMatcher")
    def test_prepare_beam_calls_setup_correctly(self, mock_rfbucket_matcher_class):
        # Arrange
        mock_beam = MagicMock()
        mock_beam.particle_type.mass = proton.mass
        mock_beam.particle_type.charge = proton.charge
        mock_beam.setup_beam = MagicMock()

        mock_drift = MagicMock()
        mock_drift.transition_gamma = 55.759505
        mock_drift.apply_schedules = MagicMock()

        mock_simulation = MagicMock()
        mock_simulation.ring.circumference = 26658.883
        mock_simulation.ring.elements.get_elements.return_value = [mock_drift]

        mock_matcher_instance = MagicMock()
        mock_matcher_instance.generate.return_value = (np.array([1.0]), np.array([0.01]))
        mock_rfbucket_matcher_class.return_value = mock_matcher_instance

        # Act
        self.matcher.prepare_beam(simulation=mock_simulation, beam=mock_beam)

        # Assert
        mock_beam.setup_beam.assert_called_once()
        args, kwargs = mock_beam.setup_beam.call_args
        self.assertIn("dt", kwargs)
        self.assertIn("dE", kwargs)
        self.assertTrue(np.allclose(kwargs["dE"], 0.01 * 450e9))

    def test_prepare_beam_raises_without_energy(self):
        self.matcher.energy_init = None
        with self.assertRaises(ValueError) as cm:
            self.matcher.prepare_beam(simulation=MagicMock(), beam=MagicMock())
        self.assertIn("Initial energy is not set", str(cm.exception))

    def test_prepare_beam_raises_without_cavity(self):
        self.matcher.cavity = None
        with self.assertRaises(ValueError) as cm:
            self.matcher.prepare_beam(simulation=MagicMock(), beam=MagicMock())
        self.assertIn("Cavity is not set", str(cm.exception))

    def test_prepare_beam_raises_without_transition_gamma(self):
        mock_drift = MagicMock()
        mock_drift.transition_gamma = None

        mock_simulation = MagicMock()
        mock_simulation.ring.elements.get_elements.return_value = [mock_drift]

        self.matcher.cavity = self.mock_cavity
        self.matcher.energy_init = 450e9

        with self.assertRaises(ValueError) as cm:
            self.matcher.prepare_beam(simulation=mock_simulation, beam=MagicMock())
        self.assertIn("transition_gamma is not set", str(cm.exception))


if __name__ == "__main__":
    unittest.main()



