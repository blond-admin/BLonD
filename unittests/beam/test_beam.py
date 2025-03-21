import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation


class TestBeam(unittest.TestCase):
    def setUp(self):
        number_of_turns = 20

        # Define general parameters
        ring = Ring(
            ring_length=26658.883,
            alpha_0=1.0 / 55.759505**2,
            synchronous_data=np.linspace(
                450e9, 460.005e9, number_of_turns + 1
            ),
            Particle=Proton(),
            n_turns=number_of_turns,
        )

        # Define beam and distribution
        self.beam = Beam(Ring=ring, n_macroparticles=1001, intensity=1e9)
        self.ring = ring

    def test_n_macroparticles_not_alive(self):
        self.beam.id[1] = 0
        self.assertEqual(self.beam.n_macroparticles_not_alive, 1)

    def test_n_macroparticles_eliminated(self):
        self.beam.id[1] = 0
        self.beam.eliminate_lost_particles()
        self.assertEqual(self.beam.n_macroparticles_eliminated, 1)
        self.assertEqual(self.beam.n_macroparticles_not_alive, 0)

    def test_get_new_beam_with_weights(self):
        bins = 16
        beam = Beam(Ring=self.ring, n_macroparticles=10_000, intensity=1e9)

        beam.dt = np.random.randn(beam.n_macroparticles) - 5
        beam.dE = np.random.randn(beam.n_macroparticles) + 5
        hist_range = [
            [np.min(beam.dt), np.max(beam.dt)],
            [np.min(beam.dE), np.max(beam.dE)]
        ]
        weighted_beam = beam.get_new_beam_with_weights(bins=bins)
        H1, xbins1, ybins1 = np.histogram2d(
            beam.dt,
            beam.dE,
            bins=bins,
            range=hist_range,
            density=True
        )
        H2, xbins2, ybins2 = np.histogram2d(
            weighted_beam.dt,
            weighted_beam.dE,
            weights=weighted_beam.weights,
            bins=bins,
            range=hist_range,
            density=True
        )
        np.testing.assert_allclose(H1, H2)
        np.testing.assert_allclose(xbins1, xbins2)
        np.testing.assert_allclose(ybins1, ybins2)


if __name__ == "__main__":
    unittest.main()
