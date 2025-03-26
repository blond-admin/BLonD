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
            [np.min(beam.dE), np.max(beam.dE)],
        ]
        weighted_beam = beam.get_new_beam_with_weights(bins=bins)
        H1, xbins1, ybins1 = np.histogram2d(
            beam.dt, beam.dE, bins=bins, range=hist_range, density=True
        )
        H2, xbins2, ybins2 = np.histogram2d(
            weighted_beam.dt,
            weighted_beam.dE,
            weights=weighted_beam.weights,
            bins=bins,
            range=hist_range,
            density=True,
        )

        np.testing.assert_allclose(H1, H2)
        np.testing.assert_allclose(xbins1, xbins2)
        np.testing.assert_allclose(ybins1, ybins2)

    def test_statistics(self):
        dt = np.concatenate((np.zeros(10), 5 * np.ones(10), 199 * np.ones(1)))
        dE = np.concatenate((np.zeros(10), 10.0 * np.ones(10), 98 * np.ones(
            1)))
        weights = np.concatenate((np.ones(10), np.zeros(10), np.ones(1)))
        beam = Beam(
            Ring=self.ring,
            n_macroparticles=21,
            intensity=1e9,
            dt=dt,
            dE=dE,
            weights=weights,
        )
        beam.id[:10] = 0
        assert beam.dE_mean(ignore_id_0=False) == 8.909090909090908
        assert beam.dE_mean(ignore_id_0=True) == 98.0
        assert beam.dE_std(ignore_id_0=False) == 28.17301915422738
        assert beam.dE_std(ignore_id_0=True) == 0.0
        assert beam.dE_min(ignore_id_0=False) == 0
        assert beam.dE_min(ignore_id_0=True) == 10
        assert beam.dE_max(ignore_id_0=False) == 98
        assert beam.dE_max(ignore_id_0=True) == 98

        assert beam.dt_mean(ignore_id_0=False) == 18.09090909090909
        assert beam.dt_mean(ignore_id_0=True) == 199.0
        assert beam.dt_std(ignore_id_0=False) == 57.20847767031886
        assert beam.dt_std(ignore_id_0=True) == 0
        assert beam.dt_min(ignore_id_0=False) == 0
        assert beam.dt_min(ignore_id_0=True) == 5
        assert beam.dt_max(ignore_id_0=False) == 199
        assert beam.dt_max(ignore_id_0=True) == 199

    def test_statistics_no_weights(self):
        dt = np.concatenate((np.zeros(10), 5 * np.ones(10), 199 * np.ones(1)))
        dE = np.concatenate((np.zeros(10), 10.0 * np.ones(10), 98 * np.ones(
            1)))
        beam = Beam(
            Ring=self.ring,
            n_macroparticles=21,
            intensity=1e9,
            dt=dt,
            dE=dE,
        )
        beam.id[:10] = 0

        assert beam.dE_mean(ignore_id_0=False) == 9.428571428571429
        assert beam.dE_mean(ignore_id_0=True) == 18.0
        assert beam.dE_std(ignore_id_0=False) == 20.397412134109256
        assert beam.dE_std(ignore_id_0=True) == 25.298221281347036
        assert beam.dE_min(ignore_id_0=False) == 0
        assert beam.dE_min(ignore_id_0=True) == 10
        assert beam.dE_max(ignore_id_0=False) == 98
        assert beam.dE_max(ignore_id_0=True) == 98

        assert beam.dt_mean(ignore_id_0=False) == 11.857142857142858
        assert beam.dt_mean(ignore_id_0=True) == 22.636363636363637
        assert beam.dt_std(ignore_id_0=False) == 41.91747642609193
        assert beam.dt_std(ignore_id_0=True) == 55.77107873387869
        assert beam.dt_min(ignore_id_0=False) == 0
        assert beam.dt_min(ignore_id_0=True) == 5
        assert beam.dt_max(ignore_id_0=False) == 199
        assert beam.dt_max(ignore_id_0=True) == 199




if __name__ == "__main__":
    unittest.main()
