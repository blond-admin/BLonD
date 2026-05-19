import unittest

import numpy as np

from blond.experimental.beam_preparation.metric_functions import (
    q_percent_emittance,
    rms_emittance,
)


class TestMetricFunctions(unittest.TestCase):
    def test_q_percent_emittance(self):
        density = np.array(
            [[1, 2], [3, 4]]
        )  # Simple arrays for which q=1/2 emittance is easy to find
        dt_grid = np.array([[1, 1], [2, 2]])
        dE_grid = np.array([[1, 2], [1, 2]])

        half_emittance = q_percent_emittance(
            density, dt_grid, dE_grid, q=1 / 2
        )
        self.assertEqual(half_emittance, 1)

    def test_rms_emittance(self):
        dt_grid, dE_grid = np.meshgrid(
            np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000)
        )
        rms_dt = 1
        rms_dE = 2
        mu_dt = 2
        mu_dE = -1

        # making a zero covariance, multivariate gaussian distribution of known RMS
        density = np.exp(-((dt_grid - mu_dt) ** 2) / (2 * rms_dt**2)) * np.exp(
            -((dE_grid - mu_dE) ** 2) / (2 * rms_dE**2)
        )
        density /= np.sum(density)

        rms_emittance_calc = rms_emittance(density, dt_grid, dE_grid)
        rms_emittance_calc_with_pi = rms_emittance(
            density, dt_grid, dE_grid, multiply_by_pi=True
        )
        rms_emittance_known = rms_dt * rms_dE

        # checking equivelance with 1% tolerance
        self.assertAlmostEqual(
            rms_emittance_known,
            rms_emittance_calc,
            delta=rms_emittance_calc * 0.01,
        )
        self.assertAlmostEqual(
            rms_emittance_known * np.pi,
            rms_emittance_calc_with_pi,
            delta=rms_emittance_calc * 0.01 * np.pi,
        )
