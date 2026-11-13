import unittest
import warnings

import numpy as np

from blond.experimental.beam_preparation.bucket_filler_functions import (
    hamilton_to_density_by_bunch_length,
    hamilton_to_density_by_emittance,
    hamilton_to_density_by_metric,
)
from blond.testing.backend_testing import BLonDTestCase


def _bucket_grid(t_start=-0.5, t_stop=0.5, n_time=201, n_energy=201):
    """Hamiltonian of a cosine well, minimum at t = 0, barriers at +-0.5.

    ``H = dE**2 / 2 + 1 - cos(2 pi t)``, so the separatrix is at H = 2 and
    reaches dE = +-2 at the bucket centre.
    """
    time_grid, deltaE_grid = np.meshgrid(
        np.linspace(t_start, t_stop, n_time),
        np.linspace(-2.0, 2.0, n_energy),
        indexing="ij",
    )
    hamilton_2D = 0.5 * deltaE_grid**2 + 1 - np.cos(2 * np.pi * time_grid)
    return time_grid, deltaE_grid, hamilton_2D


def _bunch_length_4sigma(time_grid, density):
    line_density = density.sum(axis=1)
    ts = time_grid[:, 0]
    mean = np.sum(ts * line_density) / line_density.sum()
    variance = np.sum((ts - mean) ** 2 * line_density) / line_density.sum()
    return 4 * np.sqrt(variance)


def _emittance_4pi_rms(time_grid, deltaE_grid, density):
    weights = density / density.sum()
    mean_t = np.sum(weights * time_grid)
    mean_e = np.sum(weights * deltaE_grid)
    var_t = np.sum(weights * (time_grid - mean_t) ** 2)
    var_e = np.sum(weights * (deltaE_grid - mean_e) ** 2)
    cov = np.sum(weights * (time_grid - mean_t) * (deltaE_grid - mean_e))
    return 4 * np.pi * np.sqrt(var_t * var_e - cov**2)


class TestHamiltonToDensityByBunchLength(BLonDTestCase):
    def test_reaches_target(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        for exponent in (0.5, 1.0, 3.0):
            with self.subTest(exponent=exponent):
                density = hamilton_to_density_by_bunch_length(
                    time_grid=time_grid,
                    deltaE_grid=deltaE_grid,
                    hamilton_2D=hamilton_2D,
                    bunch_length_4sigma=0.3,
                    exponent=exponent,
                )
                self.assertAlmostEqual(
                    _bunch_length_4sigma(time_grid, density), 0.3, places=4
                )

    def test_density_is_binomial_in_hamiltonian(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        density = hamilton_to_density_by_bunch_length(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            bunch_length_4sigma=0.3,
            exponent=2.0,
        )
        inside = density > 0
        # rho = (1 - H / H_cut) ** 2: sqrt(rho) is linear in H, zero at
        # H_cut, and H_cut lies below the separatrix
        hamiltons = hamilton_2D[inside]
        slope, intercept = np.polyfit(hamiltons, np.sqrt(density[inside]), 1)
        np.testing.assert_allclose(
            np.sqrt(density[inside]), intercept + slope * hamiltons, atol=1e-9
        )
        self.assertAlmostEqual(intercept, 1.0)
        self.assertLess(-intercept / slope, 2.0)

    def test_independent_of_offset_of_hamiltonian(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        densities = [
            hamilton_to_density_by_bunch_length(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D + offset,
                bunch_length_4sigma=0.3,
            )
            for offset in (0.0, 123.0)
        ]
        np.testing.assert_allclose(densities[0], densities[1], atol=1e-9)

    def test_infeasible_target_fills_bucket_and_warns(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        with self.assertWarnsRegex(UserWarning, "does not fit the bucket"):
            density = hamilton_to_density_by_bunch_length(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                bunch_length_4sigma=10.0,
                exponent=0.0,
            )
        # waterbag up to the separatrix: the full bucket
        np.testing.assert_array_equal(density > 0, hamilton_2D < 2)

    def test_too_short_target_warns(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        with self.assertWarnsRegex(UserWarning, "grid"):
            density = hamilton_to_density_by_bunch_length(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                bunch_length_4sigma=1e-6,
            )
        self.assertGreater(density.sum(), 0)

    def test_stays_in_bucket_of_minimum(self):
        # 1.5 buckets: the deepest well at t = 0 and half a neighbour
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid(
            t_start=-0.5, t_stop=1.0, n_time=301
        )
        hamilton_2D = hamilton_2D + 0.2 * (time_grid > 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            density = hamilton_to_density_by_bunch_length(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                bunch_length_4sigma=10.0,
                exponent=0.0,
            )
        self.assertEqual(density[time_grid > 0.5].sum(), 0)
        self.assertGreater(density[np.abs(time_grid) < 0.4].sum(), 0)


class TestHamiltonToDensityByEmittance(BLonDTestCase):
    def test_reaches_target(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        density = hamilton_to_density_by_emittance(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            emittance_4pi_rms=0.5,
            exponent=1.5,
        )
        self.assertAlmostEqual(
            _emittance_4pi_rms(time_grid, deltaE_grid, density), 0.5, places=4
        )

    def test_infeasible_target_warns_with_largest_possible(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        with self.assertWarnsRegex(UserWarning, "largest possible"):
            density = hamilton_to_density_by_emittance(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                emittance_4pi_rms=100.0,
            )
        self.assertEqual(density[hamilton_2D >= 2].sum(), 0)


class TestGaussianDistribution(BLonDTestCase):
    def test_reaches_target_with_density_exponential_in_hamiltonian(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        density = hamilton_to_density_by_bunch_length(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            bunch_length_4sigma=0.3,
            distribution="gaussian",
        )
        self.assertAlmostEqual(
            _bunch_length_4sigma(time_grid, density), 0.3, places=4
        )
        # rho = exp(-H / H_0) inside the separatrix: log(rho) linear in H
        inside = density > 0
        np.testing.assert_array_equal(inside, hamilton_2D < 2)
        hamiltons = hamilton_2D[inside]
        slope, intercept = np.polyfit(hamiltons, np.log(density[inside]), 1)
        np.testing.assert_allclose(
            np.log(density[inside]), intercept + slope * hamiltons, atol=1e-9
        )

    def test_infeasible_target_warns(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        with self.assertWarnsRegex(UserWarning, "largest possible"):
            hamilton_to_density_by_emittance(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                emittance_4pi_rms=100.0,
                distribution="gaussian",
            )

    def test_unknown_distribution_raises(self):
        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        with self.assertRaises(ValueError):
            hamilton_to_density_by_bunch_length(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                bunch_length_4sigma=0.3,
                distribution="lorentzian",
            )


class TestHamiltonToDensityByMetric(BLonDTestCase):
    def test_user_metric_reaches_target(self):
        def rms_energy_spread(weights, ts, deltaEs):
            mean = np.average(deltaEs, weights=weights)
            return np.sqrt(np.average((deltaEs - mean) ** 2, weights=weights))

        time_grid, deltaE_grid, hamilton_2D = _bucket_grid()
        density = hamilton_to_density_by_metric(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            metric=rms_energy_spread,
            target=0.4,
        )
        weights = density / density.sum()
        spread = np.sqrt(np.sum(weights * deltaE_grid**2))
        self.assertAlmostEqual(spread, 0.4, places=4)


if __name__ == "__main__":
    unittest.main()
