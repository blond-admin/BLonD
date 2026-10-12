import unittest

import numpy as np

from blond.acc_math.analytic.simple_math import gaussian_distribution
from blond.testing.backend_testing import BLonDTestCase


class TestGaussianDistribution(BLonDTestCase):
    def setUp(self):
        self.sigma_t = 2.5e-9
        self.center = 7e-9
        # +-10 sigma so the truncated tails are negligible
        self.time_array = np.linspace(
            self.center - 10 * self.sigma_t,
            self.center + 10 * self.sigma_t,
            10001,
        )
        self.density = gaussian_distribution(
            self.time_array, sigma_t=self.sigma_t, center=self.center
        )

    def test_is_normalized(self):
        integral = np.trapezoid(self.density, self.time_array)
        self.assertAlmostEqual(integral, 1.0, places=10)

    def test_mean(self):
        mean = np.trapezoid(self.time_array * self.density, self.time_array)
        np.testing.assert_allclose(mean, self.center, rtol=1e-10)

    def test_std(self):
        variance = np.trapezoid(
            (self.time_array - self.center) ** 2 * self.density,
            self.time_array,
        )
        np.testing.assert_allclose(np.sqrt(variance), self.sigma_t, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
