import unittest

import numpy as np

from blond.experimental.beam_preparation.bucket_filler_functions import (
    multibunch_match_metric_to_hamilton,
)
from blond.experimental.beam_preparation.density_functions import (
    binomial_density,
    gaussian_density,
)
from blond.experimental.beam_preparation.metric_functions import (
    q_percent_emittance,
    rms_emittance,
)


class TestGeneralizedBucketFiller(unittest.TestCase):
    def setUpClass(cls):
        cls.time_grid, cls.deltaE_grid = np.meshgrid(
            np.arange(-10, 11, 21), np.arange(-10, 11, 21)
        )
        cls.hamilton_2D = np.exp(-cls.time_grid * cls.deltaE_grid)

    def test_can_converge(cls):
        desired_metric = 5
        tolerance = 0.1
        density = multibunch_match_metric_to_hamilton(
            time_grid=cls.time_grid,
            deltaE_grid=cls.deltaE_grid,
            hamilton_2D=cls.hamilton_2D,
            metric_list=[desired_metric],
            intensity_frac_list=[1],
            n_buckets=[1],
            max_metric_diff=tolerance,
            density_function=gaussian_density,
            metric_function=rms_emittance,
            max_iterations=100,
            free_parameter_guess=10,
        )

        fitted_metric = rms_emittance(density, cls.time_grid, cls.deltaE_grid)
        cls.assertAlmostEqual(desired_metric, fitted_metric, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
