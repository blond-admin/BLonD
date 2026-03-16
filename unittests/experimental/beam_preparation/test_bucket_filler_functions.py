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
    def setUp(self):
        grid_size = 100
        self.time_grid, self.deltaE_grid = np.meshgrid(
            np.linspace(-10, 11, num=grid_size),
            np.linspace(-10, 11, num=grid_size),
        )
        self.hamilton_2D = np.exp(-self.time_grid * self.deltaE_grid)

    def test_cannot_converge(self):
        desired_metric = 5
        tolerance = 0.001
        with self.assertWarnsRegex(
            RuntimeWarning, "Specified metric accuracy was not reached"
        ):
            density = multibunch_match_metric_to_hamilton(
                time_grid=self.time_grid,
                deltaE_grid=self.deltaE_grid,
                hamilton_2D=self.hamilton_2D,
                metric_list=[desired_metric],
                intensity_frac_list=[1],
                n_buckets=1,
                max_metric_diff=tolerance,
                density_function=gaussian_density,
                metric_function=rms_emittance,
                max_iterations=1000,
                # free_parameter_guess=10,
            )

    def test_can_converge(self):
        desired_metric = 25
        tolerance = 0.001

        density = multibunch_match_metric_to_hamilton(
            time_grid=self.time_grid,
            deltaE_grid=self.deltaE_grid,
            hamilton_2D=self.hamilton_2D,
            metric_list=[desired_metric],
            intensity_frac_list=[1],
            n_buckets=1,
            max_metric_diff=tolerance,
            density_function=gaussian_density,
            metric_function=rms_emittance,
            max_iterations=1000,
            # free_parameter_guess=10,
        )

        fitted_metric = rms_emittance(
            density, self.time_grid, self.deltaE_grid
        )
        self.assertAlmostEqual(desired_metric, fitted_metric, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
