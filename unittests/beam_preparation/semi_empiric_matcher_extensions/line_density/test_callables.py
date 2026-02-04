import unittest
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes import (
    ProfileMatcherAddon,
)
from blond.handle_results.helpers import callers_relative_path


class TestProfileMatcherAddon(unittest.TestCase):
    def test__solve_for_density(self):
        mock = Mock(ProfileMatcherAddon)
        mock.maxiter = 10
        mock.smoothness = 0
        mock.atol = 1e-3
        mock._animation_pause = 1
        mock.animate_fitting = False
        mock._animation_fignumber = None
        mock._draw_animation = lambda **kwargs: (
            ProfileMatcherAddon._draw_animation(mock, **kwargs)
        )
        mock._solve_for_density_single_bucket = lambda **kwargs: (
            ProfileMatcherAddon._solve_for_density_single_bucket(
                mock, **kwargs
            )
        )
        hamilton_2D = np.load(
            callers_relative_path("resources/hamilton_2D.npy", stacklevel=1)
        )
        histogram_desired = (
            np.load(
                callers_relative_path(
                    "resources/histogram_desired.npy", stacklevel=1
                )
            )
            ** 5
        )

        density = ProfileMatcherAddon._solve_for_density(
            mock,
            hamilton_2D=hamilton_2D,
            histogram_desired=histogram_desired,
        )
        DEV_PLOT_0 = True
        if DEV_PLOT_0:
            plt.figure()
            ax0 = plt.subplot(2, 2, 1)
            ax1 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)

            ax0.matshow(hamilton_2D.T)
            ax1.matshow(density.T)
            ax3.plot(hamilton_2D.sum(axis=1))
            ax4.plot(density.sum(axis=1))
            plt.show()


if __name__ == "__main__":
    unittest.main()
