import unittest
from unittest.mock import Mock
from unittest.mock import call as mock_call
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes import (
    ProfileMatcherAddon,
)
from blond.handle_results.helpers import callers_relative_path
from blond.testing.backend_testing import BLonDTestCase


def _addon_mock() -> Mock:
    """A `ProfileMatcherAddon` stand-in that runs the real solver code."""
    addon = Mock(ProfileMatcherAddon)
    addon.maxiter = 10
    addon.smoothness = 0
    addon.atol = 1e-3
    addon.animate_fitting_pause_time = 1
    addon.animate_fitting = False
    addon._animation_fignumber = None
    addon._draw_animation = lambda **kwargs: (
        ProfileMatcherAddon._draw_animation(addon, **kwargs)
    )
    addon._solve_for_density_single_bucket = lambda **kwargs: (
        ProfileMatcherAddon._solve_for_density_single_bucket(addon, **kwargs)
    )
    return addon


def _load_inputs() -> tuple[np.ndarray, np.ndarray]:
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
    return hamilton_2D, histogram_desired


class TestProfileMatcherAddon(BLonDTestCase):
    def tearDown(self):
        plt.close("all")

    def test__solve_for_density(self):
        mock = _addon_mock()
        hamilton_2D, histogram_desired = _load_inputs()

        density = ProfileMatcherAddon._solve_for_density(
            mock,
            hamilton_2D=hamilton_2D,
            histogram_desired=histogram_desired,
        )
        DEV_PLOT_0 = False
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

    def test_animate_fitting_pauses_for_animate_fitting_pause_time(self):
        addon = _addon_mock()
        addon.maxiter = 2
        addon.atol = 0  # never converge early, so the animation is drawn
        addon.animate_fitting = True
        addon.animate_fitting_pause_time = 0.0
        hamilton_2D, histogram_desired = _load_inputs()

        with patch("matplotlib.pyplot.pause") as pause:
            ProfileMatcherAddon._solve_for_density(
                addon,
                hamilton_2D=hamilton_2D,
                histogram_desired=histogram_desired,
            )

        self.assertGreater(pause.call_count, 0)
        for call in pause.call_args_list:
            self.assertEqual(call, mock_call(0.0))

    def test_plot_result_pauses_for_plot_result_pause_time(self):
        addon = Mock(ProfileMatcherAddon)
        addon.recenter = False
        addon.plot_result = True
        addon.plot_result_blocking = False
        addon.plot_result_pause_time = 0.0
        addon._interpolate_hist.return_value = (None, None)

        with patch("matplotlib.pyplot.pause") as pause:
            ProfileMatcherAddon.hamilton_to_density_function(
                addon, time_grid=None, deltaE_grid=None, hamilton_2D=None
            )

        pause.assert_called_once_with(0.0)

    def test_pause_times_default_to_previous_values(self):
        addon = ProfileMatcherAddon(
            hist_x=np.linspace(0, 1, 10), hist_y=np.ones(10)
        )
        self.assertEqual(addon.animate_fitting_pause_time, 1e-3)
        self.assertEqual(addon.plot_result_pause_time, 0.1)


if __name__ == "__main__":
    unittest.main()
