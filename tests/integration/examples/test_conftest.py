import os
import tempfile
from unittest import mock

import matplotlib.pyplot as plt

from blond.testing.backend_testing import BLonDTestCase


class TestSkipPlotRendering(BLonDTestCase):
    def tearDown(self):
        plt.close("all")

    def test_pause_does_not_sleep(self):
        # The examples call `plt.pause` to animate their plots. With the
        # non-interactive test backend that is pure waiting time.
        plt.plot([0, 1])
        with mock.patch("time.sleep") as sleep:
            plt.pause(0.5)
        sleep.assert_not_called()

    def test_figures_are_never_rendered(self):
        # The example tests only check that the scripts run, so drawing
        # their figures is wasted time.
        figure = plt.figure()
        plt.plot([0, 1])
        with mock.patch.object(figure.canvas, "draw") as draw:
            plt.draw()
            plt.pause(0.5)
            plt.show()
        draw.assert_not_called()

    def test_savefig_writes_nothing(self):
        plt.plot([0, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "figure.png")
            plt.savefig(path)
            self.assertFalse(os.path.exists(path))
