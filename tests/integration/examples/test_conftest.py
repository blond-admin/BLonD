from unittest import mock

import matplotlib.pyplot as plt

from blond.testing.backend_testing import BLonDTestCase


class TestPauseWithoutSleep(BLonDTestCase):
    def tearDown(self):
        plt.close("all")

    def test_pause_does_not_sleep(self):
        # The examples call `plt.pause` to animate their plots. With the
        # non-interactive test backend that is pure waiting time.
        plt.plot([0, 1])
        with mock.patch("time.sleep") as sleep:
            plt.pause(0.5)
        sleep.assert_not_called()

    def test_pause_still_draws_the_figure(self):
        # Rendering is kept, so errors raised only while drawing a figure
        # still fail the example tests.
        figure = plt.figure()
        plt.plot([0, 1])
        with mock.patch.object(figure.canvas, "draw") as draw:
            plt.pause(0.5)
        draw.assert_called()
