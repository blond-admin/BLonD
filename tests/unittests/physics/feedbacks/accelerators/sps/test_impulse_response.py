import unittest

import numpy as np

from blond.physics.feedbacks.accelerators.sps.impulse_response import (
    SPS3Section200MHzTWC,
    SPS4Section200MHzTWC,
    SPS5Section200MHzTWC,
    rectangle,
    triangle,
)


class TestRectangleAndTriangleFunctions(unittest.TestCase):
    def test_rectangle_function(self):
        n_points = 200
        tau = 1

        x_arr = np.linspace(0, 2, n_points)

        y_arr = rectangle(t=x_arr - tau / 2, tau=tau)

        self.assertEqual(0.5, y_arr[0])

        self.assertAlmostEqual(
            float(np.mean(y_arr[(0 < x_arr) & (x_arr < tau)])), 1.0
        )

        self.assertAlmostEqual(float(np.mean(y_arr[x_arr > tau][1:])), 0.0)

        # TODO: implement signal where the falling edge is outside the window

    def test_rectangle_fail(self):

        # Check fail when time array does not start at rising edge
        with self.assertRaises(RuntimeError):
            n_points = 200
            tau = 1

            x_arr = np.linspace(0, 2, n_points)

            y_arr = rectangle(t=x_arr, tau=tau)

        # Check fail when the time array has multiple falling edges
        with self.assertRaises(RuntimeError):
            n_points = 200
            tau = 1

            x_arr = np.linspace(0, 2, n_points)
            x_arr = np.concatenate((x_arr, x_arr[n_points // 2 :]))

            y_arr = rectangle(t=x_arr - tau / 2, tau=tau)

    def test_triangle_function(self):
        n_points = 200
        tau = 1

        x_arr = np.linspace(0, 2, n_points)

        y_arr = triangle(t=x_arr, tau=tau)

        self.assertEqual(0.5, y_arr[0])

        self.assertAlmostEqual(
            float(
                np.mean(np.gradient(y_arr[(0 < x_arr) & (x_arr < tau)]))
                / np.mean(np.gradient(x_arr))
            ),
            -1.0,
        )

        self.assertAlmostEqual(float(np.mean(y_arr[x_arr > tau][1:])), 0.0)

    def test_triangle_fail(self):
        with self.assertRaises(RuntimeError):
            n_points = 200
            tau = 1

            x_arr = np.linspace(0, 2, n_points)

            y_arr = triangle(t=x_arr + tau / 2, tau=tau)


class TestSPS200MHzTravellingWaveCavities(unittest.TestCase):
    def test_3section_cavities(self):
        tws = SPS3Section200MHzTWC()
        pass

    def test_4section_cavities(self):
        # TODO: implement
        tws = SPS4Section200MHzTWC()
        pass

    def test_5section_cavities(self):
        # TODO: implement
        tws = SPS5Section200MHzTWC()
        pass
