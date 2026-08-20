import unittest

import numpy as np

from blond import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.accelerators.sps.impulse_response import (
    SPS3Section200MHzTWC,
    SPS4Section200MHzTWC,
    SPS5Section200MHzTWC,
    TravellingWaveCavity,
    rectangle,
    triangle,
)
from blond.physics.impedances.sources import TravelingWaveCavity as ImpTWC


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

        x_arr = np.linspace(0, 0.5, n_points)

        y_arr = rectangle(t=x_arr - tau / 2, tau=tau)

        self.assertEqual(0.5, y_arr[0])

        self.assertAlmostEqual(float(np.mean(y_arr[1:])), 1.0)

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


class TestTravellingWaveCavities(unittest.TestCase):
    def test_incorrect_group_velocity(self):
        with self.assertRaises(ValueError):
            twc = TravellingWaveCavity(
                l_cell=0.374,
                n_cells=32,
                rho=2.71e4,
                v_g=1.1,
                omega_r=2 * np.pi * 200.03766667e6,
            )

    def test_carrier_far_from_central_frequency(self):
        twc = SPS3Section200MHzTWC()
        n_points = 10_000
        time = 100e-9
        omega_c = 2 * np.pi * 20e6

        t_arr = np.linspace(0, time, n_points)

        # Test generator response
        with self.assertRaises(ValueError):
            twc.impulse_response_gen(omega_c=omega_c, time_coarse=t_arr)

        # Test beam response
        with self.assertRaises(ValueError):
            twc.impulse_response_beam(omega_c=omega_c, time_fine=t_arr)

        # Test beam response
        with self.assertRaises(ValueError):
            twc.impulse_response_beam(
                omega_c=omega_c, time_fine=t_arr, time_coarse=t_arr
            )

    def test_3section_cavities(self):
        tws = SPS3Section200MHzTWC()

        n_points = 10_000
        time = 400e-9

        t_arr = np.linspace(0, time, n_points)

        tws.impulse_response_gen(omega_c=tws.omega_r, time_coarse=t_arr)

        _t_arr = t_arr - t_arr[0]
        expected_output = (
            tws.R_gen / tws.tau * rectangle(_t_arr - 0.5 * tws.tau, tws.tau)
        ).astype(complex)

        np.testing.assert_array_equal(expected_output, tws.h_gen)

    def test_4section_cavities(self):
        tws = SPS4Section200MHzTWC()

        n_points = 10_000
        time = 400e-9

        t_arr = np.linspace(0, time, n_points)

        tws.impulse_response_beam(omega_c=tws.omega_r, time_fine=t_arr)

        _t_arr = t_arr - t_arr[0]
        expected_output = (
            -2 * tws.R_beam / tws.tau * triangle(_t_arr, tws.tau)
        ).astype(complex)

        np.testing.assert_array_equal(expected_output, tws.h_beam)

    def test_5section_cavities(self):
        tws = SPS5Section200MHzTWC()
        n_points = 10_000
        time = 400e-9

        t_arr = np.linspace(0, time, n_points)

        tws.impulse_response_beam(omega_c=tws.omega_r, time_fine=t_arr)

        _t_arr = t_arr - t_arr[0]
        expected_output = (
            -2 * tws.R_beam / tws.tau * triangle(_t_arr, tws.tau)
        ).astype(complex)

        np.testing.assert_array_equal(expected_output, tws.h_beam)

    def test_wake_field_computation(self):
        tws = SPS4Section200MHzTWC()

        n_points = 10_000
        time = 400e-9

        t_arr = np.linspace(0, time, n_points)

        tws.impulse_response_beam(omega_c=tws.omega_r, time_fine=t_arr)
        tws.impulse_response_gen(omega_c=tws.omega_r, time_coarse=t_arr)
        tws.compute_wakes(time=t_arr)

        tws_imp = ImpTWC(
            R_S=tws.R_beam,
            frequency_R=tws.omega_r / 2 / np.pi,
            a_factor=2 * np.pi * tws.tau,
        )

        imp_wake = tws_imp.wake_calc(backend.array(t_arr))

        np.testing.assert_allclose(
            actual=copy_to_cpu(tws.W_beam),
            desired=copy_to_cpu(-imp_wake),
        )
