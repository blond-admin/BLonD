import unittest

import matplotlib.pyplot as plt
import numpy as np
from physics.impedances.compare_with_legacy.test_integration_InducedVoltageFreq import (
    DEV_PLOT,
)

from blond.acc_math.analytic.ellipse import (
    calc_ellipse_gamma,
    ellipse_residuals,
    fit_ellipse,
    get_points_on_ellipse,
    plot_ellipse,
    transform_twiss,
)


class TestCallables(unittest.TestCase):
    def test_calc_ellipse_gamma(self):
        result = calc_ellipse_gamma(
            alpha=1,
            beta=1,
        )
        self.assertEqual(result, 2.0)

    def test_ellipse_residuals(self):
        theta = np.linspace(0, 2 * np.pi, 10)
        x = np.sin(theta)
        y = np.cos(theta)

        result = ellipse_residuals(
            x=x,
            y=y,
            alpha=0,
            beta=1,
            epsilon=1,
        )
        np.testing.assert_allclose(
            result,
            np.zeros_like(result),
            atol=1e-15,
            rtol=0,
        )

    def test_fit_ellipse(self):
        theta = np.linspace(0, 2 * np.pi, 10)
        x = np.sin(theta)
        y = np.cos(theta)
        alpha, beta, epsilon = fit_ellipse(x, y)
        self.assertAlmostEqual(alpha, 0)
        self.assertAlmostEqual(beta, 1)
        self.assertAlmostEqual(epsilon, 1)

    def test_fit_ellipse_scaled(self):
        scale_x = 1e3
        scale_y = 1e-9
        theta = np.linspace(0, 2 * np.pi, 10)
        x = np.sin(theta) * scale_x
        y = np.cos(theta) * scale_y
        x += 1e12 * y
        epsilon_expected = scale_x * scale_y
        beta_expected = 2000000000000
        alpha, beta, epsilon = fit_ellipse(x, y, scale_x=1e3, scale_y=1e-9)
        print(alpha, beta, epsilon)
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(x, y)
            plot_ellipse(alpha=alpha, beta=beta, epsilon=epsilon)
            plt.show()
        self.assertAlmostEqual(alpha, -1, places=5)
        self.assertAlmostEqual(beta / beta_expected, 1, places=5)
        self.assertAlmostEqual(epsilon, epsilon_expected, places=5)

    def test_fit_ellipse_bug(self):
        x = [
            1.46498681511808e-13,
            1.141656512246134e-13,
            -1.3394118542062348e-14,
            -1.2978171677016255e-13,
            -1.3791705529661366e-13,
            -3.1014364937451536e-14,
            1.0175772504771023e-13,
            1.496526538824355e-13,
            7.272077087111255e-14,
            -6.486825232063233e-14,
            -1.483500015142775e-13,
            -1.0809149454144166e-13,
            2.2327235600192883e-14,
            1.3412258971951733e-13,
            1.340449175517764e-13,
            2.2159006247570104e-14,
            -1.0820995925193884e-13,
            -1.4831988895409022e-13,
            -6.47146796845705e-14,
            7.286970707372688e-14,
        ]
        y = [
            0.0,
            399802.9070409406,
            466126.1132160497,
            143648.75192491166,
            -298647.5044534823,
            -491838.81828507246,
            -274782.33349731227,
            171472.9101725638,
            474700.8433671321,
            381975.93869198574,
            -29359.00974168649,
            -416205.30750219675,
            -455890.49395218474,
            -115312.74798226255,
            321448.5419711921,
            490086.3102899492,
            249938.0648936102,
            -198686.0803693901,
            -481584.13113656495,
            -362787.9230721302,
        ]
        scale = np.max(x)
        twiss = fit_ellipse(x, y, scale_x=np.max(x), scale_y=np.max(y))
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(x, y)
            plot_ellipse(*twiss)
            plt.show()
        np.testing.assert_allclose(
            twiss,
            (
                -0.24165788308183464,
                3.1511321926481105e-19,
                7.208585223392236e-08,
            ),
        )

    def test_plot_ellipse(self):
        plot_ellipse(alpha=1, beta=2, epsilon=3)
        plt.clf()

    def test_transform_twiss(self):
        n_points = 21
        twiss_before = (0, 2, 3)
        twiss_after = (4, 1.5, 3)

        x, y = get_points_on_ellipse(*twiss_before, n_points)
        plt.scatter(x, y)
        x, y = transform_twiss(x, y, *twiss_before, *twiss_after)
        twiss_after_points = fit_ellipse(x, y)
        np.testing.assert_allclose(twiss_after_points, twiss_after)
        DEV_PLOT = True
        if DEV_PLOT:
            plt.scatter(x, y)

            plot_ellipse(*twiss_before, n_points)
            plot_ellipse(*twiss_after, n_points)

            plt.show()
