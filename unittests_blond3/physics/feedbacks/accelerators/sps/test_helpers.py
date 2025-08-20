import unittest

import numpy as np

from blond3.physics.feedbacks.accelerators.sps.helpers import get_power_gen_i, \
    modulator, moving_average, comb_filter

if __name__ == '__main__':
    unittest.main()


class TestGetPowerGenI(unittest.TestCase):
    def test_get_power_gen_i_1(self):
        I_gen_per_cav = np.linspace(0, 1, 10) - 1j * np.linspace(0, 1, 10)
        Z_0 = 15
        actual = get_power_gen_i(
            I_gen_per_cav=I_gen_per_cav,
            Z_0=15,
        )
        # Just reuse internal formula.
        # function definition can be refactored then.
        # TODO test if formula is correct
        expected = 0.5 * Z_0 * np.abs(I_gen_per_cav) ** 2
        np.testing.assert_allclose(actual, expected)


class TestModulator(unittest.TestCase):
    def setUp(self, f_rf=200.1e6, f_0=200.222e6, T_s=5e-10, n=1000):
        self.f_rf = f_rf  # initial frequency in Hz
        self.f_0 = f_0  # final frequency in Hz
        self.T_s = T_s  # sampling time
        self.n = n  # number of points

    def test_v1(self):
        # Forwards and backwards transformation of a sine wave
        signal = np.cos(
            2 * np.pi * np.arange(self.n) * self.f_rf * self.T_s
        ) + 1j * np.sin(2 * np.pi * np.arange(self.n) * self.f_rf * self.T_s)
        signal_1 = modulator(signal, self.f_rf, self.f_0, self.T_s)
        signal_2 = modulator(signal_1, self.f_0, self.f_rf, self.T_s)

        # Drop some digits to avoid rounding errors
        signal = np.around(signal, 12)
        signal_2 = np.around(signal_2, 12)
        self.assertSequenceEqual(
            signal.tolist(),
            signal_2.tolist(),
            msg="In TestModulator, initial and final signals do not match",
        )

    def test_v2(self):
        signal = np.array([42])

        with self.assertRaises(
            RuntimeError, msg="In TestModulator, no exception for wrong signal length"
        ):
            modulator(signal, self.f_rf, self.f_0, self.T_s)


class TestMovingAverage(unittest.TestCase):
    # Run before every test
    def setUp(self, N=3, x_prev=None):
        self.x = np.array([0, 3, 6, 3, 0, 3, 6, 3, 0], dtype=float)
        self.y = moving_average(self.x, N, x_prev)

    # Run after every test
    def tearDown(self):
        del self.x
        del self.y

    def test_1(self):
        self.setUp(N=3)
        self.assertEqual(
            len(self.x),
            len(self.y) + 3 - 1,
            msg="In TestMovingAverage, test_1: wrong array length",
        )
        self.assertSequenceEqual(
            self.y.tolist(),
            np.array([3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(),
            msg="In TestMovingAverage, test_1: arrays differ",
        )

    def test_2(self):
        self.setUp(N=4)
        self.assertEqual(
            len(self.x),
            len(self.y) + 4 - 1,
            msg="In TestMovingAverage, test_2: wrong array length",
        )
        self.assertSequenceEqual(
            self.y.tolist(),
            np.array([3, 3, 3, 3, 3, 3], dtype=float).tolist(),
            msg="In TestMovingAverage, test_2: arrays differ",
        )

    def test_3(self):
        self.setUp(N=3, x_prev=np.array([0, 3]))
        self.assertEqual(
            len(self.x),
            len(self.y),
            msg="In TestMovingAverage, test_3: wrong array length",
        )
        self.assertSequenceEqual(
            self.y.tolist(),
            np.array([1, 2, 3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(),
            msg="In TestMovingAverage, test_3: arrays differ",
        )


class TestComb(unittest.TestCase):
    def test_1(self):
        y = np.random.rand(42)

        self.assertListEqual(
            y.tolist(),
            comb_filter(y, y, 15 / 16).tolist(),
            msg="In TestComb test_1, filtered signal not correct",
        )

    def test_2(self):
        t = np.arange(0, 2 * np.pi, 2 * np.pi / 120)
        y = np.cos(t)
        # Shift cosine by quarter period
        x = np.roll(y, int(len(t) / 4))

        # Drop some digits to avoid rounding errors
        result = np.around(comb_filter(y, x, 0.5), 12)
        result_theo = np.around(np.sin(np.pi / 4 + t) / np.sqrt(2), 12)

        self.assertListEqual(
            result.tolist(),
            result_theo.tolist(),
            msg="In TestComb test_2, filtered signal not correct",
        )
