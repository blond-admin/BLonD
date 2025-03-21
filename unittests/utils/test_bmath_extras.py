import unittest

import numpy as np

from blond.utils.bmath_extras import mean_and_std


class MyTestCase(unittest.TestCase):
    def test_no_weights(self):
        array = np.random.randn(
            100,
        )
        weights = np.ones_like(array)
        mean, std = mean_and_std(array, weights)
        assert np.isclose(mean, np.mean(array)), f"{mean=}, {np.mean(array)=}"
        assert np.isclose(std, np.std(array)), f"{std=}, {np.std(array)=}"

    def test_weights(self):
        n = 50
        array = np.random.randn(
            100,
        )
        weights = np.ones_like(array)
        weights[:n] = 0  # changed to previous test
        mean, std = mean_and_std(array, weights)
        assert np.isclose(mean, np.mean(array[n:])), (
            f"{mean=}, {np.mean(array[n:])=}"
        )
        assert np.isclose(std, np.std(array[n:])), (
            f"{std=}, {np.std(array[n:])=}"
        )

    def test_weights2(self):
        array = np.random.randn(
            100,
        )
        weights = np.ones_like(array)
        weights *= 0.5  # changed to previous test
        mean, std = mean_and_std(array, weights)
        assert np.isclose(mean, np.mean(array))
        assert np.isclose(std, np.std(array))


if __name__ == "__main__":
    unittest.main()
