import unittest

import numpy as np

from blond.physics.feedbacks.buffers import TwoTurnArray


class TestTwoTurnArray(unittest.TestCase):
    @staticmethod
    def get_expected_array(start_slice, end_slice, n_points):
        expected_array = np.zeros(2 * n_points, dtype=float)
        expected_array[n_points : n_points + n_points // 2] = 1

        return expected_array[start_slice + n_points : end_slice + n_points]

    def test_getitem(self):
        n_points = 20
        two_turn_array = TwoTurnArray(n_samples=n_points, dtype=float)

        two_turn_array[: n_points // 2] = 1

        start_slice = -10
        end_slice = -5
        expected_array = self.get_expected_array(
            start_slice, end_slice, n_points
        )

        np.testing.assert_array_equal(
            expected_array, two_turn_array[start_slice:end_slice]
        )

        self.assertEqual(0, np.sum(two_turn_array[start_slice:end_slice]))

        start_slice = -10
        end_slice = 5
        expected_array = self.get_expected_array(
            start_slice, end_slice, n_points
        )

        np.testing.assert_array_equal(
            expected_array, two_turn_array[start_slice:end_slice]
        )

        self.assertEqual(5, np.sum(two_turn_array[start_slice:end_slice]))

        start_slice = 5
        end_slice = 15
        expected_array = self.get_expected_array(
            start_slice, end_slice, n_points
        )

        np.testing.assert_array_equal(
            expected_array, two_turn_array[start_slice:end_slice]
        )

        self.assertEqual(5, np.sum(two_turn_array[start_slice:end_slice]))

    def test_incorrect_getitem(self):
        n_points = 20
        two_turn_array = TwoTurnArray(n_samples=n_points, dtype=float)

        with self.assertRaises(IndexError):
            _ = two_turn_array[-30]

        with self.assertRaises(TypeError):
            _ = two_turn_array[1.0]

    def test_incorrect_setitem(self):
        n_points = 20
        two_turn_array = TwoTurnArray(n_samples=n_points, dtype=float)

        with self.assertRaises(IndexError):
            two_turn_array[-1] = 1

    def test_length(self):
        n_points = 20
        two_turn_array = TwoTurnArray(n_samples=n_points, dtype=float)

        self.assertEqual(n_points, len(two_turn_array))
