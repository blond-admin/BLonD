import unittest

import numpy as np

from blond.physics.feedbacks.accelerators.sps.helpers import (
    comb_filter,
    feedforward_filter_generator,
    feedforward_filter_TWC3,
    feedforward_filter_TWC4,
    feedforward_filter_TWC5,
    get_power_from_current,
    modulator,
    moving_average,
)


class TestPowerFromCurrent(unittest.TestCase):
    def test_real_current(self):
        current = 1
        matched_impedance = 50

        self.assertEqual(
            get_power_from_current(current, matched_impedance),
            25.0,
        )

        current_arr = current * np.ones(10, dtype=complex)

        np.testing.assert_array_equal(
            get_power_from_current(current_arr, matched_impedance),
            25.0 * np.ones(10, dtype=float),
        )

    def test_imaginary_current(self):
        current = 1j
        matched_impedance = 50

        self.assertEqual(
            get_power_from_current(current, matched_impedance),
            25.0,
        )

        current_arr = current * np.ones(10, dtype=complex)

        np.testing.assert_array_equal(
            get_power_from_current(current_arr, matched_impedance),
            25.0 * np.ones(10, dtype=float),
        )


class TestMovingAverage(unittest.TestCase):
    def test_step(self):
        n_points = 20
        n_mov = 5

        step_arr = np.zeros(n_points)
        step_arr[n_points // 2 :] = 1

        filtered_step = moving_average(x=step_arr, n_mov=n_mov)

        expected_result = np.zeros(n_points - n_mov + 1)
        expected_result[n_points // 2 :] = 1
        expected_result[n_points // 2 - n_mov : n_points // 2 + 1] = (
            np.linspace(0, 1, n_mov + 1)
        )

        np.testing.assert_almost_equal(filtered_step, expected_result)

        filtered_step = moving_average(
            x=step_arr, n_mov=n_mov, x_prev=np.zeros(n_mov)
        )
        expected_result = np.concatenate((np.zeros(n_mov), expected_result))

        np.testing.assert_almost_equal(filtered_step, expected_result)


class TestCombFilter(unittest.TestCase):
    def test_cancellation(self):
        n_points = 20

        dirac_delta = np.zeros(n_points)
        dirac_delta[0] = 1

        result = comb_filter(y=np.zeros(n_points), x=dirac_delta, a=0.5)

        np.testing.assert_almost_equal(0.5 * dirac_delta, result)


class TestModulator(unittest.TestCase):
    def test_modulation_to_frequency_and_back(self):
        n_points = 2000
        initial_frequency = 1e6
        sampling_time = 1 / initial_frequency
        final_frequency = 1.01e6

        time_arr = np.arange(n_points) * sampling_time

        dfreq = final_frequency - initial_frequency

        initial_arr = np.ones(n_points) * (1 + 1j * 0)

        modulated_signal = modulator(
            signal=initial_arr,
            omega_i=2 * np.pi * initial_frequency,
            omega_f=2 * np.pi * final_frequency,
            t_sampling=sampling_time,
        )

        expected_modulated_signal = np.exp(-1j * 2 * np.pi * dfreq * time_arr)

        np.testing.assert_almost_equal(
            modulated_signal.real, expected_modulated_signal.real
        )

        np.testing.assert_almost_equal(
            modulated_signal.imag, expected_modulated_signal.imag
        )

        remodulated_signal = modulator(
            signal=modulated_signal,
            omega_i=2 * np.pi * final_frequency,
            omega_f=2 * np.pi * initial_frequency,
            t_sampling=sampling_time,
        )

        np.testing.assert_almost_equal(
            remodulated_signal.real, initial_arr.real
        )

        np.testing.assert_almost_equal(
            remodulated_signal.imag, initial_arr.imag
        )

    def test_minimum_signal_length(self):
        n_points = 1
        initial_frequency = 1e6
        sampling_time = 1 / initial_frequency
        final_frequency = 1.01e6

        initial_arr = np.ones(n_points) * (1 + 1j * 0)

        with self.assertRaises(RuntimeError):
            modulated_signal = modulator(
                signal=initial_arr,
                omega_i=2 * np.pi * initial_frequency,
                omega_f=2 * np.pi * final_frequency,
                t_sampling=sampling_time,
            )


class TestFeedforwardFilterGenerator(unittest.TestCase):
    def test_3_section(self):
        fir_filter = feedforward_filter_generator(number_of_sections=3)

        np.testing.assert_array_equal(fir_filter, feedforward_filter_TWC3)

    def test_4_section(self):
        fir_filter = feedforward_filter_generator(number_of_sections=4)

        np.testing.assert_array_equal(fir_filter, feedforward_filter_TWC4)

    def test_5_section(self):
        fir_filter = feedforward_filter_generator(number_of_sections=5)

        np.testing.assert_array_equal(fir_filter, feedforward_filter_TWC5)

    def test_wrong_section(self):
        with self.assertRaises(ValueError):
            feedforward_filter_generator(number_of_sections=6)
