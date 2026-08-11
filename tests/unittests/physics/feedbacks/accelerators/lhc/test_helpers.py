import unittest

import numpy as np

from blond.physics.feedbacks.accelerators.lhc.helpers import (
    cavity_response_sparse_matrix,
    fir_filter_lhc_otfb_coeff,
    ideal_switch_and_limit,
    klystron_saturation_curve,
)


class TestKlystronSaturationCurveModel(unittest.TestCase):
    def test_maximum_output_current(self):
        n_samples = 2000
        max_current = 1
        input_signal = np.linspace(0, 2, n_samples)

        output_signal = klystron_saturation_curve(
            predrive=input_signal,
            onset=0.8 * max_current,
            maximum_current=max_current,
            zero_gain_current=None,
        )

        self.assertAlmostEqual(max_current, np.max(output_signal), places=5)

    def test_zero_gain_point(self):
        n_samples = 20000
        zero_point = 1
        input_signal = np.linspace(0, 2, n_samples)

        output_signal = klystron_saturation_curve(
            predrive=input_signal,
            onset=0.8 * zero_point,
            maximum_current=None,
            zero_gain_current=zero_point,
        )

        output_grad = np.gradient(output_signal)
        min_ind: int = np.argmin(np.abs(output_grad))

        self.assertAlmostEqual(zero_point, input_signal[min_ind], places=3)

    def test_incorrect_parameters(self):
        with self.assertRaises(ValueError):
            _ = klystron_saturation_curve(
                predrive=0,
                onset=0.8,
                maximum_current=None,
                zero_gain_current=None,
            )

        with self.assertRaises(ValueError):
            _ = klystron_saturation_curve(
                predrive=0,
                onset=0.8,
                maximum_current=1.0,
                zero_gain_current=1.0,
            )

        with self.assertRaises(ValueError):
            _ = klystron_saturation_curve(
                predrive=0,
                onset=1.1,
                maximum_current=1.0,
                zero_gain_current=None,
            )

        with self.assertRaises(ValueError):
            _ = klystron_saturation_curve(
                predrive=0,
                onset=1.1,
                maximum_current=None,
                zero_gain_current=1.1,
            )


class TestSwitchAndLimit(unittest.TestCase):
    def test_limiting(self):
        n_samples = 200
        max_current = 1
        input_signal = np.linspace(0, 2, n_samples)

        output_signal = ideal_switch_and_limit(
            signal=input_signal, limit=max_current
        )

        self.assertEqual(
            n_samples, len(output_signal[output_signal <= max_current])
        )

        self.assertEqual(max_current, np.max(output_signal))


class TestCavityResponseMatrix(unittest.TestCase):
    def test_beam_induced_voltage_lhc_flatbottom(self):
        # Initialize a single bunch simulation
        # calculate beam induced voltage using the resonator and the cavity response matrix
        pass

    def test_beam_induced_voltage_lhc_flattop(self):
        pass

    def test_constant_drive(self):
        pass

    def test_constant_drive_and_beam_induced_voltage(self):
        pass


class TestFIRFilterCoefficients(unittest.TestCase):
    def test_n_taps_short(self):
        pass

    def test_n_taps_long(self):
        pass

    def test_n_taps_wrong(self):
        pass
