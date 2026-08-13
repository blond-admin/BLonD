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
        # TODO: Initialize a single bunch simulation
        # TODO: calculate beam induced voltage using the resonator and the cavity response matrix
        pass

    def test_beam_induced_voltage_lhc_flattop(self):
        pass

    def test_constant_drive(self):
        n_bins = 20_000
        t_rf = 2.495080129972677e-09
        hist_step = t_rf / 100

        v_ant_init = 1e6 + 1j * 0

        r_over_q = 45
        ql = 20_000
        f_rf = 1 / t_rf
        f_r = f_rf  # 400.789e6

        i_beam = np.zeros((n_bins,), dtype=complex)
        i_gen = np.zeros((n_bins,), dtype=complex)
        i_gen_init = v_ant_init / r_over_q / ql / 2
        i_gen[:] = i_gen_init

        v_ant = cavity_response_sparse_matrix(
            i_beam=i_beam,
            i_gen=i_gen,
            n_samples=n_bins,
            v_ant_init=v_ant_init,
            i_gen_init=i_gen_init,
            samples_per_rf=2 * np.pi * hist_step / t_rf,
            r_over_q=r_over_q,
            q_l=ql,
            detuning=(f_r - f_rf) / f_rf,
        )

        self.assertEqual(v_ant.shape[0], n_bins)

        self.assertAlmostEqual(np.mean(v_ant.imag), 0.0)

        self.assertAlmostEqual(np.mean(v_ant.real), v_ant_init, places=5)

    def test_constant_drive_and_beam_induced_voltage(self):
        pass


class TestFIRFilterCoefficients(unittest.TestCase):
    def test_n_taps_short(self):
        taps = fir_filter_lhc_otfb_coeff(n_taps=15)

        coeff = [
            -0.0469,
            -0.016,
            0.001,
            0.0321,
            0.0724,
            0.1127,
            0.1425,
            0.1534,
            0.1425,
            0.1127,
            0.0724,
            0.0321,
            0.001,
            -0.016,
            -0.0469,
        ]

        self.assertEqual(coeff, taps)

    def test_n_taps_long(self):
        taps = fir_filter_lhc_otfb_coeff(n_taps=63)

        coeff = [
            -0.038636,
            -0.00687283,
            -0.00719296,
            -0.00733319,
            -0.00726159,
            -0.00694037,
            -0.00634775,
            -0.00548098,
            -0.00432789,
            -0.00288188,
            -0.0011339,
            0.00090253,
            0.00321323,
            0.00577238,
            0.00856464,
            0.0115605,
            0.0147307,
            0.0180265,
            0.0214057,
            0.0248156,
            0.0282116,
            0.0315334,
            0.0347311,
            0.0377502,
            0.0405575,
            0.0431076,
            0.0453585,
            0.047243,
            0.0487253,
            0.049782,
            0.0504816,
            0.0507121,
            0.0504816,
            0.049782,
            0.0487253,
            0.047243,
            0.0453585,
            0.0431076,
            0.0405575,
            0.0377502,
            0.0347311,
            0.0315334,
            0.0282116,
            0.0248156,
            0.0214057,
            0.0180265,
            0.0147307,
            0.0115605,
            0.00856464,
            0.00577238,
            0.00321323,
            0.00090253,
            -0.0011339,
            -0.00288188,
            -0.00432789,
            -0.00548098,
            -0.00634775,
            -0.00694037,
            -0.00726159,
            -0.00733319,
            -0.00719296,
            -0.00687283,
            -0.038636,
        ]

        self.assertEqual(coeff, taps)

    def test_n_taps_wrong(self):
        with self.assertRaises(ValueError):
            taps = fir_filter_lhc_otfb_coeff(n_taps=62)
