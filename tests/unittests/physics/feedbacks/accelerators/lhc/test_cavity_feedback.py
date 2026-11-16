"""
Unit tests for the LHC cavity feedback.

Every block of the model is tested on a small standalone setup with
100 coarse samples per turn, so the whole file runs in seconds. The
full-machine physics checks (transfer functions, coarse vs. fine grid,
klystron and tuner regressions) are in
``tests/integration/feedback_integration/test_lhc_cavity_feedback.py``.
"""

import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    proton,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.accelerators.lhc import (
    LHCCavityFeedback,
    LHCCavityFeedbackCommissioning,
)
from blond.physics.feedbacks.accelerators.lhc.cavity_feedback import (
    LHCCavityFeedbackCoarseBuffers,
)
from blond.physics.feedbacks.helpers import generate_white_noise

g_a = 6.79e-6
g_d = 10
g_o = 10
tau_a = 170e-6
tau_d = 400e-6
tau_o = 110e-6
tau_loop = 650e-9
tau_otfb = 1200e-9

# Small standalone setup for the block-level tests. The harmonic must be
# large enough that n_coarse exceeds the longest look-back in the model
# (n_otfb = 79 samples), otherwise the indexing reaches past one turn.
small_harmonic = 1000
small_f_rf = 400.789e6


def make_small_feedback(
    commissioning: LHCCavityFeedbackCommissioning | None = None,
    n_pretrack: int = 1,
    reset_buffers: bool = True,
    **kwargs,
) -> LHCCavityFeedback:
    """Build a standalone feedback with n_coarse=100.

    With ``reset_buffers`` the coarse buffers are replaced by zeros after
    the pre-tracking, so a test can fill exactly the samples it needs.
    """
    profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)
    cavity_feedback = LHCCavityFeedback(
        profile=profile,
        commissioning=commissioning,
        n_pretrack=n_pretrack,
        **kwargs,
    )
    cavity_feedback.disable_fine_grid = True
    cavity_feedback.set_hardware_commissioning(
        omega_rf=2 * np.pi * small_f_rf, harmonic=small_harmonic
    )
    if reset_buffers:
        cavity_feedback.buffers_coarse = LHCCavityFeedbackCoarseBuffers(
            samples_per_turn=cavity_feedback.n_coarse
        )
    return cavity_feedback


def stub_parent_rf_station(
    cavity_feedback: LHCCavityFeedback, voltage: float
) -> None:
    """Replace the RF-station getters so the feedback runs standalone."""
    omega_rf = 2 * np.pi * small_f_rf
    cavity_feedback.get_harmonic_and_omega_rf_phi_rf = lambda: (
        small_harmonic,
        omega_rf,
        0.0,
    )
    cavity_feedback.get_voltage_from_parent_rf_station = lambda: voltage


class TestLHCCavityFeedbackBlocks(unittest.TestCase):
    """Fast tests of the individual per-sample blocks of the model."""

    def test_small_setup_dimensions(self):
        cavity_feedback = make_small_feedback()
        self.assertEqual(cavity_feedback.n_coarse, 100)
        self.assertGreater(cavity_feedback.n_coarse, cavity_feedback.n_otfb)

    def test_update_fb_variables(self):
        detuning_frequency = 5e3
        cavity_feedback = make_small_feedback(
            f_c=small_f_rf + detuning_frequency
        )
        t_s = cavity_feedback.T_s
        self.assertAlmostEqual(t_s, 10 / small_f_rf)
        self.assertEqual(cavity_feedback.n_delay, round(tau_loop / t_s))
        self.assertEqual(cavity_feedback.n_fir, 31)
        self.assertEqual(cavity_feedback.n_otfb, round(tau_otfb / t_s) + 31)
        self.assertAlmostEqual(cavity_feedback.samples, 20 * np.pi)
        np.testing.assert_allclose(
            cavity_feedback.d_omega, 2 * np.pi * detuning_frequency, rtol=1e-6
        )
        self.assertAlmostEqual(
            cavity_feedback.detuning,
            cavity_feedback.d_omega / cavity_feedback.omega_c,
        )

    def _check_cavity_response(self, ind: int):
        cavity_feedback = make_small_feedback()
        cavity_feedback.detuning = 1e-5
        buffers = cavity_feedback.buffers_coarse
        i_gen_prev = 0.3 + 0.1j
        v_ant_prev = 5e5 - 2e4j
        i_beam_prev = 0.7 - 0.2j
        if ind == 0:
            buffers.i_gen.prev[-1] = i_gen_prev
            buffers.v_ant.prev[-1] = v_ant_prev
            buffers.i_beam.prev[-1] = i_beam_prev
        else:
            buffers.i_gen.curr[ind - 1] = i_gen_prev
            buffers.v_ant.curr[ind - 1] = v_ant_prev
            buffers.i_beam.curr[ind - 1] = i_beam_prev

        samples = cavity_feedback.samples
        cavity_feedback.ind = ind
        cavity_feedback.cavity_response(samples=samples)

        r_over_q = cavity_feedback.r_over_q
        expected = (
            i_gen_prev * r_over_q * samples
            + v_ant_prev
            * (
                1
                - 0.5 * samples / cavity_feedback.q_l
                + 1j * cavity_feedback.detuning * samples
            )
            - 0.5 * i_beam_prev * r_over_q * samples
        )
        np.testing.assert_allclose(buffers.v_ant.curr[ind], expected)

    def test_cavity_response_within_turn(self):
        self._check_cavity_response(ind=5)

    def test_cavity_response_reads_previous_turn(self):
        self._check_cavity_response(ind=0)

    def test_rf_feedback_closed_loop(self):
        commissioning = LHCCavityFeedbackCommissioning(
            open_otfb=True, d_phi_ad=90
        )
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        ind = 5
        setpoint = 1e5 + 2e4j
        v_ant = 9e4 + 1e4j
        v_feedback_in_prev = 8e3 - 1e3j
        v_otfb_ac_in_prev = 2e3 + 5e2j
        v_analog_in_prev = 7e3 + 3e2j
        i_analog_out_prev = 0.02 - 0.01j
        i_digital_out_prev = 0.03 + 0.02j
        buffers.v_setpoint.curr[ind] = setpoint
        buffers.v_ant.curr[ind] = v_ant
        buffers.v_feedback_in.curr[ind - 1] = v_feedback_in_prev
        buffers.v_otfb_ac_in.curr[ind - 1] = v_otfb_ac_in_prev
        buffers.v_analog_in.curr[ind - 1] = v_analog_in_prev
        buffers.i_analog_out.curr[ind - 1] = i_analog_out_prev
        buffers.i_digital_out.curr[ind - 1] = i_digital_out_prev

        t_s = cavity_feedback.T_s
        cavity_feedback.ind = ind
        cavity_feedback.rf_feedback(t_s=t_s)

        v_feedback_in = setpoint - v_ant
        v_otfb_ac_in = (
            (1 - t_s / tau_o) * v_otfb_ac_in_prev
            + v_feedback_in
            - v_feedback_in_prev
        )
        # OTFB is open, so the analog input is just the error signal
        v_analog_in = v_feedback_in
        i_analog_out = i_analog_out_prev * (1 - t_s / tau_a) + g_a * (
            v_analog_in - v_analog_in_prev
        )
        # d_phi_ad = 90 deg rotates the digital branch by 1j
        i_digital_out = (
            i_digital_out_prev * (1 - t_s / tau_d)
            + t_s / tau_d * g_a * g_d * 1j * v_feedback_in_prev
        )
        np.testing.assert_allclose(
            buffers.v_feedback_in.curr[ind], v_feedback_in
        )
        np.testing.assert_allclose(
            buffers.v_otfb_ac_in.curr[ind], v_otfb_ac_in
        )
        np.testing.assert_allclose(buffers.v_analog_in.curr[ind], v_analog_in)
        np.testing.assert_allclose(
            buffers.i_analog_out.curr[ind], i_analog_out
        )
        np.testing.assert_allclose(
            buffers.i_digital_out.curr[ind], i_digital_out
        )
        np.testing.assert_allclose(
            buffers.i_feedback_out.curr[ind], i_analog_out + i_digital_out
        )

    def test_rf_feedback_open_loop_ignores_antenna(self):
        commissioning = LHCCavityFeedbackCommissioning(open_loop=True)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        buffers.v_setpoint.curr[5] = 1e5
        buffers.v_ant.curr[5] = 9e4
        cavity_feedback.ind = 5
        cavity_feedback.rf_feedback(t_s=cavity_feedback.T_s)
        self.assertAlmostEqual(buffers.v_feedback_in.curr[5], 1e5)

    def test_rf_feedback_open_rffb_gives_no_output(self):
        commissioning = LHCCavityFeedbackCommissioning(open_rffb=True)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        buffers.v_setpoint.curr[5] = 1e5
        buffers.i_analog_out.curr[4] = 0.5
        cavity_feedback.ind = 5
        cavity_feedback.rf_feedback(t_s=cavity_feedback.T_s)
        self.assertNotEqual(buffers.i_analog_out.curr[5], 0)
        self.assertEqual(buffers.i_feedback_out.curr[5], 0)

    def test_rf_feedback_closed_otfb_feeds_analog_branch(self):
        commissioning = LHCCavityFeedbackCommissioning(open_otfb=False)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        buffers.v_setpoint.curr[5] = 1e5
        buffers.v_otfb_out.curr[4] = 3e3 + 1e3j
        cavity_feedback.ind = 5
        cavity_feedback.rf_feedback(t_s=cavity_feedback.T_s)
        self.assertNotEqual(buffers.v_otfb_out.curr[5], 0)
        np.testing.assert_allclose(
            buffers.v_analog_in.curr[5],
            buffers.v_feedback_in.curr[5] + buffers.v_otfb_out.curr[5],
        )

    def test_rf_feedback_otfb_excitation_is_injected(self):
        commissioning = LHCCavityFeedbackCommissioning(open_otfb=True)
        cavity_feedback = make_small_feedback(commissioning)
        cavity_feedback.excitation_otfb = True
        buffers = cavity_feedback.buffers_coarse
        buffers.v_setpoint.curr[5] = 1e5
        buffers.v_excitation.curr[5] = 2e3 - 4e2j
        cavity_feedback.ind = 5
        cavity_feedback.rf_feedback(t_s=cavity_feedback.T_s)
        np.testing.assert_allclose(
            buffers.v_analog_in.curr[5],
            buffers.v_feedback_in.curr[5] + (2e3 - 4e2j),
        )

    def test_one_turn_feedback(self):
        cavity_feedback = make_small_feedback()
        buffers = cavity_feedback.buffers_coarse
        n_coarse = cavity_feedback.n_coarse
        n_otfb = cavity_feedback.n_otfb
        ind = 50
        comb_one_turn_ago = 4e3 + 1e3j
        ac_in_delayed = 2e3 - 3e2j
        comb_three_samples_ago = 1e3 + 2e2j
        otfb_out_prev = 5e2 + 5e2j
        fir_out_prev = 1e2 - 1e2j
        # ind - n_coarse lands in the previous turn
        buffers.v_otfb_comb.prev[ind] = comb_one_turn_ago
        buffers.v_otfb_ac_in.curr[ind - n_coarse + n_otfb] = ac_in_delayed
        buffers.v_otfb_comb.curr[ind - 3] = comb_three_samples_ago
        buffers.v_otfb_out.curr[ind - 1] = otfb_out_prev
        buffers.v_otfb_fir_out.curr[ind - 1] = fir_out_prev

        t_s = cavity_feedback.T_s
        cavity_feedback.ind = ind
        cavity_feedback.one_turn_feedback(t_s=t_s)

        alpha_comb = 15 / 16
        comb = (
            alpha_comb * comb_one_turn_ago
            + g_o * (1 - alpha_comb) * ac_in_delayed
        )
        fir_coeff = cavity_feedback.fir_coeff
        fir_out = fir_coeff[0] * comb + fir_coeff[3] * comb_three_samples_ago
        otfb_out = (1 - t_s / tau_o) * otfb_out_prev + fir_out - fir_out_prev
        np.testing.assert_allclose(buffers.v_otfb_comb.curr[ind], comb)
        np.testing.assert_allclose(buffers.v_otfb_fir_out.curr[ind], fir_out)
        np.testing.assert_allclose(buffers.v_otfb_out.curr[ind], otfb_out)

    def test_swap_without_clamping_passes_through(self):
        cavity_feedback = make_small_feedback()
        buffers = cavity_feedback.buffers_coarse
        large_current = 10 * cavity_feedback.i_swap_threshold * 1j
        buffers.i_feedback_out.curr[5] = large_current
        cavity_feedback.ind = 5
        cavity_feedback.swap()
        self.assertEqual(buffers.i_swap_out.curr[5], large_current)

    def test_swap_clamping_limits_magnitude_keeps_phase(self):
        commissioning = LHCCavityFeedbackCommissioning(clamping=True)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        threshold = cavity_feedback.i_swap_threshold
        phase = 0.7
        buffers.i_feedback_out.curr[5] = 3 * threshold * np.exp(1j * phase)
        buffers.i_feedback_out.curr[6] = 0.5 * threshold * np.exp(1j * phase)
        for ind in (5, 6):
            cavity_feedback.ind = ind
            cavity_feedback.swap()
        np.testing.assert_allclose(
            buffers.i_swap_out.curr[5], threshold * np.exp(1j * phase)
        )
        np.testing.assert_allclose(
            buffers.i_swap_out.curr[6], buffers.i_feedback_out.curr[6]
        )

    def test_i_swap_threshold_from_power_limit(self):
        cavity_feedback = make_small_feedback(g_gen=2)
        expected = np.sqrt(2 * 300e3 / (45 * 20_000)) / 2
        self.assertAlmostEqual(cavity_feedback.i_swap_threshold, expected)

    def _check_generator_current_delay(self, ind: int):
        cavity_feedback = make_small_feedback(g_gen=2)
        buffers = cavity_feedback.buffers_coarse
        n_delay = cavity_feedback.n_delay
        delayed_predrive = 0.4 - 0.1j
        buffers.i_swap_out.curr[ind] = 0.1 + 0.2j
        if ind - n_delay >= 0:
            buffers.i_gen_predrive.curr[ind - n_delay] = delayed_predrive
        else:
            buffers.i_gen_predrive.prev[ind - n_delay] = delayed_predrive
        cavity_feedback.ind = ind
        cavity_feedback.generator_current()
        np.testing.assert_allclose(
            buffers.i_gen_test.curr[ind], 2 * (0.1 + 0.2j)
        )
        np.testing.assert_allclose(
            buffers.i_gen_predrive.curr[ind], 2 * (0.1 + 0.2j)
        )
        self.assertEqual(buffers.i_gen.curr[ind], delayed_predrive)

    def test_generator_current_loop_delay_within_turn(self):
        self._check_generator_current_delay(ind=40)

    def test_generator_current_loop_delay_reads_previous_turn(self):
        self._check_generator_current_delay(ind=10)

    def test_generator_current_open_drive_uses_offset(self):
        commissioning = LHCCavityFeedbackCommissioning(open_drive=True)
        offset = 3 + 1j
        cavity_feedback = make_small_feedback(
            commissioning, i_gen_offset=offset
        )
        buffers = cavity_feedback.buffers_coarse
        buffers.i_swap_out.curr[5] = 0.5
        cavity_feedback.ind = 5
        cavity_feedback.generator_current()
        self.assertEqual(buffers.i_gen_predrive.curr[5], offset)

    def test_generator_current_saturation(self):
        commissioning = LHCCavityFeedbackCommissioning(saturation=True)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        threshold = cavity_feedback.i_swap_threshold
        onset = 0.8 * threshold
        phase = -0.4
        magnitude = 0.95 * threshold
        buffers.i_swap_out.curr[5] = magnitude * np.exp(1j * phase)
        cavity_feedback.ind = 5
        cavity_feedback.generator_current()

        span = threshold - onset
        overshoot = magnitude - onset
        saturated = onset + overshoot - overshoot**3 / (3 * span**2)
        np.testing.assert_allclose(
            buffers.i_gen_predrive.curr[5], saturated * np.exp(1j * phase)
        )
        self.assertLess(saturated, magnitude)

    def test_generator_current_klystron_fir(self):
        commissioning = LHCCavityFeedbackCommissioning(enable_klystron=True)
        cavity_feedback = make_small_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse
        klystron_fir = cavity_feedback.klystron_fir
        # Unit impulse 3 samples back in this turn, 15 back across the turn
        buffers.i_gen_predrive.curr[57] = 1.0
        buffers.i_gen_predrive.prev[-5] = 1.0
        for ind in (60, 10):
            cavity_feedback.ind = ind
            cavity_feedback.generator_current()
        self.assertAlmostEqual(buffers.i_gen.curr[60], klystron_fir[3])
        self.assertAlmostEqual(buffers.i_gen.curr[10], klystron_fir[15])

    def _check_tuner_input(self, ind: int):
        cavity_feedback = make_small_feedback()
        buffers = cavity_feedback.buffers_coarse

        def write(array, index, value):
            if index >= 0:
                array.curr[index] = value
            else:
                array.prev[index] = value

        i_gen = 0.3 + 0.4j
        v_ant = 6e5 - 1e5j
        tuner_in_8 = 1e4 + 2e3j
        tuner_in_16 = -3e3 + 1e3j
        integrated_1 = 5e2 - 2e2j
        integrated_2 = 1e2 + 4e2j
        write(buffers.i_gen, ind, i_gen)
        write(buffers.v_ant, ind, v_ant)
        write(buffers.tuner_in, ind - 8, tuner_in_8)
        write(buffers.tuner_in, ind - 16, tuner_in_16)
        write(buffers.tuner_integrated, ind - 1, integrated_1)
        write(buffers.tuner_integrated, ind - 2, integrated_2)

        cavity_feedback.ind = ind
        cavity_feedback.tuner_input()

        tuner_in = i_gen * np.conj(v_ant)
        integrated = (
            (tuner_in - 2 * tuner_in_8 + tuner_in_16) / 64
            + 2 * integrated_1
            - integrated_2
        )
        np.testing.assert_allclose(buffers.tuner_in.curr[ind], tuner_in)
        np.testing.assert_allclose(
            buffers.tuner_integrated.curr[ind], integrated
        )

    def test_tuner_input_within_turn(self):
        self._check_tuner_input(ind=20)

    def test_tuner_input_reads_previous_turn(self):
        self._check_tuner_input(ind=1)

    def _run_tuner(self, open_tuner: bool):
        mu = -10
        voltage = 1e6
        commissioning = LHCCavityFeedbackCommissioning(
            open_tuner=open_tuner, mu=mu
        )
        cavity_feedback = make_small_feedback(commissioning)
        stub_parent_rf_station(cavity_feedback, voltage=voltage)
        n_coarse = cavity_feedback.n_coarse
        # min + max of the imaginary part is -3 + 5 = 2
        cavity_feedback.buffers_coarse.tuner_integrated.curr = 1j * (
            np.linspace(-3, 5, n_coarse)
        )
        initial_detuning = 1e-5
        cavity_feedback.detuning = initial_detuning
        initial_omega_c = cavity_feedback.omega_c
        cavity_feedback.tuner()
        dtune = -(mu / 2) * 2 / (voltage / 8) ** 2
        return cavity_feedback, initial_detuning, initial_omega_c, dtune

    def test_tuner_closed_updates_detuning(self):
        cavity_feedback, initial_detuning, initial_omega_c, dtune = (
            self._run_tuner(open_tuner=False)
        )
        expected_detuning = initial_detuning + dtune
        expected_d_omega = expected_detuning * initial_omega_c
        self.assertAlmostEqual(cavity_feedback.detuning, expected_detuning)
        np.testing.assert_allclose(cavity_feedback.d_omega, expected_d_omega)
        np.testing.assert_allclose(
            cavity_feedback.omega_c,
            2 * np.pi * small_f_rf + expected_d_omega,
        )

    def test_tuner_open_keeps_detuning(self):
        cavity_feedback, initial_detuning, _, _ = self._run_tuner(
            open_tuner=True
        )
        self.assertEqual(cavity_feedback.detuning, initial_detuning)

    def test_update_set_point(self):
        cavity_feedback = make_small_feedback()
        buffers = cavity_feedback.buffers_coarse
        n_coarse = cavity_feedback.n_coarse
        old_last_setpoint = 2e5 + 1e4j
        new_setpoint = np.full(n_coarse, 6e5 + 0j)
        buffers.v_setpoint.prev[-1] = old_last_setpoint
        cavity_feedback.get_next_setpoint = lambda: new_setpoint

        cavity_feedback.update_set_point()

        np.testing.assert_allclose(buffers.v_setpoint.curr, new_setpoint)
        # Previous turn is replaced by a linear ramp that starts at the old
        # last sample and heads towards the new set point
        ramp = buffers.v_setpoint.prev
        np.testing.assert_allclose(ramp[0], old_last_setpoint)
        steps = np.diff(ramp)
        np.testing.assert_allclose(steps, steps[0])
        self.assertGreater(steps[0].real, 0)
        self.assertLess(ramp[-1].real, new_setpoint[0].real)


class TestLHCCavityFeedbackStaticHelpers(unittest.TestCase):
    def test_half_detuning(self):
        detuning = LHCCavityFeedback.half_detuning(
            imag_peak_beam_current=2,
            r_over_q=45,
            rf_frequency=400e6,
            voltage=1e6,
        )
        self.assertAlmostEqual(detuning, -0.25 * 45 * 2 / 1e6 * 400e6)

    def test_half_detuning_power(self):
        power = LHCCavityFeedback.half_detuning_power(
            peak_beam_current=2, voltage=1e6
        )
        self.assertAlmostEqual(power, 0.125 * 2 * 1e6)

    def test_optimum_ql(self):
        q_l = LHCCavityFeedback.optimum_Q_L(detuning=-1e4, rf_frequency=4e8)
        self.assertAlmostEqual(q_l, 20_000)

    def test_optimum_ql_with_beam(self):
        optimum_q = LHCCavityFeedback.optimum_Q_L_beam(
            r_over_q=45, real_peak_beam_current=1.111, voltage=1e6
        )
        self.assertAlmostEqual(optimum_q, 20002.0002000)

    def test_linear_interp_scalar(self):
        x_arr = np.array([0, 1, 2, 3, 4, 5])
        y_arr = np.array([0, 1, 2, 3, 4, 5])

        # Extrapolation on both sides and interpolation inside
        for t_val in (-1, 6, 2.5):
            val = LHCCavityFeedback._linear_interp_scalar(
                x=x_arr, y=y_arr, t=t_val
            )
            self.assertAlmostEqual(val, t_val)

    def test_generator_power(self):
        cavity_feedback = make_small_feedback()
        buffers = cavity_feedback.buffers_coarse
        buffers.i_gen.prev[3] = 2.0
        buffers.i_gen.curr[7] = 1.0 + 1.0j
        power = cavity_feedback.generator_power()

        n_coarse = cavity_feedback.n_coarse
        expected = np.zeros(2 * n_coarse)
        expected[3] = 0.5 * 45 * 20_000 * 4.0
        expected[n_coarse + 7] = 0.5 * 45 * 20_000 * 2.0
        np.testing.assert_allclose(power, expected)


class TestLHCCavityFeedbackCommissioningModes(unittest.TestCase):
    n_pretrack = 3

    def test_without_excitation_stays_at_zero(self):
        cavity_feedback = make_small_feedback(
            LHCCavityFeedbackCommissioning(),
            n_pretrack=self.n_pretrack,
            reset_buffers=False,
        )
        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_ant.curr,
            np.zeros(cavity_feedback.n_coarse, dtype=complex),
        )

    def _excited_feedback(self, **commissioning_kwargs):
        commissioning = LHCCavityFeedbackCommissioning(**commissioning_kwargs)
        cavity_feedback = make_small_feedback(
            commissioning, n_pretrack=self.n_pretrack, reset_buffers=False
        )
        n_samples = self.n_pretrack * cavity_feedback.n_coarse
        noise = generate_white_noise(
            n_samples, commissioning.seed1, commissioning.seed2
        )
        self.assertEqual(len(cavity_feedback.v_excitation_out), n_samples)
        self.assertTrue(np.any(cavity_feedback.v_excitation_out != 0))
        return cavity_feedback, noise

    def test_excitation_at_set_point(self):
        cavity_feedback, noise = self._excited_feedback(excitation=True)
        n_coarse = cavity_feedback.n_coarse
        np.testing.assert_allclose(
            cavity_feedback.v_excitation_in, 1e3 * noise
        )
        # Last turn of the set point is the last slice of the noise
        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_setpoint.curr,
            1e3 * noise[-n_coarse:],
        )
        # The output records the antenna voltage
        np.testing.assert_allclose(
            cavity_feedback.v_excitation_out[-n_coarse:],
            cavity_feedback.buffers_coarse.v_ant.curr,
        )
        self.assertFalse(cavity_feedback.excitation_otfb)

    def test_excitation_at_otfb_measures_error_signal(self):
        cavity_feedback, noise = self._excited_feedback(excitation_otfb_1=True)
        n_coarse = cavity_feedback.n_coarse
        self.assertTrue(cavity_feedback.excitation_otfb)
        np.testing.assert_allclose(
            cavity_feedback.v_excitation_in, 1e4 * noise
        )
        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_excitation.curr,
            1e4 * noise[-n_coarse:],
        )
        np.testing.assert_allclose(
            cavity_feedback.v_excitation_out[-n_coarse:],
            cavity_feedback.buffers_coarse.v_feedback_in.curr,
        )

    def test_excitation_at_otfb_measures_otfb_output(self):
        cavity_feedback, _ = self._excited_feedback(excitation_otfb_2=True)
        n_coarse = cavity_feedback.n_coarse
        self.assertTrue(cavity_feedback.excitation_otfb)
        np.testing.assert_allclose(
            cavity_feedback.v_excitation_out[-n_coarse:],
            cavity_feedback.buffers_coarse.v_otfb_out.curr,
        )

    def test_otfb_excitation_without_measurement_point_records_nothing(self):
        cavity_feedback = make_small_feedback()
        cavity_feedback.track_no_beam_excitation_otfb(self.n_pretrack)
        self.assertTrue(np.any(cavity_feedback.v_excitation_in != 0))
        np.testing.assert_array_equal(
            cavity_feedback.v_excitation_out,
            np.zeros(self.n_pretrack * cavity_feedback.n_coarse),
        )


class TestLHCCavityFeedbackFineGrid(unittest.TestCase):
    def test_cavity_response_fine_matrix(self):
        cavity_feedback = make_small_feedback()
        stub_parent_rf_station(cavity_feedback, voltage=5e6)
        cavity_feedback.detuning = 1e-5
        n_coarse = cavity_feedback.n_coarse
        n_bins = cavity_feedback.profile.n_bins

        # Antenna voltage that is linear in the coarse sample index over
        # both turns, so the initial value has a closed form
        v_ant_start = 4e5 + 1e4j
        v_ant_slope = 1e2 - 3e1j
        v_ant_full = v_ant_start + v_ant_slope * np.arange(2 * n_coarse)
        cavity_feedback.buffers_coarse.v_ant.prev = v_ant_full[:n_coarse]
        cavity_feedback.buffers_coarse.v_ant.curr = v_ant_full[n_coarse:]

        rng = np.random.default_rng(42)
        i_gen_fine = rng.normal(size=n_bins + 1) + 1j * rng.normal(
            size=n_bins + 1
        )
        i_beam_fine = rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)
        cavity_feedback.buffers_fine.i_gen = i_gen_fine
        cavity_feedback.buffers_fine.i_beam = i_beam_fine

        cavity_feedback.cavity_response_fine_matrix()

        # Reference: step the same recurrence sample by sample
        hist_step = cavity_feedback.profile.hist_step
        samples_fine = 2 * np.pi * small_f_rf * hist_step
        self.assertAlmostEqual(cavity_feedback.samples_fine, samples_fine)
        # The profile grid lives on the device for GPU backends
        hist_x = copy_to_cpu(cavity_feedback.profile.hist_x)
        t_at_init = hist_x[0] - hist_step
        t_first = (
            cavity_feedback.rf_centers[0] - cavity_feedback.T_s * n_coarse
        )
        v_ant_init = (
            v_ant_start
            + v_ant_slope * (t_at_init - t_first) / cavity_feedback.T_s
        )
        gen_coeff = 45 * samples_fine
        beam_coeff = 0.5 * 45 * samples_fine
        decay = 1 - 0.5 * samples_fine / 20_000 + 1j * 1e-5 * samples_fine
        i_beam_padded = np.concatenate(([0j], i_beam_fine))
        v_ant = [v_ant_init]
        for k in range(1, n_bins + 1):
            v_ant.append(
                decay * v_ant[-1]
                + gen_coeff * i_gen_fine[k - 1]
                - beam_coeff * i_beam_padded[k - 1]
            )
        expected = cavity_feedback.n_cavities * np.array(v_ant[1:])

        np.testing.assert_allclose(
            cavity_feedback.buffers_fine.v_ant, expected, rtol=1e-10
        )

    def test_circuit_track_with_fine_grid_disabled(self):
        cavity_feedback = make_small_feedback()
        stub_parent_rf_station(cavity_feedback, voltage=5e6)
        n_bins = cavity_feedback.profile.n_bins
        cavity_feedback.buffers_fine.i_beam = np.zeros(n_bins, dtype=complex)
        v_ant_fine_before = np.copy(cavity_feedback.buffers_fine.v_ant)
        cavity_feedback.get_next_setpoint = lambda: np.full(
            cavity_feedback.n_coarse, 6e5 + 0j
        )

        # Two turns, so the start of the turn has seen generator current
        # despite the loop delay
        for _ in range(2):
            cavity_feedback.buffers_coarse.shift()
            cavity_feedback.circuit_track()

        # The generator current is still resampled onto the fine grid, but
        # the fine-grid antenna voltage is left untouched
        # The profile grid lives on the device for GPU backends
        hist_x = copy_to_cpu(cavity_feedback.profile.hist_x)
        hist_step = cavity_feedback.profile.hist_step
        expected_i_gen_fine = np.interp(
            np.concatenate(([hist_x[0] - hist_step], hist_x)),
            cavity_feedback.rf_centers,
            cavity_feedback.buffers_coarse.i_gen.curr,
        )
        self.assertTrue(np.any(expected_i_gen_fine != 0))
        np.testing.assert_allclose(
            cavity_feedback.buffers_fine.i_gen, expected_i_gen_fine
        )
        np.testing.assert_array_equal(
            cavity_feedback.buffers_fine.v_ant, v_ant_fine_before
        )


class TestLHCCavityFeedbackInSimulation(unittest.TestCase):
    """Smoke test: the feedback inside a real, very small simulation."""

    @classmethod
    def setUpClass(cls):
        momentum = 450e9
        alpha = 1 / 53.8**2
        # Same RF frequency as the LHC, but 1000 buckets so n_coarse = 100
        circumference = 26658.8832 * small_harmonic / 35640
        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_beta = np.sqrt(1 - (proton.mass / energy) ** 2)

        beam = Beam(1.6e11, proton)
        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")
        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )
        cavity = SingleHarmonicRFStation(
            voltage=5e6, phi_rf=0.0, harmonic=small_harmonic
        )
        omega_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, circumference
        )
        t_rf = 2 * np.pi / omega_rf
        # The profile must span more than one coarse sample (10 buckets)
        profile = StaticProfile(
            cut_left=-0.5 * t_rf, cut_right=24.5 * t_rf, n_bins=25 * 32
        )
        cavity_feedback = LHCCavityFeedback(
            profile,
            commissioning=LHCCavityFeedbackCommissioning(open_tuner=False),
            n_pretrack=1,
        )
        cavity.attach_cavity_feedback(cavity_feedback)

        ring = Ring(circumference)
        ring.add_elements([profile, cavity, lattice])
        simulation = Simulation(ring, cycle)
        simulation.prepare_beam(
            beam,
            BiGaussian(n_macroparticles=10_000, sigma_dt=0.3e-9, seed=1234),
        )
        dt = beam.write_partial_dt()
        dt += 5 * t_rf
        beam.write_partial_dt()
        simulation.finalize((beam,), 2)
        profile.track(beam)

        cls.cavity_feedback = cavity_feedback
        cls.detuning_before = cavity_feedback.detuning
        cavity_feedback.track(beam)

    def test_coarse_grid_size(self):
        self.assertEqual(self.cavity_feedback.n_coarse, 100)

    def test_beam_current_is_seen(self):
        i_beam = self.cavity_feedback.buffers_coarse.i_beam.curr
        self.assertGreater(np.max(np.abs(i_beam)), 0)

    def test_fine_grid_follows_coarse_grid(self):
        cavity_feedback = self.cavity_feedback
        v_ant_fine = cavity_feedback.buffers_fine.v_ant
        self.assertEqual(v_ant_fine.shape, (cavity_feedback.profile.n_bins,))
        self.assertTrue(np.all(np.isfinite(v_ant_fine)))
        # The fine grid holds the total voltage of all cavities; the profile
        # covers the first three coarse samples
        coarse_total = cavity_feedback.n_cavities * np.mean(
            np.abs(cavity_feedback.buffers_coarse.v_ant.curr[:3])
        )
        np.testing.assert_allclose(
            np.mean(np.abs(v_ant_fine)), coarse_total, rtol=0.05
        )
        self.assertEqual(
            cavity_feedback.relative_amplitude_correction.shape,
            v_ant_fine.shape,
        )

    def test_tuner_acts_on_detuning(self):
        self.assertNotEqual(
            self.cavity_feedback.detuning, self.detuning_before
        )
