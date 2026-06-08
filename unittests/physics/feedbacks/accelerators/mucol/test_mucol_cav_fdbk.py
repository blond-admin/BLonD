import unittest
import warnings
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from blond import StaticProfile
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass

class TestCavityFeedback(unittest.TestCase):
    def setUp(self):
        # RCS1 4 stations
        self.prof = Mock(StaticProfile)
        self.prof.hist_x = np.linspace(5.791514370530446e-10, 1.7351942079901463e-09, num=1024)
        self.prof.hist_y = np.zeros(1024)
        self.prof.cut_left = self.prof.hist_x[0]

        self.R_over_Q = 518
        self.Q_L = 1287601.7251526634
        self.n_cavities = 42.217908605563096
        self.generator_current = (0.0233441090290177+0.04958176818202371j)
        self.initial_voltage = 30e6
        self.n_rf_periods_per_coarse_grid = 1  # TODO: check for 2 and 0.5
        self.delta_omega = -6717.47508329349

        self.cav_fdbk = IQCavityFeedbackTimingClass(
            profile=self.prof,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=self.generator_current,
            n_cavities=self.n_cavities,
            initial_voltage=self.initial_voltage,
            n_rf_periods_per_coarse_grid=self.n_rf_periods_per_coarse_grid,
            delta_omega=self.delta_omega,
        )

    def test_circuit_track_applies_delta_omega_phase_shift(self):
        """
        circuit_track() feeds `relative_detuning = delta_omega / omega_input`
        into cavity_response(), which advances the antenna voltage each
        coarse-grid step by a complex factor containing `1j * relative_detuning
        * omega_times_T_s == 1j * delta_omega * delta_t`.

        This test drives circuit_track() with a hand-built, constant-step
        rf_centers grid and zero generator/beam current (no_beam=True), so
        the antenna voltage on the coarse grid should evolve purely by that
        per-step complex multiplier. We recompute the expected trajectory
        from the multiplier directly and compare it element-wise to what
        circuit_track() actually produced.
        """
        omega_input = 2 * np.pi * 1e9
        n_steps = 50
        dt = 1e-9

        # Build a constant-step rf_centers grid covering a single segment.
        self.cav_fdbk.rf_centers = np.arange(1, n_steps + 1) * dt
        self.cav_fdbk.rf_centers_lengths = np.array([n_steps])
        self.cav_fdbk.residual_time_last_rf_centers_calculation = 0.0
        self.cav_fdbk.last_rf_centers_entry = None

        # Zero out generator/beam current contributions so only the
        # `(1 - 0.5*omega*dt/Q_L + 1j*relative_detuning*omega*dt)` term
        # governs the antenna voltage evolution.
        self.cav_fdbk.generator_current_constant = 0.0 + 0.0j
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(n_steps, dtype=complex)
        self.cav_fdbk.last_val_generator_current = 0.0 + 0.0j
        self.cav_fdbk.last_val_beam_current = 0.0 + 0.0j

        v0 = self.initial_voltage + 0.0j
        self.cav_fdbk.last_val_ant_voltage = v0
        self.cav_fdbk.antenna_voltage_coarse_grid = np.zeros(n_steps, dtype=complex)

        self.cav_fdbk.circuit_track(
            omega_input=omega_input,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        v = self.cav_fdbk.antenna_voltage_coarse_grid

        omega_times_T_s = omega_input * dt
        expected_multiplier = (
            1
            - 0.5 * omega_times_T_s / self.Q_L
            + 1j * self.delta_omega * dt
        )
        expected = v0 * expected_multiplier ** np.arange(1, n_steps + 1)

        np.testing.assert_allclose(v, expected, rtol=1e-12)

    def _patched_carrier_props(self, omega_carrier, sampling_time_coarse):
        return (
            patch.object(
                IQCavityFeedbackTimingClass,
                "omega_carrier",
                new_callable=PropertyMock,
                return_value=omega_carrier,
            ),
            patch.object(
                IQCavityFeedbackTimingClass,
                "sampling_time_coarse",
                new_callable=PropertyMock,
                return_value=sampling_time_coarse,
            ),
        )

    def test_on_init_simulation_warns_for_large_decay_per_step(self):
        # 0.5 * omega * dt / Q_L should be between the soft (0.1) and hard
        # (2.0) thresholds: large enough to warn, small enough not to raise
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-9
        self.cav_fdbk.Q_L = 10.0
        self.cav_fdbk.delta_omega = 0.0  # avoid triggering the other warning

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt:
            with self.assertWarns(UserWarning) as cm:
                self.cav_fdbk.on_init_simulation(simulation=Mock())
        self.assertIn("decay_per_step", str(cm.warning))

    def test_on_init_simulation_warns_for_large_detuning_phase_per_step(self):
        # delta_omega * dt should clearly exceed the 0.1 threshold
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-6
        self.cav_fdbk.Q_L = 1e12  # avoid triggering the decay warning
        self.cav_fdbk.delta_omega = 1e6

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt:
            with self.assertWarns(UserWarning) as cm:
                self.cav_fdbk.on_init_simulation(simulation=Mock())
        self.assertIn("detuning_phase_per_step", str(cm.warning))

    def test_on_init_simulation_no_warning_for_small_step_parameters(self):
        # both 0.5 * omega * dt / Q_L and delta_omega * dt are well below 0.1
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-9
        self.cav_fdbk.Q_L = 1e12
        self.cav_fdbk.delta_omega = 1.0

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.cav_fdbk.on_init_simulation(simulation=Mock())
        self.assertEqual(caught, [])

    def test_cavity_response_warns_for_large_beam_kick(self):
        # relative_kick should be between the soft (0.1) and hard (1.0)
        # thresholds: large enough to warn, small enough not to raise
        omega_times_T_s = 1.0
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [1.0 + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1e-3 + 0.0j]
        )

        with self.assertWarns(UserWarning) as cm:
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertIn("relative_kick", str(cm.warning))

    def test_cavity_response_no_warning_for_small_beam_kick(self):
        # beam_current * 0.5 * R_over_Q * omega_times_T_s is a tiny fraction
        # of the previous antenna voltage
        omega_times_T_s = 1e-9
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [self.initial_voltage + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1.0 + 0.0j]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertEqual(caught, [])

    def test_on_init_simulation_raises_for_unphysical_decay_per_step(self):
        # 0.5 * omega * dt / Q_L > 2.0 makes the Euler decay factor negative
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-3
        self.cav_fdbk.Q_L = 1.0
        self.cav_fdbk.delta_omega = 0.0  # avoid the other (hard) error first

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt:
            with self.assertRaises(ValueError) as cm:
                self.cav_fdbk.on_init_simulation(simulation=Mock())
        self.assertIn("decay_per_step", str(cm.exception))

    def test_on_init_simulation_raises_for_unphysical_detuning_phase_per_step(
        self,
    ):
        # delta_omega * dt > 2.0 makes the per-step rotation exceed one
        # step's worth of phase -- the discretization can no longer track
        # the cavity phase
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-6
        self.cav_fdbk.Q_L = 1e12  # avoid the decay error/warning
        self.cav_fdbk.delta_omega = 1e7

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt:
            with self.assertRaises(ValueError) as cm:
                self.cav_fdbk.on_init_simulation(simulation=Mock())
        self.assertIn("detuning_phase_per_step", str(cm.exception))

    def test_cavity_response_raises_for_unphysical_beam_kick(self):
        # beam-induced kick exceeds the previous antenna voltage itself,
        # i.e. the Euler step would flip the sign of the antenna voltage
        omega_times_T_s = 1.0
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [1.0 + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1e8 + 0.0j]
        )

        with self.assertRaises(ValueError) as cm:
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertIn("relative_kick", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
