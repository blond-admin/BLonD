"""Physical-time contracts for the coarse-to-fine cavity handoff."""

import unittest
from unittest.mock import Mock

import numpy as np

from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackCoarseGrid
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)
from blond.physics.profiles import StaticProfile


class TestFineGridInitialization(unittest.TestCase):
    """Compare the handoff with analytic envelopes at histogram centres."""

    omega = 2 * np.pi * 1e9
    # One coarse cell per RF period, the first centre half a period in.
    coarse_step = 1e-9
    first_center = 0.5e-9
    # The carried state is timestamped at the centre PRECEDING the forward
    # span -- one coarse step before its first centre -- so that the first
    # cell can carry its own charge without the seed already containing it
    # (``_state_before_forward_span``).  With no earlier passage that
    # centre is virtual, and free evolution is referenced to it.  The
    # generator drive keeps its own origin: the command held over the step
    # into the first cell is the carried one, the bias, so a drive written
    # onto the grid only starts acting at ``first_center``.
    seed_time = first_center - coarse_step

    def make_feedback(self, left=101e-9, second_order=True, **kwargs):
        """Build a one-nanosecond window with a timestamped coarse seed."""
        profile = StaticProfile(
            cut_left=left, cut_right=left + 1e-9, n_bins=100
        )
        feedback = IQCavityFeedbackCoarseGrid(
            profile,
            R_over_Q=1,
            Q_L=100,
            generator_current_bias=0,
            n_cavities=1,
            initial_voltage=1,
            second_order_fine_grid_solver_enable=second_order,
            **kwargs,
        )
        feedback._rf_centers = np.arange(
            self.first_center, 200e-9, self.coarse_step
        )
        feedback._rf_centers_lengths = np.array([200])
        # The production initialiser: it sizes the three source-split
        # coarse grids and carries ``initial_voltage`` into the generator
        # component, which is the quantity the fine solve is seeded from.
        feedback.reset_arrays()
        feedback.beam_current_fine_grid = np.zeros(100, complex)
        return feedback

    @staticmethod
    def clear_seed(feedback):
        """Start from an empty cavity, with no carried voltage."""
        feedback._last_val_ant_voltage_gen = 0.0 + 0.0j
        feedback._last_val_ant_voltage_beam = 0.0 + 0.0j

    def test_decay_and_detuning_use_actual_sample_times(self):
        """A later window includes all free evolution since the seed."""
        for second_order in (False, True):
            for left in (0.5e-9, 1e-9, 101e-9):
                for detuning in (0, 2e7):
                    with self.subTest(
                        second_order=second_order,
                        left=left,
                        detuning=detuning,
                    ):
                        feedback = self.make_feedback(
                            left, second_order, delta_omega=detuning
                        )
                        exponent = -self.omega / 200 + 1j * detuning
                        expected = np.exp(
                            exponent
                            * (
                                copy_to_cpu(feedback.profile.hist_x)
                                - self.seed_time
                            )
                        )
                        feedback._resolve_fine_grid_voltage(self.omega)
                        np.testing.assert_allclose(
                            feedback.antenna_voltage_fine_grid,
                            expected,
                            rtol=1e-5 if not second_order else 1e-9,
                        )

    def test_carried_seed_is_composed_in_the_forward_frame(self):
        """The carried generator seed enters in the passage's frame."""
        # The drive is the bias, zero, so the seed is the only term: what
        # reaches the fine grid is the carried generator component times
        # the rotation of the span it continues. Nothing else in this
        # module keeps a seed AND a rotation at once, so without this a
        # composition that drops the factor goes unnoticed.
        for rotation in (1.0 + 0.0j, 1j):
            with self.subTest(rotation=rotation):
                feedback = self.make_feedback()
                feedback._generator_frame_rotation = rotation
                feedback._resolve_fine_grid_voltage(self.omega)
                expected = rotation * np.exp(
                    -self.omega
                    / 200
                    * (copy_to_cpu(feedback.profile.hist_x) - self.seed_time)
                )
                np.testing.assert_allclose(
                    feedback.antenna_voltage_fine_grid, expected, rtol=1e-9
                )

    def test_current_changes_in_gap_use_applied_history(self):
        """The drive is held until the next recorded coarse command."""
        feedback = self.make_feedback()
        self.clear_seed(feedback)
        # A finite rectangular pulse entirely before the profile window.
        feedback.generator_current_coarse_grid[10:20] = 0.01 + 0.02j
        pulse_start, pulse_end = feedback._rf_centers[[10, 20]]
        exponent = -self.omega / 200
        expected = (
            self.omega
            * (0.01 + 0.02j)
            * np.expm1(exponent * (pulse_end - pulse_start))
            / exponent
            * np.exp(
                exponent * (copy_to_cpu(feedback.profile.hist_x) - pulse_end)
            )
        )
        feedback._resolve_fine_grid_voltage(self.omega)
        np.testing.assert_allclose(
            feedback.antenna_voltage_fine_grid, expected, rtol=1e-9
        )

    def test_gap_drive_respects_frame_limit_and_cavity_count(self):
        """Reconstruction limits current without stepping the controller."""
        feedback = self.make_feedback()
        self.clear_seed(feedback)
        feedback.n_cavities = 3
        feedback._generator_frame_rotation = 1j
        feedback.generator_current_coarse_grid[:] = 0.1
        controller = GeneratorCurrentPIController(
            gain_proportional=0,
            gain_integral=0,
            generator_current_bias=0,
            max_output=0.01,
        )
        controller.update_generator_current = Mock(
            side_effect=AssertionError("Reconstruction must not run PI")
        )
        feedback._controller = controller
        exponent = -self.omega / 200
        # The command held over the step into the first cell is the
        # carried one (the zero bias), so the drive starts at the first
        # centre rather than at the seed centre.
        expected = (
            3
            * self.omega
            * 0.01j
            * np.expm1(
                exponent
                * (copy_to_cpu(feedback.profile.hist_x) - self.first_center)
            )
            / exponent
        )
        feedback._resolve_fine_grid_voltage(self.omega)
        np.testing.assert_allclose(
            feedback.antenna_voltage_fine_grid, expected, rtol=1e-9
        )
        controller.update_generator_current.assert_not_called()

    def test_window_translation_preserves_overlapping_samples(self):
        """An empty prefix changes neither physical time nor voltage."""
        early = self.make_feedback(left=100.5e-9)
        later = self.make_feedback(left=101e-9)
        for feedback in (early, later):
            feedback._resolve_fine_grid_voltage(self.omega)
        np.testing.assert_allclose(
            early.antenna_voltage_fine_grid[50:],
            later.antenna_voltage_fine_grid[:50],
            rtol=1e-9,
        )

    def test_first_bin_has_half_bin_self_loading(self):
        """The first centre is half a bin after the beam-current onset."""
        for second_order in (False, True):
            with self.subTest(second_order=second_order):
                feedback = self.make_feedback(second_order=second_order)
                self.clear_seed(feedback)
                feedback.beam_current_fine_grid[0] = 0.02
                feedback._resolve_fine_grid_voltage(self.omega)
                exponent = -self.omega / 200
                half_step = feedback.profile.hist_step / 2
                expected = (
                    -self.omega
                    * 0.01
                    * (np.expm1(exponent * half_step) / exponent)
                )
                self.assertAlmostEqual(
                    feedback.antenna_voltage_fine_grid[0],
                    expected,
                    delta=abs(expected) * 1e-4,
                )

    def test_span_coarse_voltages_never_enter_the_fine_solve(self):
        """Only the carried state seeds it, never the span's own cells."""
        reference, contaminated = self.make_feedback(), self.make_feedback()
        # The coarse voltages of the very span being resolved, which the
        # fine grid re-derives from the seed and the fine beam current.
        # Seeding from any of them -- as this handoff did before the seed
        # moved ahead of the span -- would show up here.
        contaminated.antenna_voltage_gen_coarse_grid[:] = 1e9j
        contaminated.antenna_voltage_beam_coarse_grid[:] = 1e9j
        # The composed grid as well: it is the one the handoff read
        # before the seed moved ahead of the span, so a regression to
        # that spelling has to show up here too.
        contaminated.antenna_voltage_coarse_grid[:] = 1e9j
        for feedback in (reference, contaminated):
            feedback._resolve_fine_grid_voltage(self.omega)
        np.testing.assert_array_equal(
            reference.antenna_voltage_fine_grid,
            contaminated.antenna_voltage_fine_grid,
        )
        # Non-degeneracy: the comparison above is only worth something
        # while the fine grid carries voltage at all, and while the seed
        # it is allowed to read does reach it.
        self.assertGreater(
            float(np.max(np.abs(reference.antenna_voltage_fine_grid))),
            0.0,
            "the reference carries no voltage, so equality is vacuous",
        )
        seeded = self.make_feedback()
        seeded._last_val_ant_voltage_gen = 2.0 + 0.0j
        seeded._resolve_fine_grid_voltage(self.omega)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        seeded.antenna_voltage_fine_grid
                        - reference.antenna_voltage_fine_grid
                    )
                )
            ),
            0.0,
            "the carried seed does not reach the fine grid, so the "
            "comparison above could not have failed",
        )

    def test_localized_charge_has_no_early_kick_and_is_counted_once(self):
        """Each bin contributes half at its centre and the rest afterward."""
        for second_order in (False, True):
            for charged_bin in (0, 50):
                with self.subTest(
                    second_order=second_order, charged_bin=charged_bin
                ):
                    feedback = self.make_feedback(second_order=second_order)
                    feedback.Q_L = np.inf
                    self.clear_seed(feedback)
                    feedback.beam_current_fine_grid[charged_bin] = 0.02
                    feedback._resolve_fine_grid_voltage(self.omega)
                    full_kick = -self.omega * 0.01 * feedback.profile.hist_step
                    expected = np.zeros(100, complex)
                    expected[charged_bin] = full_kick / 2
                    expected[charged_bin + 1 :] = full_kick
                    np.testing.assert_allclose(
                        feedback.antenna_voltage_fine_grid,
                        expected,
                        rtol=1e-14,
                        atol=1e-16,
                    )

    def test_zero_exponent_has_finite_driven_limit(self):
        """The beam-free propagator also handles a lossless tuned cavity."""
        feedback = self.make_feedback()
        feedback.Q_L = np.inf
        feedback.generator_current_coarse_grid[:] = 0.01j
        feedback._resolve_fine_grid_voltage(self.omega)
        # A lossless tuned cavity does not decay, so the carried seed
        # contributes its unit amplitude whatever its timestamp; the
        # drive is the one term the time origin still shows in.
        expected = 1 + self.omega * 0.01j * (
            copy_to_cpu(feedback.profile.hist_x) - self.first_center
        )
        np.testing.assert_allclose(
            feedback.antenna_voltage_fine_grid, expected, rtol=1e-14
        )

    def test_circuit_track_hands_off_at_the_physical_seed_time(self):
        """Exercise the handoff after the real coarse recursion."""
        feedback = self.make_feedback(delta_omega=2e7)
        feedback._residual_time_last_rf_centers_calculation = 0
        feedback._last_rf_centers_entry = None
        feedback.antenna_voltage_coarse_grid = None
        feedback.reset_arrays()
        feedback.beam_current_forward_coarse_grid = np.zeros(200, complex)
        feedback.beam_current_fine_grid = np.zeros(100, complex)
        feedback.circuit_track(
            omega_input=self.omega,
            no_beam=False,
            start_index=0,
            end_index=200,
        )
        expected = feedback.antenna_voltage_coarse_grid[0] * np.exp(
            (-self.omega / 200 + 2e7j)
            * (copy_to_cpu(feedback.profile.hist_x) - self.first_center)
        )
        np.testing.assert_allclose(
            feedback.antenna_voltage_fine_grid, expected, rtol=1e-9
        )
