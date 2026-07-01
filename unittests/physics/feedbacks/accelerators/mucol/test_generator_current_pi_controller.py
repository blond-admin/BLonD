"""Unit tests for the standalone generator-current PI controller."""

import unittest

import numpy as np

from blond.physics.feedbacks.generator_current_pi_controller import (
    GeneratorCurrentPIController,
    clamp_magnitude,
)


class TestClampMagnitude(unittest.TestCase):
    """Tests for the phase-preserving magnitude clamp helper."""

    def test_preserves_phase_and_clamps_magnitude(self):
        """Clamping keeps the phase and sets the magnitude to the limit."""
        i_max = 0.03
        value = 0.05 * np.exp(1j * 0.7)
        out = clamp_magnitude(value, i_max)
        self.assertAlmostEqual(np.abs(out), i_max, places=12)
        self.assertAlmostEqual(np.angle(out), np.angle(value), places=12)

    def test_leaves_small_values_unchanged(self):
        """A value below the limit passes through unchanged."""
        value = 0.01 * np.exp(1j * 1.2)
        np.testing.assert_allclose(
            clamp_magnitude(value, 0.03), value, rtol=1e-15
        )

    def test_handles_arrays_including_zero(self):
        """Array input is clamped element-wise; zero entries stay zero."""
        i_max = 0.03
        values = np.array([0.0 + 0.0j, 0.01j, -0.05, 0.04 + 0.04j])
        out = clamp_magnitude(values, i_max)
        np.testing.assert_allclose(
            np.abs(out), np.minimum(np.abs(values), i_max), rtol=1e-12
        )
        # Phases of the non-zero entries are preserved.
        np.testing.assert_allclose(
            np.angle(out[1:]), np.angle(values[1:]), rtol=1e-12
        )

    def test_none_limit_is_a_no_op(self):
        """With no limit the input is returned unchanged."""
        value = 123.0 + 456.0j
        self.assertEqual(clamp_magnitude(value, None), value)


class TestGeneratorCurrentPIController(unittest.TestCase):
    """Tests for the PI controller error -> generator-current mapping."""

    def test_feedforward_passthrough_with_zero_gains(self):
        """With zero gains the output is the feedforward, for any error."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0, gain_integral=0.0, feedforward=0.02
        )
        self.assertEqual(controller.update(error=1e6, delta_t=1e-9), 0.02)
        self.assertEqual(controller.update(error=-3e5j, delta_t=1e-9), 0.02)

    def test_proportional_only(self):
        """Pure P control returns feedforward + K_p * error."""
        controller = GeneratorCurrentPIController(
            gain_proportional=2e-8, gain_integral=0.0, feedforward=0.02
        )
        error = 1.0e6 + 0.5e6j
        out = controller.update(error=error, delta_t=1e-9)
        self.assertAlmostEqual(out, 0.02 + 2e-8 * error, places=15)

    def test_loop_delay_holds_output_until_error_propagates(self):
        """With n_delay samples the error acts only after n_delay updates."""
        n_delay = 3
        controller = GeneratorCurrentPIController(
            gain_proportional=1e-8,
            gain_integral=0.0,
            feedforward=0.02,
            n_delay=n_delay,
        )
        error = 1.0e6
        # The first n_delay updates still act on the zero-prefilled errors.
        for _ in range(n_delay):
            self.assertEqual(controller.update(error, delta_t=1e-9), 0.02)
        # The (n_delay + 1)-th update finally acts on the first error.
        out = controller.update(error, delta_t=1e-9)
        self.assertAlmostEqual(out, 0.02 + 1e-8 * error, places=15)

    def test_integral_accumulates_linearly(self):
        """Pure I control integrates a constant error linearly."""
        gain_integral = 3e-5
        delta_t = 2e-9
        error = 1.0e5
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=gain_integral,
            feedforward=0.02,
        )
        for step in range(1, 6):
            out = controller.update(error=error, delta_t=delta_t)
            expected_integral = step * error * delta_t
            self.assertAlmostEqual(
                controller.integral, expected_integral, places=15
            )
            self.assertAlmostEqual(
                out, 0.02 + gain_integral * expected_integral, places=15
            )

    def test_anti_windup_freezes_integral_while_saturated(self):
        """The integrator does not wind up while the output is clamped."""
        # A large error saturates the limited controller from the first step.
        limited = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=1.0,
            feedforward=0.0,
            max_output=1.0,
        )
        free = GeneratorCurrentPIController(
            gain_proportional=0.0, gain_integral=1.0, feedforward=0.0
        )
        for _ in range(10):
            out = limited.update(error=10.0, delta_t=1.0)
            free.update(error=10.0, delta_t=1.0)
        # The output is clamped and the integral never accumulated.
        self.assertLessEqual(np.abs(out), 1.0 + 1e-12)
        self.assertEqual(limited.integral, 0.0)
        # ...whereas without a limit it winds up.
        self.assertGreater(np.abs(free.integral), 90.0)

    def test_integral_resumes_after_desaturation(self):
        """Once the output is back in range the integrator accumulates again."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=1.0,
            feedforward=0.0,
            max_output=1.0,
        )
        for _ in range(5):
            controller.update(error=10.0, delta_t=1.0)
        self.assertEqual(controller.integral, 0.0)  # frozen while saturated
        # A small error keeps the output within the limit, so it integrates.
        controller.update(error=0.1, delta_t=1.0)
        self.assertAlmostEqual(controller.integral, 0.1, places=15)

    def test_output_magnitude_is_clamped_with_phase_preserved(self):
        """The returned command never exceeds the configured limit."""
        controller = GeneratorCurrentPIController(
            gain_proportional=1.0,
            gain_integral=0.0,
            feedforward=0.0,
            max_output=2.0,
        )
        error = 5.0 * np.exp(1j * 0.9)
        out = controller.update(error=error, delta_t=1e-9)
        self.assertAlmostEqual(np.abs(out), 2.0, places=12)
        self.assertAlmostEqual(np.angle(out), np.angle(error), places=12)

    def test_negative_delay_is_rejected(self):
        """A negative loop delay is a programming error."""
        with self.assertRaises(AssertionError):
            GeneratorCurrentPIController(
                gain_proportional=0.0,
                gain_integral=0.0,
                feedforward=0.0,
                n_delay=-1,
            )


if __name__ == "__main__":
    unittest.main()
