# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests for the proportional generator-current controller.

The P controller is a second control law beside the PI one, not a PI with
its integral gain set to zero: it has no integral, no anti-windup and no
integral state to carry, and its compiled form is its own. These tests pin
the law, its delay line, its clamp and its compiled-scan handoff in
isolation; the kernel-against-reference identity lives in
``test_envelope_kernel.py``.
"""

import collections
import unittest

import numpy as np

from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentController,
    GeneratorCurrentPController,
    GeneratorCurrentPIController,
    clamp_magnitude,
)

GAIN = 1.7
BIAS = 0.2 + 0.05j


def _controller(n_delay=0, max_output=None, gain=GAIN, bias=BIAS):
    return GeneratorCurrentPController(
        gain_proportional=gain,
        generator_current_bias=bias,
        n_delay=n_delay,
        max_output=max_output,
    )


class TestProportionalLaw(unittest.TestCase):
    """``I = clamp(I_0 + K_p e_d)``, and nothing else."""

    def test_is_a_controller_and_not_a_pi(self):
        """A separate law, not the PI one with its integral switched off."""
        controller = _controller()
        self.assertIsInstance(controller, GeneratorCurrentController)
        self.assertNotIsInstance(controller, GeneratorCurrentPIController)
        self.assertFalse(
            issubclass(
                GeneratorCurrentPController, GeneratorCurrentPIController
            )
        )
        self.assertFalse(hasattr(controller, "gain_integral"))
        self.assertFalse(hasattr(controller, "integral"))

    def test_output_is_the_bias_plus_the_gain_times_the_error(self):
        controller = _controller()
        error = 1.0e-3 + 0.5e-3j
        self.assertEqual(
            controller.update_generator_current(error, 1e-9),
            BIAS + GAIN * error,
        )

    def test_a_constant_error_gives_a_constant_output(self):
        """No integrator: the command never ramps under a steady error.

        This is the defining difference from the PI law, whose output
        grows linearly under the same input.
        """
        controller = _controller()
        error = 2.0e-3 - 1.0e-3j
        outputs = [
            controller.update_generator_current(error, 1e-9)
            for _ in range(1000)
        ]
        self.assertEqual(len(set(outputs)), 1)
        self.assertEqual(outputs[0], BIAS + GAIN * error)

    def test_the_sample_time_does_not_enter(self):
        """A proportional law has no time constant of its own."""
        error = 1.0e-3
        short = _controller().update_generator_current(error, 1e-12)
        long = _controller().update_generator_current(error, 1e-3)
        self.assertEqual(short, long)

    def test_the_delay_counts_samples(self):
        n_delay = 3
        controller = _controller(n_delay=n_delay)
        error = 1.0e-3
        for _ in range(n_delay):
            self.assertEqual(
                controller.update_generator_current(error, 1e-9), BIAS
            )
        self.assertEqual(
            controller.update_generator_current(error, 1e-9),
            BIAS + GAIN * error,
        )

    def test_no_state_outlives_the_delay_line(self):
        """Two histories agree once the delay line holds the same errors.

        The delay line is the whole of the P controller's memory, so it
        forgets everything older than ``n_delay + 1`` samples; a PI would
        not, because its integral remembers the whole run.
        """
        n_delay = 4
        rng = np.random.default_rng(3)
        first, second = _controller(n_delay=n_delay), _controller(n_delay)
        for _ in range(50):
            first.update_generator_current(
                complex(rng.normal(), rng.normal()), 1e-9
            )
        for _ in range(7):
            second.update_generator_current(-5.0 + 2.0j, 1e-9)
        shared = [complex(rng.normal(), rng.normal()) for _ in range(40)]
        tail_first = [first.update_generator_current(e, 1e-9) for e in shared]
        tail_second = [
            second.update_generator_current(e, 1e-9) for e in shared
        ]
        self.assertEqual(tail_first[n_delay + 1 :], tail_second[n_delay + 1 :])

    def test_matches_a_deque_reference_bit_for_bit(self):
        """The circular buffer behaves exactly like a deque delay line."""
        rng = np.random.default_rng(7)
        for n_delay in (0, 1, 5, 137):
            for max_output in (None, 0.5):
                with self.subTest(n_delay=n_delay, max_output=max_output):
                    controller = _controller(n_delay, max_output)
                    line = collections.deque(
                        [0.0 + 0.0j] * (n_delay + 1), maxlen=n_delay + 1
                    )
                    for _ in range(300):
                        error = 1e-3 * complex(rng.normal(), rng.normal())
                        line.append(error)
                        expected = clamp_magnitude(
                            BIAS + GAIN * line[0], max_output
                        )
                        self.assertEqual(
                            controller.update_generator_current(error, 3e-10),
                            expected,
                        )


class TestProportionalClamp(unittest.TestCase):
    """The klystron limit bounds the magnitude and keeps the phase."""

    def test_output_is_clamped_with_its_phase_kept(self):
        controller = _controller(max_output=0.5)
        error = 1.0 + 1.0j
        output = controller.update_generator_current(error, 1e-9)
        unclamped = BIAS + GAIN * error
        self.assertAlmostEqual(abs(output), 0.5, places=14)
        self.assertAlmostEqual(
            np.angle(output), np.angle(unclamped), places=14
        )

    def test_limit_clamps_an_array(self):
        controller = _controller(max_output=0.5)
        currents = np.array([0.1 + 0.0j, 3.0 + 4.0j])
        limited = controller.limit(currents)
        self.assertEqual(limited[0], currents[0])
        self.assertAlmostEqual(abs(limited[1]), 0.5, places=14)

    def test_limit_is_a_no_op_without_a_limit(self):
        currents = np.array([10.0 + 0.0j, 3.0 + 4.0j])
        self.assertTrue(
            np.array_equal(_controller().limit(currents), currents)
        )


class TestProportionalConstruction(unittest.TestCase):
    """The delay is fixed at construction, as for the PI."""

    def test_negative_delay_is_refused(self):
        with self.assertRaises(ValueError):
            _controller(n_delay=-1)

    def test_n_delay_is_read_only(self):
        controller = _controller(n_delay=5)
        self.assertEqual(controller.n_delay, 5)
        with self.assertRaises(AttributeError):
            controller.n_delay = 6


class TestProportionalScanHandoff(unittest.TestCase):
    """The P law has its own compiled scan and carries no integral."""

    def test_advertises_a_compiled_scan_of_its_own(self):
        from blond.physics.feedbacks import control_law_kernels

        controller = _controller()
        self.assertTrue(controller.supports_envelope_scan)
        self.assertIs(
            controller.envelope_scan_kernel(),
            control_law_kernels.envelope_p_scan,
        )
        self.assertIsNot(
            controller.envelope_scan_kernel(),
            control_law_kernels.envelope_pi_scan,
        )

    def test_scan_state_carries_no_integral(self):
        """Gains, bias, delay line and limit: no running integral."""
        state = _controller(n_delay=3, max_output=0.5).envelope_scan_state()
        gain, bias, buffer, head, max_output = state
        self.assertEqual((gain, bias, head, max_output), (GAIN, BIAS, 0, 0.5))
        self.assertEqual(buffer.shape, (4,))

    def test_an_unlimited_controller_hands_the_kernel_infinity(self):
        self.assertEqual(_controller().envelope_scan_state()[-1], np.inf)

    def test_scan_state_round_trip_is_lossless(self):
        controller = _controller(n_delay=9)
        rng = np.random.default_rng(11)
        for _ in range(25):
            controller.update_generator_current(
                1e-3 * complex(rng.normal(), rng.normal()), 3e-10
            )
        before = list(controller._delay_line)
        _, _, buffer, head, _ = controller.envelope_scan_state()
        controller.absorb_envelope_scan_state((buffer, head))
        self.assertEqual(list(controller._delay_line), before)

    def test_scan_state_hands_out_a_copy(self):
        """A scan whose result is never absorbed leaves the state alone."""
        controller = _controller(n_delay=4)
        controller.update_generator_current(1e-3 + 0j, 3e-10)
        before = list(controller._delay_line)
        buffer = controller.envelope_scan_state()[2]
        buffer[:] = -99.0
        self.assertEqual(list(controller._delay_line), before)


if __name__ == "__main__":
    unittest.main()
