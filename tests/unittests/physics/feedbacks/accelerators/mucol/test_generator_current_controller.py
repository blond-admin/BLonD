"""Unit tests for the standalone generator-current PI controller."""

import collections
import unittest

import numpy as np

from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentController,
    GeneratorCurrentPIController,
    clamp_magnitude,
    current_limit_from_power,
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


class TestCurrentLimitFromPower(unittest.TestCase):
    """Tests for the klystron power -> current-limit conversion."""

    def test_matched_generator_relation(self):
        """I_max = sqrt(2 P / ((R/Q) Q_L))."""
        p_max, r_over_q, q_l = 1e6, 518.0, 1.2876e6
        self.assertAlmostEqual(
            current_limit_from_power(p_max, r_over_q, q_l),
            np.sqrt(2.0 * p_max / (r_over_q * q_l)),
            places=15,
        )


class TestAbstractController(unittest.TestCase):
    """Tests for the abstract controller interface."""

    def test_cannot_instantiate_abstract_base(self):
        """The interface has an abstract update() and cannot be built."""
        with self.assertRaises(TypeError):
            GeneratorCurrentController()

    def test_default_limit_is_a_no_op(self):
        """The base limit() applies no actuator limit."""

        class _Trivial(GeneratorCurrentController):
            def update_generator_current(self, error, delta_t):
                return 0.0 + 0.0j

        value = np.array([5.0, -3.0j, 1.0 + 1.0j])
        np.testing.assert_array_equal(_Trivial().limit(value), value)

    def test_compiled_scan_interface_names_the_controller(self):
        """Without a compiled scan, each scan method raises by name.

        A controller that does not advertise ``supports_envelope_scan``
        must reject all three compiled-scan calls with a message naming
        the offending class, so a feedback wired to the wrong controller
        fails loudly instead of running a half-implemented scan.
        """

        class _Trivial(GeneratorCurrentController):
            def update_generator_current(self, error, delta_t):
                return 0.0 + 0.0j

        controller = _Trivial()
        self.assertFalse(controller.supports_envelope_scan)
        scan_calls = {
            "kernel": controller.envelope_scan_kernel,
            "state": controller.envelope_scan_state,
            "absorb": lambda: controller.absorb_envelope_scan_state(()),
        }
        for name, scan_call in scan_calls.items():
            with self.subTest(method=name):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "_Trivial supplies no compiled envelope scan",
                ):
                    scan_call()


class TestGeneratorCurrentPIController(unittest.TestCase):
    """Tests for the PI controller error -> generator-current mapping."""

    def test_constant_current_passthrough_with_zero_gains(self):
        """With zero gains the output is the constant current, for any error."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=0.0,
            generator_current_bias=0.02,
        )
        self.assertEqual(
            controller.update_generator_current(error=1e6, delta_t=1e-9), 0.02
        )
        self.assertEqual(
            controller.update_generator_current(error=-3e5j, delta_t=1e-9),
            0.02,
        )

    def test_proportional_only(self):
        """Pure P control returns constant current + K_p * error."""
        controller = GeneratorCurrentPIController(
            gain_proportional=2e-8,
            gain_integral=0.0,
            generator_current_bias=0.02,
        )
        error = 1.0e6 + 0.5e6j
        out = controller.update_generator_current(error=error, delta_t=1e-9)
        self.assertAlmostEqual(out, 0.02 + 2e-8 * error, places=15)

    def test_loop_delay_holds_output_until_error_propagates(self):
        """With n_delay samples the error acts only after n_delay updates."""
        n_delay = 3
        controller = GeneratorCurrentPIController(
            gain_proportional=1e-8,
            gain_integral=0.0,
            generator_current_bias=0.02,
            n_delay=n_delay,
        )
        error = 1.0e6
        # The first n_delay updates still act on the zero-prefilled errors.
        for _ in range(n_delay):
            self.assertEqual(
                controller.update_generator_current(error, delta_t=1e-9), 0.02
            )
        # The (n_delay + 1)-th update finally acts on the first error.
        out = controller.update_generator_current(error, delta_t=1e-9)
        self.assertAlmostEqual(out, 0.02 + 1e-8 * error, places=15)

    def test_integral_accumulates_linearly(self):
        """Pure I control integrates a constant error linearly."""
        gain_integral = 3e-5
        delta_t = 2e-9
        error = 1.0e5
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=gain_integral,
            generator_current_bias=0.02,
        )
        for step in range(1, 6):
            out = controller.update_generator_current(
                error=error, delta_t=delta_t
            )
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
            generator_current_bias=0.0,
            max_output=1.0,
        )
        free = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=1.0,
            generator_current_bias=0.0,
        )
        for _ in range(10):
            out = limited.update_generator_current(error=10.0, delta_t=1.0)
            free.update_generator_current(error=10.0, delta_t=1.0)
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
            generator_current_bias=0.0,
            max_output=1.0,
        )
        for _ in range(5):
            controller.update_generator_current(error=10.0, delta_t=1.0)
        self.assertEqual(controller.integral, 0.0)  # frozen while saturated
        # A small error keeps the output within the limit, so it integrates.
        controller.update_generator_current(error=0.1, delta_t=1.0)
        self.assertAlmostEqual(controller.integral, 0.1, places=15)

    def test_output_magnitude_is_clamped_with_phase_preserved(self):
        """The returned command never exceeds the configured limit."""
        controller = GeneratorCurrentPIController(
            gain_proportional=1.0,
            gain_integral=0.0,
            generator_current_bias=0.0,
            max_output=2.0,
        )
        error = 5.0 * np.exp(1j * 0.9)
        out = controller.update_generator_current(error=error, delta_t=1e-9)
        self.assertAlmostEqual(np.abs(out), 2.0, places=12)
        self.assertAlmostEqual(np.angle(out), np.angle(error), places=12)

    def test_limit_clamps_an_array_to_max_output(self):
        """Enforce the klystron limit on an external current array."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=0.0,
            generator_current_bias=0.0,
            max_output=0.03,
        )
        values = np.array([0.0 + 0.0j, 0.05, 0.04 + 0.04j])
        out = controller.limit(values)
        np.testing.assert_allclose(
            np.abs(out), np.minimum(np.abs(values), 0.03), rtol=1e-12
        )

    def test_limit_is_a_no_op_without_a_limit(self):
        """Without max_output, limit() returns the input unchanged."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=0.0,
            generator_current_bias=0.0,
        )
        value = 0.5 + 0.5j
        self.assertEqual(controller.limit(value), value)

    def test_negative_delay_is_rejected(self):
        """A negative loop delay is a programming error."""
        with self.assertRaises(AssertionError):
            GeneratorCurrentPIController(
                gain_proportional=0.0,
                gain_integral=0.0,
                generator_current_bias=0.0,
                n_delay=-1,
            )


class TestLoopDelayIsFixedAtConstruction(unittest.TestCase):
    """``n_delay`` is read-only: the delay line is sized once, in __init__.

    The deque keeps the ``maxlen`` it was built with, so a write after
    construction used to be silently ignored -- a user retuning the loop
    delay got no delay change and no complaint.  The attribute is exposed
    as a read-only property so that mistake fails loudly instead.
    """

    def _controller(self, n_delay: int = 3):
        """
        Build a PI controller with the given loop delay.

        Parameters
        ----------
        n_delay
            Loop delay in coarse-grid samples.

        Returns
        -------
        controller
            The controller under test.
        """
        return GeneratorCurrentPIController(
            gain_proportional=1e-8,
            gain_integral=0.0,
            generator_current_bias=0.02,
            n_delay=n_delay,
        )

    def test_n_delay_reads_back_the_constructor_value(self):
        """The documented read surface is unchanged."""
        for n_delay in (0, 1, 7):
            with self.subTest(n_delay=n_delay):
                self.assertEqual(
                    self._controller(n_delay=n_delay).n_delay, n_delay
                )

    def test_n_delay_is_a_property_over_private_storage(self):
        """The value lives on ``_n_delay``; ``n_delay`` is a property."""
        controller = self._controller(n_delay=4)
        self.assertEqual(controller._n_delay, 4)
        self.assertIsInstance(type(controller).__dict__["n_delay"], property)

    def test_assigning_n_delay_raises(self):
        """A post-construction retune fails loudly instead of silently."""
        controller = self._controller(n_delay=3)
        with self.assertRaises(AttributeError):
            controller.n_delay = 5

    def test_delay_line_length_tracks_the_constructor_value(self):
        """The deque is sized ``n_delay + 1`` from the constructor value."""
        for n_delay in (0, 1, 7):
            with self.subTest(n_delay=n_delay):
                controller = self._controller(n_delay=n_delay)
                self.assertEqual(len(controller._delay_line), n_delay + 1)
                self.assertEqual(controller._delay_line.maxlen, n_delay + 1)

    def test_int_coercion_still_applies(self):
        """A float loop delay is coerced to ``int``, as before."""
        controller = self._controller(n_delay=2.0)
        self.assertIsInstance(controller.n_delay, int)
        self.assertEqual(controller.n_delay, 2)


class TestPIErrorFrame(unittest.TestCase):
    """The PI error must reach the controller in the GENERATOR frame.

    The composed coarse voltage is
    ``V = V_beam + V_gen * exp(-i (delta_phi_rf + gap))`` and the error is
    read out in the kick frame, ``e_kick = setpoint - V * exp(+i gap)``.
    The controller's output is a generator current, which drives ``V_gen``
    in the design frame, so ``dV_kick / dI_gen`` carries
    ``exp(-i delta_phi_rf)``.  Handing ``e_kick`` straight to the controller
    therefore multiplies the open-loop gain by that rotation, which grows
    without bound while an RF-frequency offset is applied.  The error must
    be rotated back by ``exp(+i delta_phi_rf)`` first.
    """

    omega_rf = 2.0 * np.pi * 1.3e9

    def _feedback(self, delta_phi_rf: float, gap: float):
        """
        A feedback wired for a single ``_update_generator_current`` call.

        Parameters
        ----------
        delta_phi_rf
            Accumulated RF-frequency-offset phase of the station [rad].
        gap
            Live carrier slip gap of this passage [rad].

        Returns
        -------
        feedback
            The feedback, with a recording stub controller attached.
        """
        from unittest.mock import Mock, PropertyMock, patch

        from blond.physics.feedbacks.cavity_feedback import (
            IQCavityFeedbackTimingClass,
        )
        from blond.physics.profiles import StaticProfile

        feedback = IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            n_rf_periods_per_coarse_grid=1,
            R_over_Q=518.0,
            Q_L=1e6,
            generator_current_bias=0.0,
            n_cavities=1,
        )
        # delta_phi_rf is a read-only view of the parent station's kick
        # clock; patch it for the duration of the wiring.
        self._delta_phi_patch = patch.object(
            IQCavityFeedbackTimingClass,
            "delta_phi_rf",
            new_callable=PropertyMock,
            return_value=delta_phi_rf,
        )
        self._delta_phi_patch.start()
        self.addCleanup(self._delta_phi_patch.stop)
        feedback._carrier_slip_gap = gap
        feedback._update_frame_rotations()
        feedback._omega_input_for_pi = self.omega_rf
        feedback._voltage_setpoint = 0.0 + 0.0j
        feedback.antenna_voltage_coarse_grid = np.array(
            [1.0 + 0.0j], dtype=complex
        )
        feedback.generator_current_coarse_grid = np.zeros(1, dtype=complex)

        recorded: list[complex] = []

        class _RecordingController:
            def update_generator_current(self, error, delta_t):
                recorded.append(complex(error))
                return 0.0 + 0.0j

        feedback._controller = _RecordingController()
        return feedback, recorded

    def test_error_is_rotated_into_the_generator_frame(self):
        """The error carries ``exp(+i delta_phi_rf)``, cancelling the gain."""
        delta_phi_rf = 0.9
        gap = 0.0
        feedback, recorded = self._feedback(delta_phi_rf, gap)
        feedback._update_generator_current(
            omega_times_dt=2.0 * np.pi, coarse_grid_index_to_update=0
        )
        # setpoint 0, V = 1 (+0j), gap 0 -> e_kick = -1.  In the generator
        # frame that is -exp(+i delta_phi_rf).
        expected = -np.exp(1j * delta_phi_rf)
        self.assertAlmostEqual(recorded[0].real, expected.real, places=12)
        self.assertAlmostEqual(recorded[0].imag, expected.imag, places=12)

    def test_no_offset_leaves_the_error_untouched(self):
        """Without an RF-frequency offset the rotation is exactly unity."""
        feedback, recorded = self._feedback(0.0, 0.0)
        feedback._update_generator_current(
            omega_times_dt=2.0 * np.pi, coarse_grid_index_to_update=0
        )
        self.assertEqual(recorded[0], -1.0 + 0.0j)


class TestDelayLineStateHandoff(unittest.TestCase):
    """
    The delay line is a circular buffer, and must behave like the deque.

    The controller hands its delay line to the compiled envelope scan once
    per tracked span, so the representation is a performance decision: a
    deque has to be rebuilt element by element in Python on the way back,
    which is O(n_delay) per span and dominates once a physically slow LLRF
    pushes the delay to ~1300 samples. These tests pin the observable
    behaviour that refactor must not change.
    """

    @staticmethod
    def _reference(n_delay, max_output):
        """
        A deque-based PI update, as the controller used to be written.

        Parameters
        ----------
        n_delay
            Loop delay in samples.
        max_output
            Klystron current limit [A], or None.

        Returns
        -------
        step
            Callable taking ``(error, delta_t)`` and returning the
            generator current, closing over its own deque and integral.
        """
        gains = dict(kp=1.7, ki=3.1e5, bias=0.2 + 0.05j)
        line = collections.deque(
            [0.0 + 0.0j] * (n_delay + 1), maxlen=n_delay + 1
        )
        state = {"integral": 0.0 + 0.0j}

        def step(error, delta_t):
            line.append(error)
            delayed = line[0]
            candidate = state["integral"] + delayed * delta_t
            out = (
                gains["bias"] + gains["kp"] * delayed + gains["ki"] * candidate
            )
            saturated = max_output is not None and np.abs(out) > max_output
            if not saturated:
                state["integral"] = candidate
            return clamp_magnitude(out, max_output)

        return step

    def test_matches_the_deque_law_bit_for_bit(self):
        """Buffer and deque must agree exactly, limited and unlimited."""
        rng = np.random.default_rng(7)
        for n_delay in (0, 1, 2, 5, 20, 137):
            for max_output in (None, 0.5):
                with self.subTest(n_delay=n_delay, max_output=max_output):
                    controller = GeneratorCurrentPIController(
                        gain_proportional=1.7,
                        gain_integral=3.1e5,
                        generator_current_bias=0.2 + 0.05j,
                        n_delay=n_delay,
                        max_output=max_output,
                    )
                    reference = self._reference(n_delay, max_output)
                    for _ in range(500):
                        error = 1e-3 * complex(rng.normal(), rng.normal())
                        self.assertEqual(
                            controller.update_generator_current(error, 3e-10),
                            reference(error, 3e-10),
                        )

    def test_scan_state_round_trip_is_lossless(self):
        """Marshalling to the kernel and back must not move the state."""
        controller = GeneratorCurrentPIController(
            gain_proportional=1.7,
            gain_integral=3.1e5,
            generator_current_bias=0.2 + 0.05j,
            n_delay=9,
        )
        rng = np.random.default_rng(11)
        for _ in range(25):
            controller.update_generator_current(
                1e-3 * complex(rng.normal(), rng.normal()), 3e-10
            )
        before = list(controller._delay_line)
        integral = controller.integral
        state = controller.envelope_scan_state()
        controller.absorb_envelope_scan_state((state[3], state[4], state[5]))
        self.assertEqual(list(controller._delay_line), before)
        self.assertEqual(controller.integral, integral)

    def test_scan_state_hands_out_a_copy(self):
        """
        The buffer must be a copy, not the live state.

        ``_circuit_track_cells`` discards the whole kernel result when a
        cell turns out to be saturated and reruns the span on the exact
        reference path, so a scan whose output is thrown away must leave
        the controller untouched.
        """
        controller = GeneratorCurrentPIController(
            gain_proportional=1.7,
            gain_integral=3.1e5,
            generator_current_bias=0.2 + 0.05j,
            n_delay=4,
        )
        controller.update_generator_current(1e-3 + 0j, 3e-10)
        before = list(controller._delay_line)
        buffer = controller.envelope_scan_state()[3]
        buffer[:] = -99.0  # what a discarded kernel run would scribble
        self.assertEqual(list(controller._delay_line), before)

    def test_delay_line_reports_oldest_first(self):
        """The deque view keeps oldest-to-newest order across a wrap."""
        controller = GeneratorCurrentPIController(
            gain_proportional=0.0,
            gain_integral=0.0,
            generator_current_bias=0.0 + 0.0j,
            n_delay=2,
        )
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            controller.update_generator_current(value + 0j, 1.0)
        self.assertEqual(
            list(controller._delay_line),
            [3.0 + 0j, 4.0 + 0j, 5.0 + 0j],
        )


if __name__ == "__main__":
    unittest.main()
