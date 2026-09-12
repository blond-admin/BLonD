"""
Bit-identity tests for the numba coarse-envelope kernel.

The coarse-grid antenna-voltage recursion in
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
is compiled to a numba host kernel
(:func:`~blond.physics.feedbacks.envelope_kernel.envelope_pi_scan`). The kernel
must reproduce the pure-Python per-cell path **byte-for-byte** (complex128
``np.array_equal``), including the exact exponential propagator,
the PI generator-current controller (delay line, conditional anti-windup,
klystron clamp) and the multi-section backfill/forward segment structure.

These tests drive the extracted cell-loop methods
``_circuit_track_cells_kernel`` and ``_circuit_track_cells_python`` directly and
compare the resulting coarse grids (and the controller state) bit-for-bit.
"""

import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from blond import StaticProfile
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentController,
    GeneratorCurrentPIController,
)

R_OVER_Q = 518.0
Q_L = 1.29e4
T_RF = 1.0e-9
OMEGA_RF = 2.0 * np.pi / T_RF
BIAS = 0.02 + 0.0j


def _make_feedback(
    use_kernel,
    *,
    controller=None,
    voltage_setpoint=None,
    delta_omega=0.0,
    generator_current_bias=BIAS,
):
    """
    Build an isolated timing feedback for direct cell-loop driving.

    Parameters
    ----------
    use_kernel
        Value for ``use_numba_envelope_kernel``.
    controller
        Optional PI controller to attach.
    voltage_setpoint
        Explicit IQ voltage setpoint (avoids needing a parent RF station).
    delta_omega
        Cavity detuning [rad/s].
    generator_current_bias
        Feedforward generator-current bias; zero (with no controller)
        leaves the generator component undriven.

    Returns
    -------
    feedback
        A freshly constructed feedback with ``use_numba_envelope_kernel`` set.
    """
    feedback = IQCavityFeedbackTimingClass(
        profile=Mock(StaticProfile),
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=generator_current_bias,
        n_cavities=1,
        delta_omega=delta_omega,
        controller=controller,
        voltage_setpoint=voltage_setpoint,
    )
    feedback.use_numba_envelope_kernel = use_kernel
    return feedback


def _seed_single_segment(
    feedback,
    n,
    *,
    v_init,
    i_init,
    beam,
    last_val_generator_current=None,
    v_beam_init=0.0 + 0.0j,
    generator_current=BIAS,
):
    """
    Populate the coarse-grid arrays for a single-segment run.

    Parameters
    ----------
    feedback
        The feedback to seed.
    n
        Number of coarse cells.
    v_init
        Carried generator-sourced antenna voltage seeding cell 0
        (``last_val_ant_voltage_gen``).
    i_init
        Carried generator current seeding cell 0.
    beam
        Forward-grid beam current samples (complex array, length ``n``), or
        None for a no-beam segment.
    last_val_generator_current
        Carried generator current (``last_val_generator_current``); defaults to
        ``i_init``. Pass a value off the bias to exercise the backfill-segment
        drive (the reference drives cells >=1 from the reset-bias grid, cell 0
        from the carried value).
    v_beam_init
        Carried beam-sourced antenna voltage seeding cell 0
        (``last_val_ant_voltage_beam``); defaults to zero.
    generator_current
        Value the generator-current grid is pre-filled with; defaults to
        the bias. Zero leaves the generator component undriven.
    """
    dt = T_RF
    feedback._rf_centers = np.arange(1, n + 1) * dt
    feedback._rf_centers_lengths = np.array([n])
    feedback._residual_time_last_rf_centers_calculation = 0.0
    feedback._last_rf_centers_entry = None
    feedback.antenna_voltage_coarse_grid = np.zeros(n, dtype=complex)
    feedback.antenna_voltage_gen_coarse_grid = np.zeros(n, dtype=complex)
    feedback.antenna_voltage_beam_coarse_grid = np.zeros(n, dtype=complex)
    feedback.generator_current_coarse_grid = np.full(
        n, generator_current, dtype=complex
    )
    feedback._last_val_ant_voltage_gen = v_init
    feedback._last_val_ant_voltage_beam = v_beam_init
    feedback._last_val_ant_voltage = v_init + v_beam_init
    feedback._last_val_generator_current = (
        i_init
        if last_val_generator_current is None
        else last_val_generator_current
    )
    if beam is not None:
        feedback.beam_current_forward_coarse_grid = beam.astype(complex)


def _snapshot(feedback):
    """
    Capture the coarse grids and controller state after a run.

    Parameters
    ----------
    feedback
        The feedback that was driven.

    Returns
    -------
    snapshot
        Dict with copies of the antenna-voltage and generator-current grids
        and, when a controller is attached, its integral and delay line.
    """
    snap = {
        "V": feedback.antenna_voltage_coarse_grid.copy(),
        "V_gen": feedback.antenna_voltage_gen_coarse_grid.copy(),
        "V_beam": feedback.antenna_voltage_beam_coarse_grid.copy(),
        "I": feedback.generator_current_coarse_grid.copy(),
    }
    if feedback._controller is not None:
        snap["integral"] = feedback._controller._integral
        snap["delay"] = list(feedback._controller._delay_line)
    return snap


#: Tolerance for comparing the two paths once the klystron clamp has fired.
#: The clamp scales by ``max_output / |I|``, and numpy's complex ``np.abs``
#: and numba's disagree by one or two ULP, so the clamped current differs at
#: the 1e-16 level (measured: 3.1e-16 worst case over a saturating segment,
#: with the antenna voltages and the PI integral still exactly equal). That
#: is four orders below anything physical -- the output power goes as
#: ``|I|**2``, so ~6e-16 relative -- while still catching any real
#: algorithmic divergence, which would be orders larger.
SATURATED_RTOL = 1.0e-12


def _assert_close(test, kernel_snap, python_snap, rtol=SATURATED_RTOL):
    """
    Assert two run snapshots agree to ``rtol``.

    Parameters
    ----------
    test
        The active ``TestCase`` (for assertions).
    kernel_snap
        Snapshot from the kernel path.
    python_snap
        Snapshot from the pure-Python path.
    rtol
        Relative tolerance.
    """
    for key in ("V", "V_gen", "V_beam", "I"):
        np.testing.assert_allclose(
            kernel_snap[key],
            python_snap[key],
            rtol=rtol,
            err_msg=f"{key} differs between kernel and python paths",
        )
    if "integral" in kernel_snap:
        np.testing.assert_allclose(
            kernel_snap["integral"], python_snap["integral"], rtol=rtol
        )


def _assert_bit_identical(test, kernel_snap, python_snap):
    """
    Assert two run snapshots are byte-for-byte identical.

    Parameters
    ----------
    test
        The active ``TestCase`` (for assertions).
    kernel_snap
        Snapshot from the kernel path.
    python_snap
        Snapshot from the pure-Python path.
    """
    test.assertTrue(
        np.array_equal(kernel_snap["V"], python_snap["V"]),
        msg="antenna voltage differs between kernel and python paths",
    )
    test.assertTrue(
        np.array_equal(kernel_snap["V_gen"], python_snap["V_gen"]),
        msg=(
            "generator-sourced voltage differs between kernel and python paths"
        ),
    )
    test.assertTrue(
        np.array_equal(kernel_snap["V_beam"], python_snap["V_beam"]),
        msg=("beam-sourced voltage differs between kernel and python paths"),
    )
    test.assertTrue(
        np.array_equal(kernel_snap["I"], python_snap["I"]),
        msg="generator current differs between kernel and python paths",
    )
    if "integral" in kernel_snap:
        test.assertEqual(kernel_snap["integral"], python_snap["integral"])
        test.assertTrue(
            np.array_equal(
                np.asarray(kernel_snap["delay"]),
                np.asarray(python_snap["delay"]),
            )
        )


class TestEnvelopeKernelBitIdentity(unittest.TestCase):
    """The numba kernel reproduces the Python cell loop byte-for-byte."""

    def _run_single_segment(
        self,
        use_kernel,
        *,
        no_beam,
        controller_kw=None,
        delta_omega=0.0,
        n=64,
        v_init=3.0e7 + 1.0e6j,
        last_val_generator_current=None,
        v_beam_init=0.0 + 0.0j,
        generator_frame_rotation=None,
        kick_frame_rotation=None,
    ):
        """
        Build, seed and drive a single-segment feedback on one path.

        Parameters
        ----------
        use_kernel
            Which path to use.
        no_beam
            Whether the segment carries no beam.
        controller_kw
            Kwargs for the PI controller, or None for constant current.
        delta_omega
            Cavity detuning [rad/s].
        n
            Number of coarse cells.
        v_init
            Carried antenna voltage seeding cell 0.
        last_val_generator_current
            Carried generator current; defaults to the bias. A value off the
            bias exercises the backfill-segment drive divergence.
        v_beam_init
            Carried beam-sourced antenna voltage seeding cell 0.
        generator_frame_rotation
            Per-passage generator frame rotation to install, or None for
            the neutral default.
        kick_frame_rotation
            Per-passage kick frame rotation to install, or None for the
            neutral default.

        Returns
        -------
        snapshot
            The post-run snapshot (see :func:`_snapshot`).
        """
        controller = None
        setpoint = None
        if controller_kw is not None:
            controller = GeneratorCurrentPIController(**controller_kw)
            setpoint = 3.0e7 + 0.0j
        feedback = _make_feedback(
            use_kernel,
            controller=controller,
            voltage_setpoint=setpoint,
            delta_omega=delta_omega,
        )
        rng = np.random.default_rng(1234)
        beam = None
        if not no_beam:
            beam = (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)
            ) * 1e-4
        _seed_single_segment(
            feedback,
            n,
            v_init=v_init,
            i_init=BIAS,
            beam=beam,
            last_val_generator_current=last_val_generator_current,
            v_beam_init=v_beam_init,
        )
        if generator_frame_rotation is not None:
            feedback._generator_frame_rotation = generator_frame_rotation
        if kick_frame_rotation is not None:
            feedback._kick_frame_rotation = kick_frame_rotation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=no_beam,
                start_index=0,
                end_index=n,
            )
        return _snapshot(feedback)

    def _compare_close(self, **kwargs):
        """
        Run one config on both paths and assert agreement to ``rtol``.

        For configurations where the klystron clamp fires; see
        :data:`SATURATED_RTOL`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`_run_single_segment`.
        """
        kernel_snap = self._run_single_segment(True, **kwargs)
        python_snap = self._run_single_segment(False, **kwargs)
        _assert_close(self, kernel_snap, python_snap)

    def _compare(self, **kwargs):
        """
        Run one config on both paths and assert bit identity.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`_run_single_segment`.
        """
        kernel_snap = self._run_single_segment(True, **kwargs)
        python_snap = self._run_single_segment(False, **kwargs)
        _assert_bit_identical(self, kernel_snap, python_snap)

    def test_no_beam_constant_current(self):
        """Backfill-style segment: no beam, no controller."""
        self._compare(no_beam=True)

    def test_forward_constant_current(self):
        """Forward segment with beam but constant generator current."""
        self._compare(no_beam=False)

    def test_forward_pi_no_delay(self):
        """Forward segment driving a PI controller (no loop delay)."""
        self._compare(
            no_beam=False,
            controller_kw={
                "gain_proportional": 1e-9,
                "gain_integral": 5e-4,
                "generator_current_bias": BIAS,
            },
        )

    def test_forward_pi_with_delay(self):
        """PI controller with a two-sample loop delay (delay line)."""
        self._compare(
            no_beam=False,
            controller_kw={
                "gain_proportional": 1e-9,
                "gain_integral": 5e-4,
                "generator_current_bias": BIAS,
                "n_delay": 2,
            },
        )

    def test_forward_pi_saturating(self):
        """PI controller hitting the klystron clamp (anti-windup path).

        Compared to a tolerance rather than bit-for-bit: once the clamp
        fires, the two paths scale by ``max_output / |I|`` computed with
        numpy's and numba's complex ``abs`` respectively, which disagree by
        one or two ULP. See :data:`SATURATED_RTOL`.
        """
        self._compare_close(
            no_beam=False,
            controller_kw={
                "gain_proportional": 1e-6,
                "gain_integral": 1e-1,
                "generator_current_bias": BIAS,
                "n_delay": 1,
                "max_output": 0.05,
            },
        )

    def test_saturating_segment_is_not_deferred_to_python(self):
        """A saturated segment must be committed from the compiled scan.

        The kernel used to flag any cell at or near the klystron limit and
        have the caller discard the whole segment and re-run it cell by
        cell in Python, because numpy's complex ``np.abs`` -- the old
        spelling of the reference clamp -- disagreed with numba's by one
        or two ULP. Both sides now spell the magnitude as
        ``hypot(real, imag)`` (see ``complex_magnitude``), which agrees
        bit-for-bit over the whole double range, so the detour is gone.

        This matters for runtime, not for physics: with a klystron-limited
        preset the generator saturates on every passage, so *every*
        forward segment used to pay the per-cell Python path.
        """
        controller_kw = {
            "gain_proportional": 1e-6,
            "gain_integral": 1e-1,
            "generator_current_bias": BIAS,
            "n_delay": 1,
            "max_output": 0.05,
        }
        original = IQCavityFeedbackTimingClass._circuit_track_cells_python
        calls = []

        def _spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        IQCavityFeedbackTimingClass._circuit_track_cells_python = _spy
        try:
            kernel_snap = self._run_single_segment(
                True, no_beam=False, controller_kw=controller_kw
            )
        finally:
            IQCavityFeedbackTimingClass._circuit_track_cells_python = original

        self.assertEqual(
            len(calls),
            0,
            msg="the kernel still defers a saturated segment to Python",
        )

        python_snap = self._run_single_segment(
            False, no_beam=False, controller_kw=controller_kw
        )
        _assert_close(self, kernel_snap, python_snap)
        # Non-vacuous: the clamp really did fire.
        current = kernel_snap["I"]
        self.assertTrue(
            np.any(
                np.hypot(current.real, current.imag)
                >= controller_kw["max_output"] * (1.0 - 1e-12)
            )
        )

    def test_forward_pi_one_sample_delay(self):
        """PI controller with a one-sample loop delay (below the clamp)."""
        self._compare(
            no_beam=False,
            controller_kw={
                "gain_proportional": 1e-9,
                "gain_integral": 5e-4,
                "generator_current_bias": BIAS,
                "n_delay": 1,
            },
        )

    def test_detuned_pi(self):
        """Non-zero detuning with an active PI controller."""
        self._compare(
            no_beam=False,
            delta_omega=-6.7e3,
            controller_kw={
                "gain_proportional": 1e-9,
                "gain_integral": 5e-4,
                "generator_current_bias": BIAS,
                "n_delay": 2,
            },
        )

    def test_no_beam_carried_generator_current_off_bias(self):
        """
        Backfill segment whose carried generator current is off the bias.

        The reference drives the carried cell 0 from ``last_val_generator_``
        ``current`` but every later cell from the reset-bias generator grid;
        a kernel that held the carried value for all cells would diverge.
        This is the normal state of the first backfill segment of a
        multi-section ring on any turn >= 1 after a PI controller has
        regulated the generator current.
        """
        self._compare(
            no_beam=True,
            last_val_generator_current=0.05 + 0.03j,
        )

    def test_split_components_with_frame_rotations(self):
        """
        Both carried components plus non-unit frame rotations.

        The live multi-section / RF-offset condition: the generator and
        beam components carry distinct nonzero state and the per-passage
        rotations are away from unity, so the kernel's composition
        ``V_beam + V_gen * generator_frame_rotation`` must reproduce the
        reference multiply bit-for-bit.
        """
        self._compare(
            no_beam=False,
            v_beam_init=-2.0e6 + 4.0e5j,
            generator_frame_rotation=np.exp(-0.37j),
            kick_frame_rotation=np.exp(0.21j),
        )

    def test_split_components_pi_with_frame_rotations(self):
        """PI regulation of the kick-frame sum under non-unit rotations."""
        self._compare(
            no_beam=False,
            v_beam_init=-2.0e6 + 4.0e5j,
            generator_frame_rotation=np.exp(-0.37j),
            kick_frame_rotation=np.exp(0.21j),
            controller_kw={
                "gain_proportional": 1e-9,
                "gain_integral": 5e-4,
                "generator_current_bias": BIAS,
                "n_delay": 2,
            },
        )

    def _run_multi_section(
        self,
        use_kernel,
        *,
        last_val_generator_current=BIAS,
        backfill_phases=None,
        forward_phase=0.0,
    ):
        """
        Drive a backfill + forward two-segment layout on one path.

        Parameters
        ----------
        use_kernel
            Which path to use.
        last_val_generator_current
            Carried generator current seeding the run (off the bias exercises
            the backfill-segment drive divergence).
        backfill_phases
            Accumulated phase [rad] of each of the 20 backfill cells,
            installed as their per-cell generator and kick frame rotations;
            None installs none, so every cell takes the per-passage scalars.
        forward_phase
            Accumulated phase [rad] of the forward span, installed as the
            per-passage rotations; only used with ``backfill_phases``.

        Returns
        -------
        snapshot
            The post-run snapshot.
        """
        n_backfill, n_frwrd = 20, 44
        n = n_backfill + n_frwrd
        controller = GeneratorCurrentPIController(
            gain_proportional=1e-9,
            gain_integral=5e-4,
            generator_current_bias=BIAS,
            n_delay=2,
        )
        feedback = _make_feedback(
            use_kernel, controller=controller, voltage_setpoint=3.0e7 + 0.0j
        )
        dt = T_RF
        feedback._rf_centers = np.arange(1, n + 1) * dt
        feedback._rf_centers_lengths = np.array([n_backfill, n_frwrd])
        feedback._residual_time_last_rf_centers_calculation = 0.0
        feedback._last_rf_centers_entry = None
        feedback.antenna_voltage_coarse_grid = np.zeros(n, dtype=complex)
        feedback.antenna_voltage_gen_coarse_grid = np.zeros(n, dtype=complex)
        feedback.antenna_voltage_beam_coarse_grid = np.zeros(n, dtype=complex)
        feedback.generator_current_coarse_grid = np.full(
            n, BIAS, dtype=complex
        )
        feedback._last_val_ant_voltage_gen = 3.0e7 + 1.0e6j
        feedback._last_val_ant_voltage_beam = -1.5e6 + 2.0e5j
        feedback._last_val_ant_voltage = (
            feedback._last_val_ant_voltage_gen
            + feedback._last_val_ant_voltage_beam
        )
        feedback._last_val_generator_current = last_val_generator_current
        rng = np.random.default_rng(77)
        feedback.beam_current_forward_coarse_grid = (
            (rng.standard_normal(n_frwrd) + 1j * rng.standard_normal(n_frwrd))
            * 1e-4
        ).astype(complex)
        if backfill_phases is not None:
            self.assertEqual(len(backfill_phases), n_backfill)
            # As ``_update_frame_rotations`` builds them: the generator
            # component turns by minus the phase, the kick frame by plus it.
            feedback._backfill_generator_frame_rotations = np.exp(
                -1j * np.asarray(backfill_phases)
            )
            feedback._backfill_kick_frame_rotations = np.exp(
                1j * np.asarray(backfill_phases)
            )
            feedback._generator_frame_rotation = complex(
                np.exp(-1j * forward_phase)
            )
            feedback._kick_frame_rotation = complex(np.exp(1j * forward_phase))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Backfill (no-beam) segment, then the forward (beam+PI) segment.
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=True,
                start_index=0,
                end_index=n_backfill,
            )
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=n_backfill,
                end_index=n,
            )
        return _snapshot(feedback)

    def test_multi_section_backfill_then_forward(self):
        """Two-segment (backfill + forward) run is bit-identical."""
        kernel_snap = self._run_multi_section(True)
        python_snap = self._run_multi_section(False)
        _assert_bit_identical(self, kernel_snap, python_snap)

    def test_multi_section_carried_state_off_trivial(self):
        """
        Two-segment run with an off-bias carried generator current.

        Reproduces the live multi-section turn >= 1 condition end-to-end:
        the first backfill segment carries a regulated (off-bias)
        generator current, and the divergence must still not appear.
        """
        kernel_snap = self._run_multi_section(
            True,
            last_val_generator_current=0.05 + 0.03j,
        )
        python_snap = self._run_multi_section(
            False,
            last_val_generator_current=0.05 + 0.03j,
        )
        _assert_bit_identical(self, kernel_snap, python_snap)

    def test_multi_section_per_cell_backfill_rotations(self):
        """
        Two-segment run with its own frame rotation on every backfill cell.

        The ramped multi-section condition: each backfill cell is composed
        and regulated with the phase accumulated up to it, rising towards
        the passage's, which the forward span then takes. The kernel gets
        the rotations as per-cell arrays and the reference reads them cell
        by cell; the two must agree bit-for-bit.
        """
        rotations = {
            "backfill_phases": np.linspace(0.0, 0.28, 20),
            "forward_phase": 0.3,
        }
        kernel_snap = self._run_multi_section(True, **rotations)
        python_snap = self._run_multi_section(False, **rotations)
        _assert_bit_identical(self, kernel_snap, python_snap)
        # Non-vacuous: the per-cell kick rotations reach the loop, so the
        # backfill current differs from the passage's rotation on every cell.
        passage_snap = self._run_multi_section(
            False, backfill_phases=np.full(20, 0.3), forward_phase=0.3
        )
        self.assertFalse(
            np.array_equal(python_snap["I"][:20], passage_snap["I"][:20])
        )


class TestUndrivenGeneratorComponentNeedsNoGate(unittest.TestCase):
    """Nothing driving the generator gives exact zeros, not a branch.

    The source split used to carry a ``_generator_active`` gate: while
    nothing could source the generator component -- no controller, zero
    bias, no carried generator current and no carried generator-sourced
    voltage (which an initial or pre-fill voltage seeds) -- its update,
    the kernel's write of its grid and every composition multiply were
    skipped, so that an undriven feedback stayed bit-identical to the
    former single-state recursion.

    It was never a physics switch. With nothing driving it the component
    is identically zero, and both ``0 * rotation`` and ``x + 0`` are exact
    in IEEE double, so the identity the gate protected is a property of
    the arithmetic. These tests pin that property -- with a deliberately
    NON-unity frame rotation, which is what the skipped multiply used --
    and pin the gate itself gone.
    """

    #: A rotation far from unity: if the generator component were not
    #: exactly zero, composing it would be plainly visible.
    FRAME_ROTATION = np.exp(-0.7j)

    def _run_undriven(self, use_kernel, n=64):
        """
        Track one beam-loaded segment with nothing driving the generator.

        Parameters
        ----------
        use_kernel
            Which path to run.
        n
            Number of coarse cells.

        Returns
        -------
        snapshot
            The post-run snapshot (see :func:`_snapshot`).
        """
        feedback = _make_feedback(
            use_kernel, generator_current_bias=0.0 + 0.0j
        )
        rng = np.random.default_rng(4321)
        beam = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-4
        _seed_single_segment(
            feedback,
            n,
            v_init=0.0 + 0.0j,
            i_init=0.0 + 0.0j,
            beam=beam,
            generator_current=0.0 + 0.0j,
            v_beam_init=3.0e7 + 1.0e6j,
        )
        feedback._generator_frame_rotation = self.FRAME_ROTATION
        feedback._kick_frame_rotation = np.conj(self.FRAME_ROTATION)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=0,
                end_index=n,
            )
        return _snapshot(feedback)

    def test_the_feedback_has_no_generator_active_gate(self):
        """The gate is gone, on both paths and before any tracking."""
        for use_kernel in (True, False):
            with self.subTest(use_kernel=use_kernel):
                feedback = _make_feedback(
                    use_kernel, generator_current_bias=0.0 + 0.0j
                )
                self.assertFalse(hasattr(feedback, "_generator_active"))

    def test_undriven_generator_component_is_exactly_zero(self):
        """Zero drive from a zero seed propagates to exact zeros."""
        for use_kernel in (True, False):
            with self.subTest(use_kernel=use_kernel):
                snapshot = self._run_undriven(use_kernel)
                np.testing.assert_array_equal(
                    snapshot["V_gen"],
                    np.zeros(len(snapshot["V_gen"]), dtype=complex),
                )

    def test_undriven_sum_is_exactly_the_beam_component(self):
        """The composed sum is the beam component, bit for bit.

        Composing adds ``0 * FRAME_ROTATION``, which is exactly zero, so
        the sum is the beam component even though the rotation is not
        unity -- the equality the retired gate used to assign by branch.
        """
        for use_kernel in (True, False):
            with self.subTest(use_kernel=use_kernel):
                snapshot = self._run_undriven(use_kernel)
                self.assertTrue(
                    np.array_equal(snapshot["V"], snapshot["V_beam"]),
                    msg="composed sum is not the beam component",
                )
                self.assertNotEqual(
                    complex(snapshot["V_beam"][-1]), 0.0 + 0.0j
                )

    def test_undriven_paths_stay_bit_identical(self):
        """The kernel and the reference agree on the undriven segment."""
        _assert_bit_identical(
            self, self._run_undriven(True), self._run_undriven(False)
        )


class TestDegenerateCoarseSteps(unittest.TestCase):
    """
    First-cell seeding, coincident points and empty segments.

    The per-cell reference loop (``_circuit_track_cells_python``) and its
    vectorised twin (``_coarse_step_sizes``) share the first-cell special
    cases; degenerate (coincident / zero-step) grids must defer the kernel
    path to the reference loop, the only one that duplicates the previous
    cell into the coincident one.
    """

    V_INIT = 3.0e7 + 1.0e6j

    def _coincident_grid_feedback(self, use_kernel, step_offset=0.0):
        """
        Feedback seeded with a grid whose third centre repeats the second.

        Parameters
        ----------
        use_kernel
            Value for ``use_numba_envelope_kernel``.
        step_offset
            Offset [s] added to the repeated centre; a tiny negative value
            makes the degenerate step a few-ULP negative one instead of an
            exact zero.

        Returns
        -------
        feedback
            The seeded feedback (4 cells, no beam).
        """
        feedback = _make_feedback(use_kernel)
        _seed_single_segment(
            feedback, 4, v_init=self.V_INIT, i_init=BIAS, beam=None
        )
        feedback._rf_centers = np.array(
            [1.0 * T_RF, 2.0 * T_RF, 2.0 * T_RF + step_offset, 3.0 * T_RF]
        )
        return feedback

    def test_first_turn_single_cell_segment_uses_own_period_step(self):
        """
        A one-cell first-ever segment steps by its own coarse period.

        With a single centre in the segment there is no next centre to take
        the step proxy from, so the reference loop falls back to this
        segment's own coarse step ``n * t_rf`` at ``omega_input`` -- not the
        (meaningless) local time of the centre.
        """
        feedback = _make_feedback(False)
        _seed_single_segment(
            feedback, 1, v_init=self.V_INIT, i_init=BIAS, beam=None
        )
        # Park the only centre OFF the coarse period so a wrong local-time
        # step would produce a distinguishable voltage.
        feedback._rf_centers = np.array([0.3 * T_RF])
        feedback._circuit_track_cells_python(
            OMEGA_RF, no_beam=True, start_index=0, end_index=1
        )
        step = 2.0 * np.pi / OMEGA_RF  # n_rf_periods_per_coarse_grid == 1
        expected = feedback._advance_coarse_voltage(
            v_prev=self.V_INIT,
            generator_current=BIAS,
            beam_current=0.0,
            omega_times_dt=OMEGA_RF * step,
            relative_detuning=0.0,
        )
        wrong = feedback._advance_coarse_voltage(
            v_prev=self.V_INIT,
            generator_current=BIAS,
            beam_current=0.0,
            omega_times_dt=OMEGA_RF * 0.3 * T_RF,
            relative_detuning=0.0,
        )
        self.assertEqual(feedback.antenna_voltage_coarse_grid[0], expected)
        self.assertNotEqual(
            feedback.antenna_voltage_coarse_grid[0],
            wrong,
            msg="the period proxy is indistinguishable from the local time",
        )

    def test_single_cell_segment_vectorised_step_matches_the_reference(self):
        """The vectorised twin uses the same one-cell period proxy."""
        feedback = _make_feedback(True)
        _seed_single_segment(
            feedback, 1, v_init=self.V_INIT, i_init=BIAS, beam=None
        )
        feedback._rf_centers = np.array([0.3 * T_RF])
        delta_t = feedback._coarse_step_sizes(OMEGA_RF, 0, 1)
        np.testing.assert_array_equal(delta_t, [2.0 * np.pi / OMEGA_RF])

    def test_coincident_points_warn_and_duplicate_the_cell(self):
        """
        Two identical consecutive centres warn and duplicate the cell.

        A coincident centre carries zero elapsed time, so the correct
        antenna voltage there is the previous cell's, ``V(t + 0) = V(t)``.
        The cell must hold that value (not the zeros prefill), so the
        following cell advances from the real carried voltage.
        """
        feedback = self._coincident_grid_feedback(False)
        with self.assertWarnsRegex(
            UserWarning, "double taking of rf_centers value, duplicating"
        ):
            feedback._circuit_track_cells_python(
                OMEGA_RF, no_beam=True, start_index=0, end_index=4
            )
        # The coincident cell holds the previous cell's state...
        self.assertEqual(
            feedback.antenna_voltage_coarse_grid[2],
            feedback.antenna_voltage_coarse_grid[1],
        )
        self.assertNotEqual(feedback.antenna_voltage_coarse_grid[2], 0.0)
        self.assertEqual(
            feedback.generator_current_coarse_grid[2],
            feedback.generator_current_coarse_grid[1],
        )
        # ...and the next cell advances from it, not from zero.
        self.assertEqual(
            feedback.antenna_voltage_coarse_grid[3],
            feedback._advance_coarse_voltage(
                v_prev=feedback.antenna_voltage_coarse_grid[1],
                generator_current=feedback.generator_current_coarse_grid[2],
                beam_current=0,
                omega_times_dt=OMEGA_RF * T_RF,
                relative_detuning=0.0,
            ),
        )
        self.assertTrue(
            np.all(np.isfinite(feedback.antenna_voltage_coarse_grid))
        )

    def test_few_ulp_negative_step_is_clamped_not_asserted(self):
        """
        A few-ULP negative step is floating-point noise, not an error.

        The reference loop clamps it to zero (then handled as a coincident
        point) instead of tripping the hard ordering assertion.
        """
        # In (-1e-9 * rf_period, 0): well inside the clamp band.
        feedback = self._coincident_grid_feedback(
            False, step_offset=-1e-10 * T_RF
        )
        with self.assertWarnsRegex(
            UserWarning, "double taking of rf_centers value, duplicating"
        ):
            # Must NOT raise AssertionError on the negative step.
            feedback._circuit_track_cells_python(
                OMEGA_RF, no_beam=True, start_index=0, end_index=4
            )
        self.assertEqual(
            feedback.antenna_voltage_coarse_grid[2],
            feedback.antenna_voltage_coarse_grid[1],
        )
        self.assertNotEqual(feedback.antenna_voltage_coarse_grid[3], 0.0)

    def test_degenerate_segment_defers_the_kernel_to_the_reference(self):
        """
        A zero coarse step makes the kernel take the reference path.

        ``_coarse_step_sizes`` reports the degenerate segment with ``None``
        and ``_circuit_track_cells_kernel`` then runs the pure-Python loop,
        whose duplicate-and-warn behaviour (the warning below) and result
        must appear bit-for-bit.
        """
        kernel_feedback = self._coincident_grid_feedback(True)
        self.assertIsNone(kernel_feedback._coarse_step_sizes(OMEGA_RF, 0, 4))
        with self.assertWarnsRegex(
            UserWarning, "double taking of rf_centers value, duplicating"
        ):
            kernel_feedback._circuit_track_cells_kernel(
                OMEGA_RF, no_beam=True, start_index=0, end_index=4
            )

        python_feedback = self._coincident_grid_feedback(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            python_feedback._circuit_track_cells_python(
                OMEGA_RF, no_beam=True, start_index=0, end_index=4
            )
        self.assertTrue(
            np.array_equal(
                kernel_feedback.antenna_voltage_coarse_grid,
                python_feedback.antenna_voltage_coarse_grid,
            ),
            msg="kernel fallback differs from the reference path",
        )

    def test_empty_segment_is_a_no_op_on_the_kernel_path(self):
        """``start_index == end_index`` leaves the grids untouched."""
        feedback = _make_feedback(True)
        _seed_single_segment(
            feedback, 4, v_init=self.V_INIT, i_init=BIAS, beam=None
        )
        feedback.antenna_voltage_coarse_grid[:] = 7.0 + 3.0j  # sentinel
        voltage_before = feedback.antenna_voltage_coarse_grid.copy()
        current_before = feedback.generator_current_coarse_grid.copy()
        feedback._circuit_track_cells_kernel(
            OMEGA_RF, no_beam=True, start_index=2, end_index=2
        )
        np.testing.assert_array_equal(
            feedback.antenna_voltage_coarse_grid, voltage_before
        )
        np.testing.assert_array_equal(
            feedback.generator_current_coarse_grid, current_before
        )


class _ProportionalOnlyController(GeneratorCurrentController):
    """
    Minimal non-PI controller implementing only the abstract interface.

    Carries no gains, delay line or error integral, so a feedback that reaches
    for :class:`GeneratorCurrentPIController` internals cannot drive it.
    """

    def __init__(self, gain: float, bias: complex):
        self.gain = gain
        self.bias = bias
        self.n_updates = 0

    def update_generator_current(
        self, error: complex, delta_t: float
    ) -> complex:
        """
        Map the error to a current with a pure proportional law.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample [V].
        delta_t
            Time step of this sample [s]; unused by this law.

        Returns
        -------
        generator_current
            The generator-current command for this sample [A].
        """
        self.n_updates += 1
        return self.bias + self.gain * error


class TestControllerAbstractionContract(unittest.TestCase):
    """Any ``GeneratorCurrentController`` drives the default (kernel) path."""

    N_CELLS = 32

    def _run(self, use_kernel):
        """
        Drive one forward segment with a non-PI controller attached.

        Parameters
        ----------
        use_kernel
            Which path to select.

        Returns
        -------
        voltage, current, n_updates
            The coarse grids and the controller's own update count.
        """
        n = self.N_CELLS
        controller = _ProportionalOnlyController(1.0e-9, BIAS)
        feedback = _make_feedback(
            use_kernel, controller=controller, voltage_setpoint=3.0e7 + 0.0j
        )
        rng = np.random.default_rng(5)
        beam = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-4
        _seed_single_segment(
            feedback, n, v_init=3.0e7 + 1.0e6j, i_init=BIAS, beam=beam
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=0,
                end_index=n,
            )
        return (
            feedback.antenna_voltage_coarse_grid.copy(),
            feedback.generator_current_coarse_grid.copy(),
            controller.n_updates,
        )

    def test_non_pi_controller_runs_on_the_default_path(self):
        """The default path honours the interface, not a concrete PI class.

        The abstract controller promises the feedback 'does not need to know
        the control law'. Reaching for PI-only attributes on the compiled path
        broke every other implementation of the interface.
        """
        voltage, current, n_updates = self._run(True)
        self.assertEqual(n_updates, self.N_CELLS)
        self.assertTrue(np.all(np.isfinite(voltage)))
        self.assertTrue(np.all(np.isfinite(current)))

    def test_non_pi_controller_matches_the_python_path(self):
        """A controller without a compiled scan still matches the reference."""
        kernel_voltage, kernel_current, _ = self._run(True)
        python_voltage, python_current, _ = self._run(False)
        self.assertTrue(
            np.array_equal(kernel_voltage, python_voltage),
            msg="antenna voltage differs between kernel and python paths",
        )
        self.assertTrue(
            np.array_equal(kernel_current, python_current),
            msg="generator current differs between kernel and python paths",
        )


if __name__ == "__main__":
    unittest.main()
