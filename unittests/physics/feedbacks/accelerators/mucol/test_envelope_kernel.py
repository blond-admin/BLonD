"""
Bit-identity tests for the numba coarse-envelope kernel.

The coarse-grid antenna-voltage recursion in
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
is compiled to a numba host kernel
(:func:`~blond.physics.feedbacks.envelope_kernel.envelope_pi_scan`). The kernel
must reproduce the pure-Python per-cell path **byte-for-byte** (complex128
``np.array_equal``), including the forward-Euler and exponential propagators,
the PI generator-current controller (delay line, conditional anti-windup,
klystron clamp) and the multi-section reverse/forward segment structure.

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
    exponential=False,
    delta_omega=0.0,
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
    exponential
        Select the exponential coarse solver.
    delta_omega
        Cavity detuning [rad/s].

    Returns
    -------
    feedback
        A freshly constructed feedback with ``use_numba_envelope_kernel`` set.
    """
    feedback = IQCavityFeedbackTimingClass(
        profile=Mock(StaticProfile),
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=BIAS,
        n_cavities=1,
        delta_omega=delta_omega,
        exponential_coarse_solver=exponential,
        controller=controller,
        voltage_setpoint=voltage_setpoint,
    )
    feedback.use_numba_envelope_kernel = use_kernel
    return feedback


def _seed_single_segment(feedback, n, *, v_init, i_init, beam):
    """
    Populate the coarse-grid arrays for a single-segment run.

    Parameters
    ----------
    feedback
        The feedback to seed.
    n
        Number of coarse cells.
    v_init
        Carried antenna voltage seeding cell 0 (``last_val_ant_voltage``).
    i_init
        Carried generator current seeding cell 0.
    beam
        Forward-grid beam current samples (complex array, length ``n``), or
        None for a no-beam segment.
    """
    dt = T_RF
    feedback.rf_centers = np.arange(1, n + 1) * dt
    feedback.rf_centers_lengths = np.array([n])
    feedback.residual_time_last_rf_centers_calculation = 0.0
    feedback.last_rf_centers_entry = None
    feedback.antenna_voltage_coarse_grid = np.zeros(n, dtype=complex)
    feedback.generator_current_coarse_grid = np.full(n, BIAS, dtype=complex)
    feedback.last_val_ant_voltage = v_init
    feedback.last_val_generator_current = i_init
    feedback.last_val_beam_current = 0.0 + 0.0j
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
        "I": feedback.generator_current_coarse_grid.copy(),
    }
    if feedback._controller is not None:
        snap["integral"] = feedback._controller._integral
        snap["delay"] = list(feedback._controller._delay_line)
    return snap


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
        exponential=False,
        delta_omega=0.0,
        n=64,
        v_init=3.0e7 + 1.0e6j,
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
        exponential
            Select the exponential coarse solver.
        delta_omega
            Cavity detuning [rad/s].
        n
            Number of coarse cells.
        v_init
            Carried antenna voltage seeding cell 0.

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
            exponential=exponential,
            delta_omega=delta_omega,
        )
        rng = np.random.default_rng(1234)
        beam = None
        if not no_beam:
            beam = (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)
            ) * 1e-4
        _seed_single_segment(
            feedback, n, v_init=v_init, i_init=BIAS, beam=beam
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=no_beam,
                start_index=0,
                end_index=n,
            )
        return _snapshot(feedback)

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
        """Reverse-style segment: no beam, no controller."""
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
        """PI controller hitting the klystron clamp (anti-windup path)."""
        self._compare(
            no_beam=False,
            controller_kw={
                "gain_proportional": 1e-6,
                "gain_integral": 1e-1,
                "generator_current_bias": BIAS,
                "n_delay": 1,
                "max_output": 0.05,
            },
        )

    def test_exponential_solver_pi(self):
        """Exponential propagator with an active PI controller."""
        self._compare(
            no_beam=False,
            exponential=True,
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

    def _run_multi_section(self, use_kernel):
        """
        Drive a reverse + forward two-segment layout on one path.

        Parameters
        ----------
        use_kernel
            Which path to use.

        Returns
        -------
        snapshot
            The post-run snapshot.
        """
        n_rev, n_frwrd = 20, 44
        n = n_rev + n_frwrd
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
        feedback.rf_centers = np.arange(1, n + 1) * dt
        feedback.rf_centers_lengths = np.array([n_rev, n_frwrd])
        feedback.residual_time_last_rf_centers_calculation = 0.0
        feedback.last_rf_centers_entry = None
        feedback.antenna_voltage_coarse_grid = np.zeros(n, dtype=complex)
        feedback.generator_current_coarse_grid = np.full(
            n, BIAS, dtype=complex
        )
        feedback.last_val_ant_voltage = 3.0e7 + 1.0e6j
        feedback.last_val_generator_current = BIAS
        feedback.last_val_beam_current = 0.0 + 0.0j
        rng = np.random.default_rng(77)
        feedback.beam_current_forward_coarse_grid = (
            (rng.standard_normal(n_frwrd) + 1j * rng.standard_normal(n_frwrd))
            * 1e-4
        ).astype(complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Reverse (no-beam) segment, then the forward (beam + PI) segment.
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=True,
                start_index=0,
                end_index=n_rev,
            )
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=n_rev,
                end_index=n,
            )
        return _snapshot(feedback)

    def test_multi_section_reverse_then_forward(self):
        """Two-segment (reverse + forward) run is bit-identical."""
        kernel_snap = self._run_multi_section(True)
        python_snap = self._run_multi_section(False)
        _assert_bit_identical(self, kernel_snap, python_snap)


if __name__ == "__main__":
    unittest.main()
