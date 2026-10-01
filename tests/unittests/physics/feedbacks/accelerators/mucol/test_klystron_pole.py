# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests for the klystron pole of the generator-current controllers.

A klystron does not follow its drive instantly: over its bandwidth it acts
as a low-pass on the RF envelope. Both controllers can carry that as one
real pole, ``klystron_time_constant``, between the clamped command and the
generator current that drives the cavity. These tests pin the conversion
from a stated bandwidth, the step response the cavity sees, that the
compiled scans and the Python reference agree, that the state carries
across spans, and that the pole leaves the klystron limit intact.
"""

import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from blond import StaticProfile
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackCoarseGrid,
)
from blond.physics.feedbacks.envelope_kernel import envelope_open_loop_scan
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPController,
    GeneratorCurrentPIController,
    klystron_time_constant_from_bandwidth,
)

R_OVER_Q = 518.0
Q_L = 1.29e4
T_RF = 1.0e-9
OMEGA_RF = 2.0 * np.pi / T_RF
BIAS = 0.02 + 0.0j
SETPOINT = 3.0e7 + 0.0j
TIME_CONSTANT = 8.0 * T_RF
#: Kernel against reference: numba's and numpy's ``exp`` and complex
#: ``abs`` may differ by an ULP, nothing more.
RTOL = 1.0e-12


def _pi(**kwargs):
    settings = {
        "gain_proportional": 1.0e-9,
        "gain_integral": 5.0e-4,
        "generator_current_bias": BIAS,
        "n_delay": 2,
    }
    settings.update(kwargs)
    return GeneratorCurrentPIController(**settings)


def _p(**kwargs):
    settings = {
        "gain_proportional": 1.0e-9,
        "generator_current_bias": BIAS,
        "n_delay": 2,
    }
    settings.update(kwargs)
    return GeneratorCurrentPController(**settings)


def _feedback(controller, use_kernel, n, *, interval=1, carried=BIAS):
    """
    A single-segment feedback seeded for direct cell-loop driving.

    Parameters
    ----------
    controller
        The controller to attach.
    use_kernel
        Whether the compiled scan runs the segment.
    n
        Number of coarse cells, one RF period each.
    interval
        Coarse cells per controller update.
    carried
        Generator current carried into the segment: the klystron output
        that drives its first cell.

    Returns
    -------
    feedback
        The seeded feedback.
    """
    feedback = IQCavityFeedbackCoarseGrid(
        profile=Mock(StaticProfile),
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=BIAS,
        n_cavities=1,
        controller=controller,
        voltage_setpoint=SETPOINT,
        controller_update_interval=interval,
    )
    feedback.use_numba_envelope_kernel = use_kernel
    feedback._rf_centers = np.arange(1, n + 1) * T_RF
    feedback._rf_centers_lengths = np.array([n])
    feedback._residual_time_last_rf_centers_calculation = 0.0
    feedback._last_rf_centers_entry = None
    feedback.antenna_voltage_coarse_grid = np.zeros(n, dtype=complex)
    feedback.antenna_voltage_gen_coarse_grid = np.zeros(n, dtype=complex)
    feedback.antenna_voltage_beam_coarse_grid = np.zeros(n, dtype=complex)
    feedback.generator_current_coarse_grid = np.full(n, BIAS, dtype=complex)
    feedback._last_val_ant_voltage_gen = 3.0e7 + 1.0e6j
    feedback._last_val_ant_voltage_beam = 0.0 + 0.0j
    feedback._last_val_ant_voltage = 3.0e7 + 1.0e6j
    feedback._last_val_generator_current = carried
    rng = np.random.default_rng(21)
    feedback.beam_current_forward_coarse_grid = (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ) * 1.0e-4
    return feedback


def _track(feedback, spans, between=None):
    """
    Track the given ``(start, end)`` spans in order.

    Parameters
    ----------
    feedback
        The seeded feedback.
    spans
        Index ranges of the coarse grid, in tracking order.
    between
        Optional callable run on the feedback before every span but the
        first.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, (start, end) in enumerate(spans):
            if index > 0 and between is not None:
                between(feedback)
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=start,
                end_index=end,
            )


class TestKlystronTimeConstantFromBandwidth(unittest.TestCase):
    """A stated bandwidth becomes the time constant of one real pole."""

    def test_the_pole_loses_the_stated_attenuation_at_the_band_edge(self):
        for bandwidth, attenuation_db in ((5.0e6, 1.0), (2.0e6, 3.0)):
            with self.subTest(bandwidth=bandwidth, db=attenuation_db):
                tau = klystron_time_constant_from_bandwidth(
                    bandwidth, attenuation_db=attenuation_db
                )
                # The band is the full width about the carrier, so its
                # edge sits half of it away.
                omega_edge = 2.0 * np.pi * bandwidth / 2.0
                gain_db = -10.0 * np.log10(1.0 + (omega_edge * tau) ** 2)
                self.assertAlmostEqual(gain_db, -attenuation_db, places=12)

    def test_one_db_at_five_megahertz_is_32_nanoseconds(self):
        self.assertAlmostEqual(
            klystron_time_constant_from_bandwidth(5.0e6) * 1e9, 32.39, 2
        )

    def test_a_non_positive_bandwidth_is_refused(self):
        for bandwidth in (0.0, -1.0e6):
            with self.subTest(bandwidth=bandwidth):
                with self.assertRaises(ValueError):
                    klystron_time_constant_from_bandwidth(bandwidth)


class TestTheControllersCarryThePole(unittest.TestCase):
    """The pole is a constructor knob of both laws, off by default."""

    def test_no_pole_by_default(self):
        for controller in (_pi(), _p()):
            with self.subTest(law=type(controller).__name__):
                self.assertEqual(controller.klystron_time_constant, 0.0)

    def test_a_negative_time_constant_is_refused(self):
        for make in (_pi, _p):
            with self.subTest(law=make.__name__):
                with self.assertRaises(ValueError):
                    make(klystron_time_constant=-1.0e-9)

    def test_the_held_command_round_trips_through_the_scan_state(self):
        """What the compiled scan hands back is what the next one gets."""
        for make in (_pi, _p):
            with self.subTest(law=make.__name__):
                controller = make(klystron_time_constant=TIME_CONSTANT)
                state = controller.envelope_scan_state()
                self.assertEqual(state[-2], TIME_CONSTANT)
                self.assertEqual(state[-1], BIAS)
                returned = list(
                    (state[3], state[4], state[5])
                    if make is _pi
                    else (state[2], state[3])
                )
                controller.absorb_envelope_scan_state(
                    (*returned, 0.03 - 0.01j)
                )
                self.assertEqual(
                    controller.envelope_scan_state()[-1], 0.03 - 0.01j
                )


class TestTheCavitySeesTheKlystronOutput(unittest.TestCase):
    """The generator grid carries the pole's step response, not the command.

    With both gains at zero every law commands its bias on every sample.
    A segment that starts from a different carried current therefore
    holds one step of the command, and the grid must relax to it as
    ``I_0 + (I_carried - I_0) exp(-(c + 1) dt / tau)`` -- between the
    controller's samples too, since the klystron does not wait for them.
    """

    N_CELLS = 40
    CARRIED = 0.035 - 0.012j

    def _relaxed(self, make, use_kernel, interval):
        controller = make(
            gain_proportional=0.0,
            klystron_time_constant=TIME_CONSTANT,
            **({"gain_integral": 0.0} if make is _pi else {}),
        )
        feedback = _feedback(
            controller,
            use_kernel,
            self.N_CELLS,
            interval=interval,
            carried=self.CARRIED,
        )
        _track(feedback, [(0, self.N_CELLS)])
        return feedback

    def test_the_grid_is_the_single_pole_step_response(self):
        cells = np.arange(1, self.N_CELLS + 1)
        expected = BIAS + (self.CARRIED - BIAS) * np.exp(
            -cells * T_RF / TIME_CONSTANT
        )
        for make in (_pi, _p):
            for use_kernel in (True, False):
                for interval in (1, 4):
                    with self.subTest(
                        law=make.__name__,
                        use_kernel=use_kernel,
                        interval=interval,
                    ):
                        feedback = self._relaxed(make, use_kernel, interval)
                        np.testing.assert_allclose(
                            feedback.generator_current_coarse_grid,
                            expected,
                            rtol=RTOL,
                        )

    def test_the_cavity_is_driven_by_the_grid(self):
        """The voltage is the open-loop cavity driven by the pole's output."""
        feedback = self._relaxed(_pi, True, 1)
        n = self.N_CELLS
        voltage_gen = np.empty(n, dtype=np.complex128)
        voltage_beam = np.empty(n, dtype=np.complex128)
        voltage = np.empty(n, dtype=np.complex128)
        omega_times_dt = np.full(n, OMEGA_RF * T_RF)
        multiplier, weight = feedback._segment_step_multipliers(
            float(omega_times_dt[0]), float(omega_times_dt[-1]), n, 0.0
        )
        rotations = feedback._calculate_coarse_frame_rotations(0, n)
        envelope_open_loop_scan(
            multiplier,
            weight,
            omega_times_dt,
            feedback.beam_current_forward_coarse_grid.astype(np.complex128),
            voltage_gen,
            voltage_beam,
            voltage,
            feedback.generator_current_coarse_grid.astype(np.complex128),
            3.0e7 + 1.0e6j,
            0.0 + 0.0j,
            self.CARRIED,
            R_OVER_Q,
            rotations[0],
            rotations[3],
        )
        np.testing.assert_allclose(
            feedback.antenna_voltage_gen_coarse_grid, voltage_gen, rtol=RTOL
        )


class TestThePoleInTheClosedLoop(unittest.TestCase):
    """A regulating loop through the pole, on both paths and across spans."""

    N_CELLS = 48

    def _run(self, make, use_kernel, spans, between=None, **kwargs):
        controller = make(klystron_time_constant=TIME_CONSTANT, **kwargs)
        feedback = _feedback(controller, use_kernel, self.N_CELLS, interval=4)
        _track(feedback, spans, between)
        return feedback

    def test_the_kernel_matches_the_reference_path(self):
        cases = (
            (_pi, {}),
            (_pi, {"max_output": 0.02001}),
            (_pi, {"max_output": 0.02001, "anti_windup": "directional"}),
            (_p, {}),
            (_p, {"max_output": 0.02001}),
        )
        spans = [(0, self.N_CELLS)]
        for make, kwargs in cases:
            with self.subTest(law=make.__name__, **kwargs):
                kernel = self._run(make, True, spans, **kwargs)
                python = self._run(make, False, spans, **kwargs)
                for name in (
                    "generator_current_coarse_grid",
                    "antenna_voltage_coarse_grid",
                ):
                    np.testing.assert_allclose(
                        getattr(kernel, name),
                        getattr(python, name),
                        rtol=RTOL,
                        err_msg=name,
                    )

    def test_the_held_command_carries_across_spans(self):
        """The compiled scan hands the held command on to the next span.

        The command the klystron is chasing is controller state. The
        reference path keeps it on the live controller; the compiled one
        must carry it out of one scan and into the next, or the second
        span relaxes towards the wrong value until the next controller
        sample. The second span starts between two samples (cell 18 at an
        interval of 4), where only the held command drives the klystron.
        """
        spans = [(0, 18), (18, self.N_CELLS)]

        def forget(feedback):
            feedback._controller._klystron_command = BIAS

        for make in (_pi, _p):
            with self.subTest(law=make.__name__):
                kernel = self._run(make, True, spans)
                python = self._run(make, False, spans)
                np.testing.assert_allclose(
                    kernel.generator_current_coarse_grid,
                    python.generator_current_coarse_grid,
                    rtol=RTOL,
                )
                # Non-vacuous: a span that forgot the command differs.
                forgetful = self._run(make, True, spans, between=forget)
                self.assertFalse(
                    np.allclose(
                        forgetful.generator_current_coarse_grid,
                        python.generator_current_coarse_grid,
                        rtol=1.0e-6,
                        atol=0.0,
                    )
                )

    def test_the_pole_keeps_the_klystron_limit(self):
        """Clamp, then pole: the output stays inside the limit circle."""
        limit = 0.02001
        for anti_windup in ("conditional", "directional"):
            with self.subTest(anti_windup=anti_windup):
                feedback = self._run(
                    _pi,
                    True,
                    [(0, self.N_CELLS)],
                    max_output=limit,
                    anti_windup=anti_windup,
                    gain_proportional=1.0e-7,
                )
                magnitude = np.abs(feedback.generator_current_coarse_grid)
                self.assertLessEqual(magnitude.max(), limit * (1 + RTOL))
                # Non-vacuous: the output does reach the rail.
                self.assertGreater(magnitude.max(), limit * 0.99)

    def test_a_zero_time_constant_is_the_loop_without_a_pole(self):
        for make in (_pi, _p):
            for use_kernel in (True, False):
                with self.subTest(law=make.__name__, use_kernel=use_kernel):
                    plain = make()
                    zero = make(klystron_time_constant=0.0)
                    grids = []
                    for controller in (plain, zero):
                        feedback = _feedback(
                            controller, use_kernel, self.N_CELLS, interval=4
                        )
                        _track(feedback, [(0, self.N_CELLS)])
                        grids.append(feedback.generator_current_coarse_grid)
                    self.assertTrue(np.array_equal(grids[0], grids[1]))


if __name__ == "__main__":
    unittest.main()
