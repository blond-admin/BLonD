# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests of the whole-run feedforward tables of the cavity feedback.

A feedforward table is a precomputed programme, not a loop: BLonD only
*receives* it. Whoever computes it -- a model inversion, a shot-to-shot
learner -- lives outside. Two tables exist and they enter at different
points of the generator loop:

- the **drive-side** table [A] is added to the generator-current bias
  inside the control law, i.e. after the law's correction and before the
  klystron clamp, so the clamp and the anti-windup see the sum;
- the **reference-side** table [V] is added to the voltage setpoint the
  error is formed against.

Both are indexed on the feedback's free-running cell clock, so one table
covers the whole run across spans, passages and turns, and both are read
on controller samples only -- which is why one entry per controller
update loses nothing.
"""

import unittest
import warnings

import numpy as np

from blond.physics.feedbacks.control_law_kernels import (
    envelope_p_scan,
    envelope_pi_scan,
)
from blond.physics.feedbacks.feedforward_table import FeedforwardTable
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPController,
    GeneratorCurrentPIController,
)

from .test_envelope_kernel import (
    BIAS,
    OMEGA_RF,
    _assert_bit_identical,
    _make_feedback,
    _seed_single_segment,
    _snapshot,
)


class TestFeedforwardTable(unittest.TestCase):
    """The table: entries on the cell clock, nothing outside them."""

    def test_an_entry_covers_its_run_of_cells(self):
        table = FeedforwardTable(
            values=[1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j], cells_per_entry=16
        )
        self.assertEqual(table.value_at(0), 1.0 + 0.0j)
        self.assertEqual(table.value_at(15), 1.0 + 0.0j)
        self.assertEqual(table.value_at(16), 2.0 + 1.0j)
        self.assertEqual(table.value_at(47), 3.0 - 1.0j)

    def test_outside_the_table_nothing_is_fed_forward(self):
        """Unlike a gain, a feedforward must not coast on its last entry."""
        table = FeedforwardTable(
            values=[1.0 + 0.0j, 2.0 + 0.0j], cells_per_entry=4, first_cell=8
        )
        self.assertEqual(table.value_at(7), 0.0 + 0.0j)
        self.assertEqual(table.value_at(8), 1.0 + 0.0j)
        self.assertEqual(table.value_at(15), 2.0 + 0.0j)
        self.assertEqual(table.value_at(16), 0.0 + 0.0j)

    def test_over_cells_is_value_at_cell_by_cell(self):
        table = FeedforwardTable(
            values=np.arange(1, 6) * (1.0 + 0.5j),
            cells_per_entry=3,
            first_cell=5,
        )
        span = table.over_cells(first_cell=2, n_cells=24)
        self.assertEqual(span.dtype, np.complex128)
        expected = np.array([table.value_at(cell) for cell in range(2, 26)])
        np.testing.assert_array_equal(span, expected)

    def test_the_span_of_cells_it_covers(self):
        table = FeedforwardTable(values=np.ones(5), cells_per_entry=16)
        self.assertEqual(table.n_entries, 5)
        self.assertEqual(table.n_cells, 80)

    def test_the_values_are_a_private_read_only_copy(self):
        values = np.ones(4, dtype=np.complex128)
        table = FeedforwardTable(values=values, cells_per_entry=1)
        values[0] = 9.0
        self.assertEqual(table.value_at(0), 1.0 + 0.0j)
        with self.assertRaises(ValueError):
            table.values[0] = 9.0

    def test_a_malformed_table_is_refused(self):
        with self.assertRaises(ValueError):
            FeedforwardTable(values=[], cells_per_entry=1)
        with self.assertRaises(ValueError):
            FeedforwardTable(values=[1.0], cells_per_entry=0)
        with self.assertRaises(ValueError):
            FeedforwardTable(values=np.ones((2, 2)), cells_per_entry=1)


class _Span:
    """One synthetic span on a lossless cavity, for either law."""

    SETPOINT = 3.0e7 + 0.0j

    def __init__(self, n_cells=32, interval=1, max_output=np.inf):
        self.n_cells = n_cells
        self.interval = interval
        self.max_output = max_output
        self.ones = np.ones(n_cells, dtype=np.complex128)
        self.zeros = np.zeros(n_cells, dtype=np.complex128)

    def run(self, law, drive=None, bias=0.05 + 0.0j):
        """
        Scan the span with one law.

        Parameters
        ----------
        law
            ``"pi"`` or ``"p"``.
        drive
            Per-cell drive-side feedforward [A], or ``None`` for zeros.
        bias
            Generator current bias [A].

        Returns
        -------
        generator_current, law_state
            The commanded current per cell and the state the scan returns.
        """
        drive = self.zeros if drive is None else drive
        current = np.full(self.n_cells, bias, dtype=np.complex128)
        delay = np.zeros(1, dtype=np.complex128)
        if law == "pi":
            scan = envelope_pi_scan
            law_state = (1.0e-10, 1.0e-6, bias, delay, 0, 0.0 + 0.0j)
        else:
            scan = envelope_p_scan
            law_state = (1.0e-10, bias, delay, 0)
        state = scan(
            self.ones,
            self.ones,
            np.full(self.n_cells, 1.0e-3),
            self.zeros,
            np.empty(self.n_cells, dtype=np.complex128),
            np.empty(self.n_cells, dtype=np.complex128),
            np.empty(self.n_cells, dtype=np.complex128),
            current,
            0.0 + 0.0j,
            0.0 + 0.0j,
            bias,
            100.0,
            self.ones,
            self.ones,
            self.ones,
            self.ones,
            self.interval,
            0,
            self.SETPOINT,
            self.zeros,
            drive,
            1.0e9,
            *law_state,
            self.max_output,
        )
        return current, state


class TestDriveFeedforwardInTheKernels(unittest.TestCase):
    """The drive-side term, in both compiled laws."""

    LAWS = ("pi", "p")

    def test_a_zero_table_is_bit_neutral(self):
        for law in self.LAWS:
            with self.subTest(law=law):
                span = _Span()
                without, _ = span.run(law)
                with_zeros, _ = span.run(law, drive=span.zeros.copy())
                np.testing.assert_array_equal(without, with_zeros)

    def test_a_uniform_table_is_a_raised_bias(self):
        """The defining property of a drive-side term, and it is exact."""
        offset = 3.0e-3 - 1.0e-3j
        for law in self.LAWS:
            with self.subTest(law=law):
                span = _Span()
                fed, _ = span.run(law, drive=np.full(span.n_cells, offset))
                raised = _Span()
                # The seed current stays the unfed bias in both runs.
                current = self._run_with_law_bias(raised, law, offset)
                np.testing.assert_array_equal(fed, current)

    @staticmethod
    def _run_with_law_bias(span, law, offset, bias=0.05 + 0.0j):
        """Scan with the LAW's bias raised, the seed current left alone."""
        current = np.full(span.n_cells, bias, dtype=np.complex128)
        delay = np.zeros(1, dtype=np.complex128)
        if law == "pi":
            scan = envelope_pi_scan
            law_state = (1.0e-10, 1.0e-6, bias + offset, delay, 0, 0.0 + 0.0j)
        else:
            scan = envelope_p_scan
            law_state = (1.0e-10, bias + offset, delay, 0)
        scan(
            span.ones,
            span.ones,
            np.full(span.n_cells, 1.0e-3),
            span.zeros,
            np.empty(span.n_cells, dtype=np.complex128),
            np.empty(span.n_cells, dtype=np.complex128),
            np.empty(span.n_cells, dtype=np.complex128),
            current,
            0.0 + 0.0j,
            0.0 + 0.0j,
            bias,
            100.0,
            span.ones,
            span.ones,
            span.ones,
            span.ones,
            span.interval,
            0,
            span.SETPOINT,
            span.zeros,
            span.zeros,
            1.0e9,
            *law_state,
            span.max_output,
        )
        return current

    def test_the_clamp_acts_on_the_sum(self):
        """Fed forward past the klystron limit, the command is the limit."""
        limit = 0.06
        for law in self.LAWS:
            with self.subTest(law=law):
                span = _Span(max_output=limit)
                fed, _ = span.run(law, drive=np.full(span.n_cells, 0.5 + 0.0j))
                np.testing.assert_allclose(np.abs(fed), limit, rtol=1e-12)

    def test_the_anti_windup_sees_the_sum(self):
        """A table that rails the output must freeze the PI integral."""
        span = _Span(max_output=0.06)
        _, (_, _, integral_free) = span.run("pi")
        _, (_, _, integral_railed) = span.run(
            "pi", drive=np.full(span.n_cells, 0.5 + 0.0j)
        )
        self.assertNotEqual(integral_free, 0.0 + 0.0j)
        self.assertEqual(integral_railed, 0.0 + 0.0j)

    def test_it_is_read_on_controller_samples_only(self):
        """Entries between two samples never reach the generator.

        The command is formed on a sample and held, so at one entry per
        controller update a table loses nothing.
        """
        for law in self.LAWS:
            with self.subTest(law=law):
                span = _Span(n_cells=32, interval=8)
                on_samples = np.zeros(span.n_cells, dtype=np.complex128)
                on_samples[::8] = 2.0e-3
                everywhere = np.full(span.n_cells, 2.0e-3 + 0.0j)
                between = everywhere - on_samples
                np.testing.assert_array_equal(
                    span.run(law, drive=on_samples)[0],
                    span.run(law, drive=everywhere)[0],
                )
                np.testing.assert_array_equal(
                    span.run(law, drive=between)[0], span.run(law)[0]
                )


class TestDriveFeedforwardInTheControllers(unittest.TestCase):
    """The Python twins take the same term, with the same meaning."""

    def _controllers(self, bias):
        return (
            GeneratorCurrentPIController(
                gain_proportional=1.0e-9,
                gain_integral=1.0e-6,
                generator_current_bias=bias,
                n_delay=2,
                max_output=0.06,
            ),
            GeneratorCurrentPController(
                gain_proportional=1.0e-9,
                generator_current_bias=bias,
                n_delay=2,
                max_output=0.06,
            ),
        )

    def test_a_fed_controller_is_one_with_a_raised_bias(self):
        bias = 0.05 + 0.0j
        offset = 2.0e-3 + 1.0e-3j
        rng = np.random.default_rng(5)
        errors = 1.0e5 * (rng.normal(size=40) + 1j * rng.normal(size=40))
        for fed, raised in zip(
            self._controllers(bias), self._controllers(bias + offset)
        ):
            with self.subTest(law=type(fed).__name__):
                for error in errors:
                    self.assertEqual(
                        fed.update_generator_current(
                            error,
                            1.0e-9,
                            generator_current_feedforward=offset,
                        ),
                        raised.update_generator_current(error, 1.0e-9),
                    )


class TestFeedbackReadsItsTablesOnTheCellClock(unittest.TestCase):
    """The feedback expands the tables; the clock runs across spans."""

    N_CELLS = 24
    INTERVAL = 4

    def _run(self, use_kernel, *, spans, law="pi", **tables):
        """
        Track ``spans`` of one seeded grid with the given tables attached.

        Parameters
        ----------
        use_kernel
            Which path to run.
        spans
            ``(start, end)`` index pairs, tracked in order.
        law
            ``"pi"`` or ``"p"``.
        **tables
            ``generator_current_feedforward`` / ``setpoint_feedforward``.

        Returns
        -------
        snapshot
            The post-run snapshot.
        """
        if law == "pi":
            controller = GeneratorCurrentPIController(
                gain_proportional=1.0e-9,
                gain_integral=5.0e-4,
                generator_current_bias=BIAS,
            )
        else:
            controller = GeneratorCurrentPController(
                gain_proportional=1.0e-9,
                generator_current_bias=BIAS,
            )
        feedback = _make_feedback(
            use_kernel,
            controller=controller,
            voltage_setpoint=3.0e7 + 0.0j,
            controller_update_interval=self.INTERVAL,
        )
        for name, table in tables.items():
            setattr(feedback, name, table)
        rng = np.random.default_rng(11)
        beam = 1e-4 * (
            rng.standard_normal(self.N_CELLS)
            + 1j * rng.standard_normal(self.N_CELLS)
        )
        _seed_single_segment(
            feedback,
            self.N_CELLS,
            v_init=3.0e7 + 1.0e6j,
            i_init=BIAS,
            beam=beam,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start, end in spans:
                feedback._circuit_track_cells(
                    omega_input=OMEGA_RF,
                    no_beam=False,
                    start_index=start,
                    end_index=end,
                )
        if law == "pi":
            return _snapshot(feedback)
        return {
            "V": feedback.antenna_voltage_coarse_grid.copy(),
            "V_gen": feedback.antenna_voltage_gen_coarse_grid.copy(),
            "V_beam": feedback.antenna_voltage_beam_coarse_grid.copy(),
            "I": feedback.generator_current_coarse_grid.copy(),
        }

    def _table(self, scale):
        """One entry per controller update over the whole grid."""
        n_entries = self.N_CELLS // self.INTERVAL
        values = scale * (np.arange(1, n_entries + 1) * (1.0 - 0.5j))
        return FeedforwardTable(values=values, cells_per_entry=self.INTERVAL)

    def test_the_tables_are_constructor_knobs_and_off_by_default(self):
        feedback = _make_feedback(True)
        self.assertIsNone(feedback.generator_current_feedforward)
        self.assertIsNone(feedback.setpoint_feedforward)

    def test_no_table_and_a_zero_table_are_the_same_run(self):
        whole = ((0, self.N_CELLS),)
        for use_kernel in (True, False):
            with self.subTest(use_kernel=use_kernel):
                _assert_bit_identical(
                    self,
                    self._run(use_kernel, spans=whole),
                    self._run(
                        use_kernel,
                        spans=whole,
                        generator_current_feedforward=self._table(0.0),
                        setpoint_feedforward=self._table(0.0),
                    ),
                )

    def test_the_drive_table_moves_the_command(self):
        whole = ((0, self.N_CELLS),)
        fed = self._run(
            True,
            spans=whole,
            generator_current_feedforward=self._table(1.0e-4),
        )
        unfed = self._run(True, spans=whole)
        self.assertFalse(np.array_equal(fed["I"], unfed["I"]))

    def test_the_two_paths_agree_with_both_tables(self):
        tables = {
            "generator_current_feedforward": self._table(1.0e-4),
            "setpoint_feedforward": self._table(1.0e4),
        }
        whole = ((0, self.N_CELLS),)
        for law in ("pi", "p"):
            with self.subTest(law=law):
                _assert_bit_identical(
                    self,
                    self._run(True, spans=whole, law=law, **tables),
                    self._run(False, spans=whole, law=law, **tables),
                )

    def test_the_tables_follow_the_cell_clock_across_spans(self):
        """A span boundary must not re-index the tables.

        Spans of 6, 7 and 11 cells put their boundaries off the
        controller clock. An entry covering global cells 12-15 is read
        on the sample at global cell 12 -- local cell 6 of the second
        span -- and one covering 16-19 at local cell 3 of the third. A
        table indexed per span would find neither there.
        """
        split = ((0, 6), (6, 13), (13, self.N_CELLS))
        n_entries = self.N_CELLS // self.INTERVAL
        for name, scale in (
            ("generator_current_feedforward", 1.0e-4),
            ("setpoint_feedforward", 1.0e4),
        ):
            for entry in (3, 4):
                values = np.zeros(n_entries, dtype=np.complex128)
                values[entry] = scale
                table = FeedforwardTable(
                    values=values, cells_per_entry=self.INTERVAL
                )
                for use_kernel in (True, False):
                    with self.subTest(
                        table=name, entry=entry, use_kernel=use_kernel
                    ):
                        fed = self._run(
                            use_kernel, spans=split, **{name: table}
                        )["I"]
                        unfed = self._run(use_kernel, spans=split)["I"]
                        moved = np.flatnonzero(fed != unfed)
                        self.assertEqual(moved[0], entry * self.INTERVAL)

    def test_a_drive_table_without_a_controller_is_refused(self):
        """Nothing would carry it: say so instead of dropping it."""
        feedback = _make_feedback(True)
        feedback.generator_current_feedforward = self._table(1.0e-4)
        _seed_single_segment(
            feedback, 8, v_init=3.0e7 + 0.0j, i_init=BIAS, beam=None
        )
        with self.assertRaises(ValueError) as caught:
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF, no_beam=True, start_index=0, end_index=8
            )
        self.assertIn("generator_current_feedforward", str(caught.exception))

    def test_the_in_feedback_learner_is_gone(self):
        """Learning a table is the caller's business, not the feedback's."""
        feedback = _make_feedback(True)
        for name in (
            "beam_loading_feedforward",
            "setpoint_feedforward_coarse_grid",
            "_learn_beam_loading_feedforward",
            "_install_beam_loading_feedforward",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(feedback, name))


class _RecordingController(GeneratorCurrentPController):
    """A P law that keeps every error it was handed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.errors = []

    #: Driven cell by cell, so the errors pass through Python.
    supports_envelope_scan = False

    def update_generator_current(
        self, error, delta_t, generator_current_feedforward=0.0 + 0.0j
    ):
        self.errors.append(complex(error))
        return super().update_generator_current(
            error, delta_t, generator_current_feedforward
        )


class TestWhatALearnerReadsBack(unittest.TestCase):
    """The public read side: where the grid sits on the clock, and the error.

    A table is written against the cell clock, so whoever computes one
    has to be able to place what it measured on that clock, and to read
    the error the loop actually regulated on -- without reaching into
    the feedback's private state.
    """

    N_CELLS = 24
    INTERVAL = 4

    def _tracked(self, spans, setpoint_feedforward=None):
        controller = _RecordingController(
            gain_proportional=1.0e-9, generator_current_bias=BIAS
        )
        feedback = _make_feedback(
            False,
            controller=controller,
            voltage_setpoint=3.0e7 + 0.0j,
            controller_update_interval=self.INTERVAL,
        )
        feedback.setpoint_feedforward = setpoint_feedforward
        rng = np.random.default_rng(2)
        beam = 1e-4 * (
            rng.standard_normal(self.N_CELLS)
            + 1j * rng.standard_normal(self.N_CELLS)
        )
        _seed_single_segment(
            feedback,
            self.N_CELLS,
            v_init=3.0e7 + 1.0e6j,
            i_init=BIAS,
            beam=beam,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start, end in spans:
                feedback._circuit_track_cells(
                    omega_input=OMEGA_RF,
                    no_beam=False,
                    start_index=start,
                    end_index=end,
                )
        return feedback, controller

    def test_the_cell_clock_is_public_and_counts_tracked_cells(self):
        feedback, _ = self._tracked(((0, 10), (10, self.N_CELLS)))
        self.assertEqual(feedback.cell_clock, self.N_CELLS)
        self.assertEqual(_make_feedback(True).cell_clock, 0)

    def test_the_standing_grid_is_placed_on_the_clock(self):
        """A second passage's grid starts where the first one ended."""
        feedback, _ = self._tracked(((0, self.N_CELLS),))
        self.assertEqual(feedback.coarse_grid_first_cell, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feedback._circuit_track_cells(
                omega_input=OMEGA_RF,
                no_beam=False,
                start_index=0,
                end_index=self.N_CELLS,
            )
        self.assertEqual(feedback.coarse_grid_first_cell, self.N_CELLS)

    def test_the_error_grid_is_the_error_the_controller_was_handed(self):
        n_entries = self.N_CELLS // self.INTERVAL
        table = FeedforwardTable(
            values=1.0e4 * np.arange(1, n_entries + 1) * (1.0 + 0.5j),
            cells_per_entry=self.INTERVAL,
        )
        for setpoint_feedforward in (None, table):
            with self.subTest(fed=setpoint_feedforward is not None):
                feedback, controller = self._tracked(
                    ((0, self.N_CELLS),), setpoint_feedforward
                )
                error = feedback.regulation_error_coarse_grid()
                self.assertEqual(error.shape, (self.N_CELLS,))
                np.testing.assert_array_equal(
                    error[:: self.INTERVAL], np.array(controller.errors)
                )

    def test_the_error_against_the_setpoint_leaves_the_table_out(self):
        """Fed or not, the voltage is as far from the setpoint as it is.

        The two errors differ by the table carried into the actuator
        frame, which is what the public frame rotation is for.
        """
        n_entries = self.N_CELLS // self.INTERVAL
        table = FeedforwardTable(
            values=1.0e4 * np.arange(1, n_entries + 1) * (1.0 + 0.5j),
            cells_per_entry=self.INTERVAL,
        )
        feedback, _ = self._tracked(((0, self.N_CELLS),), table)
        seen = feedback.regulation_error_coarse_grid()
        true = feedback.regulation_error_coarse_grid(include_feedforward=False)
        rotation = feedback.error_frame_rotation_coarse_grid()
        np.testing.assert_allclose(np.abs(rotation), 1.0, rtol=1e-14)
        np.testing.assert_allclose(
            seen - true,
            table.over_cells(0, self.N_CELLS) * rotation,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
