# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests of the reference-side beam-loading feedforward.

The generator PI regulates the antenna voltage to one scalar setpoint. On
a ring where the beam takes a large bite out of the cavity once per
passage, that setpoint is a target the loop cannot hold: it saturates on
the ripple, and because the anti-windup freezes the integral whenever the
output is clamped, the loop then cannot accumulate the steady term
either. The ripple steals the integrator.

The feedforward adds a **per-cell** term to the reference, so that the
loop regulates to the voltage it can actually hold -- setpoint plus the
predicted beam-induced ripple -- and spends its authority only on what is
left. It does not remove the ripple from the beam's view; nothing the
klystron can produce would (on the muon-collider RCS1 the RF beam current
peaks near 1100 A against 5 mA of generator headroom). It frees the
loop's authority for whatever else needs it, such as a beam phase loop.
"""

import unittest

import numpy as np

from blond.physics.feedbacks.control_law_kernels import envelope_pi_scan


class _Span:
    """One synthetic coarse-grid span, with every kernel argument."""

    def __init__(self, n_cells=64, setpoint=3.0e7 + 0.0j, max_output=np.inf):
        self.n_cells = n_cells
        self.setpoint = setpoint
        self.max_output = max_output
        #: Generator-sourced voltage seeding cell 0. Seed it at the
        #: setpoint to start from steady state, so that the only error
        #: the loop sees is the one the beam makes.
        self.voltage_gen_init = 0.0 + 0.0j
        ones = np.ones(n_cells, dtype=np.complex128)
        # A lossless, undetuned cavity: the propagator is unity, so the
        # arithmetic under test is the controller's and not the cavity's.
        self.voltage_multiplier = ones.copy()
        self.drive_weight = ones.copy()
        self.omega_times_dt = np.full(n_cells, 1.0e-3)
        self.beam_current = np.zeros(n_cells, dtype=np.complex128)
        self.generator_frame_rotation = ones.copy()
        self.kick_frame_rotation = ones.copy()
        self.pi_error_frame_rotation = ones.copy()
        self.beam_step_rotation = ones.copy()

    def run(self, feedforward=None, bias=0.05 + 0.0j):
        """
        Scan the span and return the generator current it commanded.

        Parameters
        ----------
        feedforward
            Per-cell reference feedforward [V], or ``None`` for the
            scalar setpoint alone.
        bias
            Generator current bias ``I_0`` [A].

        Returns
        -------
        generator_current, voltage, voltage_beam
            The commanded current, the antenna voltage and its
            beam-sourced component, per cell.
        """
        if feedforward is None:
            feedforward = np.zeros(self.n_cells, dtype=np.complex128)
        voltage_gen_out = np.empty(self.n_cells, dtype=np.complex128)
        voltage_beam_out = np.empty(self.n_cells, dtype=np.complex128)
        voltage_out = np.empty(self.n_cells, dtype=np.complex128)
        generator_current_out = np.full(
            self.n_cells, bias, dtype=np.complex128
        )
        envelope_pi_scan(
            self.voltage_multiplier,
            self.drive_weight,
            self.omega_times_dt,
            self.beam_current,
            voltage_gen_out,
            voltage_beam_out,
            voltage_out,
            generator_current_out,
            self.voltage_gen_init,
            0.0 + 0.0j,
            bias,
            100.0,
            self.generator_frame_rotation,
            self.kick_frame_rotation,
            self.pi_error_frame_rotation,
            self.beam_step_rotation,
            1,
            0,
            self.setpoint,
            feedforward,
            # No drive-side feedforward: this module is about the
            # reference-side term (see test_feedforward_table.py).
            np.zeros(self.n_cells, dtype=np.complex128),
            1.0e9,
            1.0e-9,
            1.0e-6,
            bias,
            np.zeros(1, dtype=np.complex128),
            0,
            0.0 + 0.0j,
            self.max_output,
        )
        return generator_current_out, voltage_out, voltage_beam_out


class TestFeedforwardEntersTheReference(unittest.TestCase):
    """The per-cell term is added to the setpoint the loop regulates to."""

    def test_a_zero_feedforward_is_bit_neutral(self):
        """Off by default: an all-zero table must change nothing.

        The term is added to the reference, so a zero table adds an exact
        zero and every bit of the commanded current must be unchanged.
        """
        span = _Span()
        without, _, _ = span.run(feedforward=None)
        zeros = np.zeros(span.n_cells, dtype=np.complex128)
        with_zeros, _, _ = span.run(feedforward=zeros)
        np.testing.assert_array_equal(without, with_zeros)

    def test_the_feedforward_shifts_the_error_by_its_own_value(self):
        """A constant offset in the reference is a constant in the error.

        A uniform feedforward ``d`` is indistinguishable from raising the
        scalar setpoint by ``d``, which is the defining property of a
        reference-side term.
        """
        span = _Span()
        offset = 1.0e5 + 0.0j
        shifted, _, _ = span.run(
            feedforward=np.full(span.n_cells, offset, dtype=np.complex128)
        )
        raised = _Span(setpoint=span.setpoint + offset).run()[0]
        np.testing.assert_allclose(shifted, raised, rtol=0.0, atol=0.0)

    def test_it_is_read_per_cell_and_not_once_per_span(self):
        """Two cells with different entries must see different references.

        A term that only ever sat on the span would be useless: the
        ripple it has to cancel lives inside one passage.
        """
        span = _Span(n_cells=8)
        table = np.zeros(span.n_cells, dtype=np.complex128)
        table[4:] = 1.0e5 + 0.0j
        current, _, _ = span.run(feedforward=table)
        flat = span.run()[0]
        np.testing.assert_array_equal(current[:4], flat[:4])
        self.assertNotEqual(complex(current[5]), complex(flat[5]))


class TestFeedforwardMakesTheLoopBlindToTheBeam(unittest.TestCase):
    """The defining property, and the authority it hands back."""

    #: Clamp just above the bias, as on RCS1 where the klystron budget
    #: leaves the loop under 10 % of its operating point in hand.
    MAX_OUTPUT = 0.0548
    BIAS = 0.05 + 0.0j

    def _span(self, max_output=np.inf):
        """A span seeded at its setpoint, so only the beam makes error."""
        span = _Span(n_cells=96, max_output=max_output)
        span.voltage_gen_init = span.setpoint
        return span

    def _with_beam(self, span, current=2.0e8):
        """Give the span a bunch every 32 cells."""
        span.beam_current = np.zeros(span.n_cells, dtype=np.complex128)
        span.beam_current[::32] = current + 0.0j
        return span

    def test_feeding_the_beam_voltage_forward_is_exactly_a_beam_free_loop(
        self,
    ):
        """The property that defines the term, and it holds bit-exactly.

        The loop forms ``error = setpoint + feedforward - V``, and ``V``
        is the sum of a beam-sourced and a generator-sourced component.
        Feed the beam-sourced component forward and it cancels out of the
        error identically, leaving ``setpoint - V_gen``: the very error a
        loop with no beam at all would see. The generator component
        depends only on the commanded current, so the two runs then track
        each other cell for cell, with nothing left to round.
        """
        beam_free, _, _ = self._span().run(bias=self.BIAS)
        loaded = self._with_beam(self._span())
        unfed, _, beam_voltage = loaded.run(bias=self.BIAS)
        fed, _, _ = loaded.run(feedforward=beam_voltage, bias=self.BIAS)
        np.testing.assert_array_equal(fed, beam_free)
        # ... and the beam really was making the loop work, unfed.
        self.assertGreater(np.abs(unfed - beam_free).max(), 1.0e-3)

    def test_it_takes_the_loop_off_the_clamp(self):
        """The product: authority, where the clamp was eating all of it.

        With the beam's bite in the reference the loop sits at its bias
        with the whole klystron margin unspent, which is what a beam
        phase loop writing ``phi_rf_loop`` needs. Without it the loop is
        railed and has none.
        """
        loaded = self._with_beam(self._span(max_output=self.MAX_OUTPUT))
        unfed, _, beam_voltage = loaded.run(bias=self.BIAS)
        fed, _, _ = loaded.run(feedforward=beam_voltage, bias=self.BIAS)
        at_clamp = self.MAX_OUTPUT * (1.0 - 1.0e-12)
        self.assertGreater(np.mean(np.abs(unfed) >= at_clamp), 0.9)
        self.assertEqual(np.sum(np.abs(fed) >= at_clamp), 0)
        self.assertGreater((self.MAX_OUTPUT - np.abs(fed)).min(), 0.0)

    def test_only_the_ripple_belongs_in_the_table_on_a_real_ring(self):
        """Feeding the steady part forward too would double-count the bias.

        A generator bias matched to the *loaded* operating point already
        replaces the average beam loading. Feed the whole beam-sourced
        voltage forward on top of that and the loop stops holding the
        average as well, so the cavity sags by it; only the deviation
        from the average is the loop's to ignore. Here that shows up as
        the two tables differing by exactly the mean.
        """
        loaded = self._with_beam(self._span())
        _, _, beam_voltage = loaded.run(bias=self.BIAS)
        ripple = beam_voltage - beam_voltage.mean()
        self.assertAlmostEqual(complex(ripple.mean()), 0.0 + 0.0j, places=9)
        np.testing.assert_allclose(
            beam_voltage - ripple,
            np.full(loaded.n_cells, beam_voltage.mean()),
            rtol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
