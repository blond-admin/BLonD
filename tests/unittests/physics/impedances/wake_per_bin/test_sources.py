# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import warnings
from unittest.mock import Mock, patch

import numpy as np
from scipy.constants import pi

from blond import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.simulation.simulation import Simulation
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.readers import CsvReader
from blond.physics.impedances.sources import (
    ImpedanceTableTime,
    Resonators,
    TravelingWaveCavity,
)
from blond.testing.backend_testing import BLonDTestCase


def _gauss_legendre_bspline_average(wake, time, dt, breakpoints, nodes=48):
    """
    Exact quadratic B-spline average of a piecewise-smooth wake.

    Split at the B-spline knots and at every breakpoint of the wake, so
    Gauss-Legendre quadrature converges to round-off on each piece.

    Parameters
    ----------
    wake
        Vectorised wake ``W(t)``.
    time
        Times at which to evaluate the average, in [s].
    dt
        Bin width, in [s].
    breakpoints
        Times where ``W`` or its slope jumps, in [s].
    nodes
        Gauss-Legendre nodes per piece.

    Returns
    -------
    wake_per_bin
        ``W * B_2`` at ``time``.
    """
    abscissae, weights = np.polynomial.legendre.leggauss(nodes)
    breakpoints = np.asarray(breakpoints, dtype=float)
    out = np.zeros(len(time))
    for index, time_entry in enumerate(time):
        relative = (breakpoints - time_entry) / dt
        cuts = np.unique(
            np.concatenate(
                (
                    [-1.5, -0.5, 0.5, 1.5],
                    relative[(relative > -1.5) & (relative < 1.5)],
                )
            )
        )
        for lower, upper in zip(cuts[:-1], cuts[1:], strict=False):
            offset = 0.5 * (upper - lower) * abscissae + 0.5 * (upper + lower)
            distance = np.abs(offset)
            bspline = np.where(
                distance < 0.5,
                0.75 - distance**2,
                np.where(distance < 1.5, 0.5 * (1.5 - distance) ** 2, 0.0),
            )
            out[index] += (
                0.5
                * (upper - lower)
                * np.sum(weights * bspline * wake(time_entry + offset * dt))
            )
    return out


class TestImpedanceTableTimeWakePerBin(BLonDTestCase):
    def test_get_wake_is_zero_below_the_table(self):
        """A causal table must not fabricate a wake before it starts.

        Bin-averaged wakes are sampled from one bin before the profile
        (the kernel has a non-causal tap), so the interpolation must return
        zero below the table, not the clamped first sample.
        """
        table = ImpedanceTableTime(
            wake_x=backend.array([1.0, 2.0, 3.0]),
            wake_y=backend.array([10.0, 20.0, 30.0]),
            first_sample_is_half_jump=True,
        )
        wake = copy_to_cpu(table.get_wake(backend.array([0.0, 0.5, 1.0])))
        np.testing.assert_allclose(wake, [0.0, 0.0, 10.0])

    def test_get_wake_per_bin_default_is_bspline_bin_average(self):
        """The generic `TimeDomain.get_wake_per_bin` bin-averages.

        A table wake is piecewise linear, so weighting it with the
        quadratic B-spline ``box * box * box`` is exactly the stencil
        (w[n-2] + 76 w[n-1] + 230 w[n] + 76 w[n+1] + w[n+2]) / 384 on the
        interior samples.
        """
        dt = 1e-11
        t = np.arange(64) * dt
        w = np.sin(2 * np.pi * 3e9 * t)
        table = ImpedanceTableTime(
            wake_x=backend.array(t), wake_y=backend.array(w)
        )
        time = backend.array(t)

        binned = copy_to_cpu(table.get_wake_per_bin(time))
        expected = (
            np.roll(w, 2)
            + 76 * np.roll(w, 1)
            + 230 * w
            + 76 * np.roll(w, -1)
            + np.roll(w, -2)
        ) / 384

        np.testing.assert_allclose(
            binned[2:-2], expected[2:-2], rtol=1e-12, atol=1e-14
        )
        # and it genuinely differs from point-sampling get_wake
        self.assertFalse(
            np.allclose(binned, copy_to_cpu(table.get_wake(time)))
        )

    def test_no_boundary_warning_for_the_non_causal_tap(self):
        """A table exactly spanning the requested axis must not warn.

        ``get_impedance_from_wake`` samples one bin below the axis it is
        given, to pick up the bin-average kernel's non-causal tap; below the
        table the wake is zero by definition, not out of bounds.
        """
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.array(np.arange(64) * 1e-11)
        table = ImpedanceTableTime(
            wake_x=time,
            wake_y=backend.array(
                np.sin(2 * np.pi * 3e9 * np.arange(64) * 1e-11)
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table.get_impedance_from_wake(
                time=time, simulation=simulation, beam=beam, n_fft=128
            )
        self.assertEqual(
            [w for w in caught if "outside boundaries" in str(w.message)], []
        )

    def test_get_wake_per_bin_is_exact_across_the_causal_onset(self):
        """A tabulated wake must bin-average exactly across its own step.

        ``TimeDomain.get_impedance_from_wake`` samples the bin-averaged wake
        from ``time - dt`` to pick up the kernel's one non-causal tap, so
        the wake must be zero below the table, not the clamped first
        sample. The generic stencil B-spline-averages the piecewise-linear
        interpolant through the *query* samples, which models the causal
        onset as a one-bin ramp instead of a step; that leaves about
        ``(76 / 384) * W(0)`` -- some 8 % of the peak -- at the samples
        straddling the onset, an error that does not shrink when the table
        is refined. ``ImpedanceTableTime`` integrates the model the table
        really represents instead, so every sample, including the two
        straddling the onset, matches the closed form of
        ``Resonators.get_wake_per_bin`` to the accuracy of the
        piecewise-linear representation alone.
        """
        resonator = Resonators(
            shunt_impedances=np.array([1.0e4]),
            center_frequencies=np.array([1.0e9]),
            quality_factors=np.array([5.0]),
        )
        bin_step = 1e-11
        time = backend.array(np.arange(400) * bin_step)
        table = ImpedanceTableTime(
            wake_x=time,
            wake_y=resonator.get_wake(time),
            first_sample_is_half_jump=True,
        )

        # sampled from `time - bin_step`, the way `get_impedance_from_wake`
        # does: index 0 is lag -dt (before the table starts), index 1 is
        # lag 0
        shifted_time = time - bin_step
        tabulated = copy_to_cpu(table.get_wake_per_bin(shifted_time))
        analytic = copy_to_cpu(resonator.get_wake_per_bin(shifted_time))
        peak = np.max(np.abs(analytic))

        # The last two samples are excluded: there the analytic wake keeps
        # going while the table has run out and clamps.
        np.testing.assert_allclose(
            tabulated[:-2], analytic[:-2], atol=4e-4 * peak
        )

    def test_get_wake_per_bin_only_visits_knots_near_the_query(self):
        """A long table must cost what the queried window holds, not more.

        Knots more than 1.5 bins outside the query cannot contribute to
        any bin average, so the kernels must only be evaluated on those
        near the query. Evaluating the whole table cost 12.6 GB and 28 s
        per call for an 81-million-point table queried over 1000 bins.
        """
        from blond.physics.impedances.wake_per_bin import sources

        knots = np.arange(200_000) * 1e-12
        table = ImpedanceTableTime(
            wake_x=backend.array(knots),
            wake_y=backend.array(np.cos(2 * np.pi * 1e9 * knots)),
        )
        query = backend.array(50e-9 + np.arange(20) * 1e-10)
        sizes = []
        original = sources.bspline_kink_kernel

        def recording_kernel(distance):
            sizes.append(distance.size)
            return original(distance)

        with patch.object(sources, "bspline_kink_kernel", recording_kernel):
            table.get_wake_per_bin(query)

        # 20 bins + 2 * 1.5 bins of margin, at 100 knots per bin
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 2400)

    def test_get_wake_per_bin_inside_a_long_table_keeps_a_repeated_onset(self):
        """An onset spelled as two samples at one time, before the window.

        ``wake_x = [0, 0, ...]`` jumps straight to ``wake_y[1]``, whatever
        ``wake_y[0]`` and the onset convention say.
        """
        bin_step = 1e-10
        wake_x = np.concatenate(([0.0], np.arange(0, 400) * 2.5e-11))
        wake_y = np.concatenate(([7.0], np.cos(2 * np.pi * 3e9 * wake_x[1:])))
        query = 3e-9 + np.arange(30) * bin_step
        expected = _gauss_legendre_bspline_average(
            lambda t: np.interp(t, wake_x[1:], wake_y[1:], left=0.0),
            query,
            bin_step,
            wake_x,
            nodes=4,
        )
        for half_jump in (False, True):
            with self.subTest(half_jump=half_jump):
                table = ImpedanceTableTime(
                    wake_x=backend.array(wake_x),
                    wake_y=backend.array(wake_y),
                    first_sample_is_half_jump=half_jump,
                )
                np.testing.assert_allclose(
                    copy_to_cpu(table.get_wake_per_bin(backend.array(query))),
                    expected,
                    rtol=0,
                    atol=1e-12,
                )

    def test_get_wake_per_bin_inside_a_long_table_keeps_earlier_jumps(self):
        """Restricting to the query window must keep the jumps before it.

        The onset jump and an explicit jump spelled as a repeated time both
        lie before the queried window; the bin average there must still
        include them, exactly as the full piecewise-linear model does.
        """
        bin_step = 1e-10
        wake_x = np.concatenate(
            (np.arange(0, 41) * 2.5e-11, np.arange(40, 400) * 2.5e-11)
        )
        wake_y = np.cos(2 * np.pi * 3e9 * wake_x) + 2.0
        wake_y[41:] += 5.0  # explicit jump at the repeated time 1 ns
        for half_jump in (False, True):
            with self.subTest(half_jump=half_jump):
                table = ImpedanceTableTime(
                    wake_x=backend.array(wake_x),
                    wake_y=backend.array(wake_y),
                    first_sample_is_half_jump=half_jump,
                )
                model_y = wake_y.copy()
                if half_jump:
                    model_y[0] *= 2

                def model(t, model_y=model_y):
                    # right-continuous at the repeated time
                    below = np.interp(t, wake_x[:41], model_y[:41], left=0.0)
                    above = np.interp(t, wake_x[41:], model_y[41:])
                    return np.where(t >= wake_x[41], above, below)

                query = 3e-9 + np.arange(30) * bin_step
                expected = _gauss_legendre_bspline_average(
                    model, query, bin_step, wake_x, nodes=4
                )
                np.testing.assert_allclose(
                    copy_to_cpu(table.get_wake_per_bin(backend.array(query))),
                    expected,
                    rtol=0,
                    atol=1e-12,
                )

    def test_get_wake_per_bin_averages_at_table_resolution(self):
        """A table finer than the bins must not be resampled at the bins.

        The table represents its piecewise-linear interpolant, so the bin
        average of that interpolant is what the exact answer is. Sampling
        it at the bin width first and smoothing with the 5-point stencil
        throws away every oscillation the bins do not resolve: an 8 GHz
        wake tabulated at 6.25 ps and binned at 100 ps came out wrong by
        orders of magnitude. Covers a mode well resolved by the bins, one
        far above their Nyquist frequency, a table that does not start on
        a bin and both onset conventions.
        """
        bin_step = 1e-10
        table_step = bin_step / 16
        query = np.arange(-4, 40) * bin_step
        for frequency in (2e8, 8e9):
            for start in (0.0, 0.37 * bin_step):
                for half_jump in (False, True):
                    with self.subTest(
                        frequency=frequency, start=start, half_jump=half_jump
                    ):
                        wake_x = start + np.arange(480) * table_step
                        wake_y = np.exp(-(wake_x - start) / 2e-9) * np.cos(
                            2 * np.pi * frequency * (wake_x - start)
                        )
                        table = ImpedanceTableTime(
                            wake_x=backend.array(wake_x),
                            wake_y=backend.array(wake_y),
                            first_sample_is_half_jump=half_jump,
                        )
                        model_y = wake_y.copy()
                        if half_jump:
                            model_y[0] *= 2
                        expected = _gauss_legendre_bspline_average(
                            lambda t, x=wake_x, y=model_y: np.interp(
                                t, x, y, left=0.0
                            ),
                            query,
                            bin_step,
                            wake_x,
                            nodes=4,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = copy_to_cpu(
                                table.get_wake_per_bin(backend.array(query))
                            )
                        np.testing.assert_allclose(
                            result, expected, rtol=0, atol=1e-12
                        )

    def test_get_wake_per_bin_away_from_the_onset_is_untouched(self):
        """A table whose onset is out of reach tracks the closed form.

        The stencil has the kernel's support, so a query axis that starts
        well above the table's first time never sees the onset -- the
        result must track the analytic closed form to the accuracy of the
        piecewise-linear representation alone, including the first and
        last query bins, where the stencil reaches beyond the query axis.
        """
        resonator = Resonators(
            shunt_impedances=np.array([1.0e4]),
            center_frequencies=np.array([1.0e9]),
            quality_factors=np.array([5.0]),
        )
        bin_step = 1e-11
        table_time = backend.array(np.arange(300, 1000) * bin_step)
        table = ImpedanceTableTime(
            wake_x=table_time, wake_y=resonator.get_wake(table_time)
        )
        query = backend.array(np.arange(320, 900) * bin_step)
        tabulated = copy_to_cpu(table.get_wake_per_bin(query))
        analytic = copy_to_cpu(resonator.get_wake_per_bin(query))
        peak = np.max(
            np.abs(
                copy_to_cpu(
                    resonator.get_wake_per_bin(
                        backend.array(np.arange(1000) * bin_step)
                    )
                )
            )
        )
        np.testing.assert_allclose(tabulated, analytic, atol=1e-3 * peak)

    def test_get_wake_per_bin_half_jump_convention_is_pinned(self):
        """The half-jump convention integrates a jump of ``2 * wake_y[0]``.

        ``wake_y[0]`` of a BLonD-sampled table is half the causal jump
        (``Resonators.heaviside_eps_at_0`` returns ``sign(t) + 1``, i.e. 1
        at ``t = 0`` and 2 above it), so the bin average must integrate a
        jump of ``2 * wake_y[0]``. The values below are pinned from when
        this was the hard-wired assumption.
        """
        wake_x = backend.array(np.arange(5) * 1e-11)
        wake_y = backend.array(np.array([7.0, 6.0, 5.0, 4.0, 3.0]))
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=True
        )
        query = backend.array(np.arange(-1, 5) * 1e-11)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            half_jump = copy_to_cpu(half_jump_table.get_wake_per_bin(query))

        np.testing.assert_allclose(
            half_jump,
            np.array(
                [
                    0.2708333333333335,
                    5.393229166666667,
                    7.109375,
                    5.018229166666667,
                    4.002604166666667,
                    3.203125,
                ]
            ),
            rtol=1e-12,
        )

    def test_get_wake_per_bin_literal_first_sample_undoubles_the_onset(self):
        """An external wake table stores the right limit, not half of it.

        A table from a wake solver (CST, ABCI, a measurement) records
        ``W(t_0^+)`` itself, so doubling it would overstate the onset by a
        factor of two. Selecting ``first_sample_is_half_jump=False`` must
        integrate a jump of ``wake_y[0]`` -- which is exactly what feeding
        half that first sample to the half-jump convention describes.
        """
        wake_x = backend.array(np.arange(5) * 1e-11)
        wake_values = np.array([7.0, 6.0, 5.0, 4.0, 3.0])
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=backend.array(wake_values),
            first_sample_is_half_jump=False,
        )
        halved_values = wake_values.copy()
        halved_values[0] *= 0.5
        halved_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=backend.array(halved_values),
            first_sample_is_half_jump=True,
        )
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=backend.array(wake_values),
            first_sample_is_half_jump=True,
        )
        query = backend.array(np.arange(-1, 5) * 1e-11)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            literal = copy_to_cpu(literal_table.get_wake_per_bin(query))
            halved = copy_to_cpu(halved_table.get_wake_per_bin(query))
            half_jump = copy_to_cpu(half_jump_table.get_wake_per_bin(query))

        np.testing.assert_allclose(literal, halved, rtol=1e-12)
        # the onset is genuinely halved where the kernel reaches over it ...
        self.assertFalse(np.allclose(literal[:3], half_jump[:3]))
        # ... and untouched beyond the kernel's support
        np.testing.assert_allclose(literal[4:], half_jump[4:], rtol=1e-12)

    def test_get_wake_per_bin_conventions_agree_for_a_zero_first_sample(self):
        """With no jump to interpret, the convention cannot matter.

        A table whose first sample is zero -- what an external solver
        starting at or below ``t = 0`` normally gives -- carries no causal
        jump at all, so both readings of ``wake_y[0]`` describe the same
        function and must agree exactly.
        """
        bin_step = 1e-11
        wake_x = backend.array(np.arange(64) * bin_step)
        wake_values = np.concatenate(
            (np.linspace(0.0, 5.0, 32), np.linspace(5.0, 1.0, 32))
        )
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=backend.array(wake_values),
            first_sample_is_half_jump=True,
        )
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=backend.array(wake_values),
            first_sample_is_half_jump=False,
        )
        query = backend.array(np.arange(-1, 63) * bin_step)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            half_jump = copy_to_cpu(half_jump_table.get_wake_per_bin(query))
            literal = copy_to_cpu(literal_table.get_wake_per_bin(query))

        np.testing.assert_array_equal(half_jump, literal)

    def test_first_sample_is_read_literally_by_default(self):
        """Without an explicit choice ``wake_y[0]`` is the wake value.

        That is what a table from an external wake solver records, and
        reading it as half a jump would silently double the onset.
        """
        wake_x = backend.array(np.arange(1.0, 6.0))
        wake_y = backend.array(10 * np.arange(1.0, 6.0))
        default = ImpedanceTableTime(wake_x=wake_x, wake_y=wake_y)
        literal = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=False
        )
        loaded = ImpedanceTableTime.from_file(
            filepath=callers_relative_path(
                "../resources/example_impedance_table.csv", stacklevel=1
            ),
            reader=CsvReader(delimiter=","),
        )
        query = backend.array(np.linspace(0.5, 5.0, 24))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = copy_to_cpu(literal.get_wake_per_bin(query))
            np.testing.assert_array_equal(
                copy_to_cpu(default.get_wake_per_bin(query)), expected
            )
            np.testing.assert_array_equal(
                copy_to_cpu(loaded.get_wake_per_bin(query)), expected
            )

    def test_from_file_passes_on_the_first_sample_convention(self):
        """``from_file`` must be able to select the convention too.

        External tables are exactly the ones loaded from disk, so the
        loader has to expose the same choice as the constructor.
        """
        filepath = callers_relative_path(
            "../resources/example_impedance_table.csv", stacklevel=1
        )
        loaded = ImpedanceTableTime.from_file(
            filepath=filepath,
            reader=CsvReader(delimiter=","),
            first_sample_is_half_jump=False,
        )
        constructed = ImpedanceTableTime(
            wake_x=backend.array(np.arange(1.0, 6.0)),
            wake_y=backend.array(10 * np.arange(1.0, 6.0)),
            first_sample_is_half_jump=False,
        )
        query = backend.array(np.linspace(0.5, 5.0, 24))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(
                copy_to_cpu(loaded.get_wake_per_bin(query)),
                copy_to_cpu(constructed.get_wake_per_bin(query)),
                rtol=1e-12,
            )

    def test_get_wake_per_bin_without_a_step_is_the_plain_stencil(self):
        """No step at the table start means no correction to the stencil.

        A table whose first sample is zero, laid out on the query grid, is
        exactly the piecewise-linear interpolant the generic stencil already
        integrates, so the override must reproduce it bit for bit. Same for a
        table that only starts long after the query axis does, where the wake
        has already decayed away.
        """
        bin_step = 1e-11
        t = np.arange(64) * bin_step
        # genuinely piecewise linear, zero at the table start
        w = np.concatenate(
            (np.linspace(0.0, 5.0, 32), np.linspace(5.0, 1.0, 32))
        )
        table = ImpedanceTableTime(
            wake_x=backend.array(t), wake_y=backend.array(w)
        )
        binned = copy_to_cpu(table.get_wake_per_bin(backend.array(t)))
        stencil = (
            np.roll(w, 2)
            + 76 * np.roll(w, 1)
            + 230 * w
            + 76 * np.roll(w, -1)
            + np.roll(w, -2)
        ) / 384
        np.testing.assert_allclose(binned[2:-2], stencil[2:-2], rtol=1e-12)

    def test_get_wake_per_bin_is_exact_for_an_off_grid_onset(self):
        """The onset need not land on a query point.

        The correction integrates the part of the cell that lies above the
        onset, so it must stay exact -- and free of any cliff -- for every
        sub-bin position of the onset, including the two degenerate ends
        where the onset all but coincides with a grid point.
        """
        resonator = Resonators(
            shunt_impedances=np.array([1.0e4]),
            center_frequencies=np.array([1.0e9]),
            quality_factors=np.array([5.0]),
        )
        bin_step = 1e-11
        table_time = backend.array(np.arange(600) * bin_step)
        table = ImpedanceTableTime(
            wake_x=table_time,
            wake_y=resonator.get_wake(table_time),
            first_sample_is_half_jump=True,
        )

        for onset_offset in (0.0, 0.13, 0.5, 0.87, 0.999):
            with self.subTest(onset_offset=onset_offset):
                query = backend.array(
                    (np.arange(-2, 500) + onset_offset) * bin_step
                )
                tabulated = copy_to_cpu(table.get_wake_per_bin(query))
                analytic = copy_to_cpu(resonator.get_wake_per_bin(query))
                peak = np.max(np.abs(analytic))
                np.testing.assert_allclose(
                    tabulated, analytic, atol=1e-3 * peak
                )

    def test_get_wake_half_jump_is_plain_interpolation(self):
        """The half-jump convention must keep the point sample verbatim.

        Under ``first_sample_is_half_jump=True`` the tabulated first sample
        already *is* the midpoint of the causal jump, so the point wake is
        the bare causal interpolant. Pinned bit-exactly, because the flag
        must not perturb this path at all.
        """
        bin_step = 1e-11
        wake_x = backend.array(np.arange(5) * bin_step)
        wake_y = backend.array(np.array([7.0, 6.0, 5.0, 4.0, 3.0]))
        table = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=True
        )
        query = backend.array(
            np.array([-1.5e-11, 0.0, 0.4e-11, 1e-11, 3.9e-11])
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wake = copy_to_cpu(table.get_wake(query))

        expected = copy_to_cpu(backend.interp(query, wake_x, wake_y, left=0.0))
        np.testing.assert_array_equal(wake, expected)

    def test_get_wake_literal_onset_is_halved(self):
        """A literal first sample must be halved at the onset itself.

        With ``first_sample_is_half_jump=False`` the table stores the right
        limit ``W(x0+)``, so a particle sitting exactly on the onset sees
        half of it -- the same beam-loading statement
        :meth:`Resonators.heaviside_eps_at_0` makes by returning 1 (not 2)
        at ``t = 0``.
        """
        bin_step = 1e-11
        wake_x = backend.array(np.arange(5) * bin_step)
        wake_y = backend.array(np.array([7.0, 6.0, 5.0, 4.0, 3.0]))
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=wake_y,
            first_sample_is_half_jump=False,
        )
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=True
        )
        query = backend.array(np.array([0.0]))

        literal = copy_to_cpu(literal_table.get_wake(query))
        half_jump = copy_to_cpu(half_jump_table.get_wake(query))

        np.testing.assert_allclose(literal, np.array([3.5]))
        np.testing.assert_allclose(half_jump, np.array([7.0]))

    def test_get_wake_literal_onset_only_at_the_onset(self):
        """Only the onset sample itself changes under the literal reading.

        Strictly below the onset the wake stays zero (causality) and
        strictly above it stays the interpolant -- the halving is the
        value *at* the discontinuity, not a rescaling of the table.
        """
        bin_step = 1e-11
        wake_x = backend.array(np.arange(5) * bin_step)
        wake_y = backend.array(np.array([7.0, 6.0, 5.0, 4.0, 3.0]))
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=wake_y,
            first_sample_is_half_jump=False,
        )
        query = backend.array(
            np.array([-2e-11, -0.5e-11, 0.5e-11, 1e-11, 2e-11])
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wake = copy_to_cpu(literal_table.get_wake(query))

        np.testing.assert_allclose(wake, np.array([0.0, 0.0, 6.5, 6.0, 5.0]))

    def test_get_wake_conventions_agree_without_a_jump(self):
        """No jump to interpret means no difference between conventions."""
        bin_step = 1e-11
        wake_x = backend.array(np.arange(5) * bin_step)
        wake_y = backend.array(np.array([0.0, 6.0, 5.0, 4.0, 3.0]))
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=True
        )
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=wake_y,
            first_sample_is_half_jump=False,
        )
        query = backend.array(np.array([-1e-11, 0.0, 0.5e-11, 2e-11]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            half_jump = copy_to_cpu(half_jump_table.get_wake(query))
            literal = copy_to_cpu(literal_table.get_wake(query))

        np.testing.assert_array_equal(half_jump, literal)

    def test_get_wake_accepts_a_single_time_sample(self):
        """A one-element query must work; there is no bin step to read.

        The onset tolerance therefore cannot come from the query axis --
        it is taken from the table's own first spacing, which also survives
        a non-uniformly sampled table.
        """
        wake_x = backend.array(np.array([0.0, 1e-11, 5e-11, 20e-11]))
        wake_y = backend.array(np.array([7.0, 6.0, 5.0, 4.0]))
        literal_table = ImpedanceTableTime(
            wake_x=wake_x,
            wake_y=wake_y,
            first_sample_is_half_jump=False,
        )
        half_jump_table = ImpedanceTableTime(
            wake_x=wake_x, wake_y=wake_y, first_sample_is_half_jump=True
        )
        query = backend.array(np.array([0.0]))

        literal = copy_to_cpu(literal_table.get_wake(query))
        half_jump = copy_to_cpu(half_jump_table.get_wake(query))

        np.testing.assert_allclose(literal, np.array([3.5]))
        np.testing.assert_allclose(half_jump, np.array([7.0]))

    def test_get_wake_single_point_literal_table_is_causal(self):
        """A one-sample table has no first spacing to place the onset by.

        It is still causal: zero below the sample, the sample above it.
        """
        literal_table = ImpedanceTableTime(
            wake_x=backend.array(np.array([0.0])),
            wake_y=backend.array(np.array([7.0])),
            first_sample_is_half_jump=False,
        )
        query = backend.array(np.array([-1e-11, 1e-11]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wake = copy_to_cpu(literal_table.get_wake(query))

        np.testing.assert_allclose(wake, np.array([0.0, 7.0]))

    def test_get_wake_per_bin_repeated_onset_time_is_an_explicit_jump(self):
        """A table may spell the causal jump as two samples at one time.

        ``wake_x = [0, 0, dt, ...]`` states the right limit at the onset
        directly in ``wake_y[1]``, so the bin average must integrate a
        jump to ``wake_y[1]`` and ignore the half-jump reading of
        ``wake_y[0]``. The reference integrates that model against the
        quadratic B-spline numerically, with the midpoint rule so that no
        sample sits on the jump.
        """
        bin_step = 1e-11
        wake_x = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * bin_step
        wake_y = np.array([7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
        table = ImpedanceTableTime(
            wake_x=backend.array(wake_x), wake_y=backend.array(wake_y)
        )
        query = np.arange(-1, 4) * bin_step

        n_steps = 300_000
        offset_step = 3.0 / n_steps  # [bins]
        offsets = -1.5 + (np.arange(n_steps) + 0.5) * offset_step
        distance = np.abs(offsets)
        bspline = np.where(
            distance < 0.5,
            0.75 - distance**2,
            np.where(distance < 1.5, 0.5 * (1.5 - distance) ** 2, 0.0),
        )
        expected = np.array(
            [
                np.sum(
                    np.interp(
                        bin_centre + offsets * bin_step,
                        wake_x[1:],
                        wake_y[1:],
                        left=0.0,
                    )
                    * bspline
                )
                * offset_step
                for bin_centre in query
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wake_per_bin = copy_to_cpu(
                table.get_wake_per_bin(backend.array(query))
            )

        np.testing.assert_allclose(wake_per_bin, expected, atol=1e-6)


class TestResonatorsWakePerBin(BLonDTestCase):
    def test_get_wake_per_bin_counter_rotation_matches_negated(self):
        """A negated counter-rotating shunt gives the negated bin average."""
        res = Resonators(
            np.array([1.0]),
            np.array([1e9]),
            np.array([5.0]),
            shunt_impedances_counter_rotating=np.array([-1.0]),
        )
        time = backend.array(np.arange(64) * 0.05e-9)
        co = copy_to_cpu(res.get_wake_per_bin(time))
        cr = copy_to_cpu(res.get_wake_per_bin_counter_rotation(time))
        np.testing.assert_allclose(co, -cr, atol=1e-12 * np.max(np.abs(co)))

    def test_get_wake_per_bin_counter_rotation_needs_shunt_impedances(self):
        """Without counter-rotating shunts there is no wake to average."""
        res = Resonators(np.array([1.0]), np.array([1e9]), np.array([5.0]))
        time = backend.array(np.arange(8) * 0.05e-9)
        with self.assertRaisesRegex(RuntimeError, "needs to be set"):
            res.get_wake_per_bin_counter_rotation(time)

    def test_get_wake_per_bin_is_exact_bin_average(self):
        """The closed form equals a fine numerical B-spline average.

        The wake is heavily undersampled here (f_res * dt = 0.18), so the
        bin average genuinely differs from point-sampling the wake.
        """
        local_res = Resonators(
            shunt_impedances=np.array([100.0]),
            center_frequencies=np.array([1.0e9]),
            quality_factors=np.array([1.0]),
        )
        dt = 0.18e-9
        time = backend.array(np.arange(256) * dt)

        # Quadratic B-spline over (-1.5 dt, 1.5 dt), fine midpoint rule.
        n_sub = 6000
        shifts = ((np.arange(n_sub) + 0.5) / n_sub * 3.0 - 1.5) * dt
        scaled = np.abs(shifts) / dt
        weights = np.where(
            scaled <= 0.5, 0.75 - scaled**2, 0.5 * (1.5 - scaled) ** 2
        )
        wake_avg = backend.zeros(len(time), dtype=backend.float)
        for shift, weight in zip(shifts, weights, strict=True):
            wake_avg = wake_avg + weight * local_res.get_wake(time + shift)
        wake_avg = copy_to_cpu(wake_avg / np.sum(weights))

        binned = copy_to_cpu(local_res.get_wake_per_bin(time))
        peak = np.max(np.abs(wake_avg))
        np.testing.assert_allclose(
            binned, wake_avg, rtol=2e-3, atol=2e-3 * peak
        )
        self.assertGreater(
            np.max(np.abs(binned - copy_to_cpu(local_res.get_wake(time)))),
            0.1 * peak,
        )

    def test_get_impedance_from_wake_is_exact_bin_average(self):
        """The impedance from the wake is the FFT of the bin-averaged wake.

        The kernel has one non-causal tap, so the wake is sampled from one
        bin before the axis and the spectrum advanced by one sample again.
        """
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        local_res = Resonators(
            shunt_impedances=np.array([100.0]),
            center_frequencies=np.array([1.0e9]),
            quality_factors=np.array([1.0]),
        )
        dt = 0.18e-9  # f_res * dt = 0.18 -> heavily undersampled wake
        time = backend.array(np.arange(256) * dt)
        n_fft = 512

        shifted_time = time - dt
        n_sub = 6000
        shifts = ((np.arange(n_sub) + 0.5) / n_sub * 3.0 - 1.5) * dt
        scaled = np.abs(shifts) / dt
        weights = np.where(
            scaled <= 0.5, 0.75 - scaled**2, 0.5 * (1.5 - scaled) ** 2
        )
        wake_avg = backend.zeros(len(time), dtype=backend.float)
        for shift, weight in zip(shifts, weights, strict=True):
            wake_avg = wake_avg + weight * local_res.get_wake(
                shifted_time + shift
            )
        wake_avg = wake_avg / np.sum(weights)
        expected = copy_to_cpu(backend.fft.rfft(wake_avg, n=n_fft)) * np.exp(
            2j * np.pi * np.arange(n_fft // 2 + 1) / n_fft
        )

        actual = copy_to_cpu(
            local_res.get_impedance_from_wake(
                time=time, simulation=simulation, beam=beam, n_fft=n_fft
            )
        )
        peak = np.max(np.abs(expected))
        np.testing.assert_allclose(
            actual, expected, rtol=2e-3, atol=2e-3 * peak
        )


class TestTravelingWaveCavityWakePerBin(BLonDTestCase):
    @staticmethod
    def _exact_wake_per_bin(twc, time, bin_step, n_nodes=60001):
        """Quadratic B-spline average of the point wake, by quadrature."""
        offsets = np.linspace(-1.5, 1.5, n_nodes)
        distance = np.abs(offsets)
        weights = np.where(
            distance < 0.5,
            0.75 - distance**2,
            np.where(distance < 1.5, 0.5 * (1.5 - distance) ** 2, 0.0),
        )
        return np.array(
            [
                np.trapezoid(
                    copy_to_cpu(
                        twc.wake_calc(backend.array(t + offsets * bin_step))
                    )
                    * weights,
                    offsets,
                )
                for t in copy_to_cpu(time)
            ]
        )

    def test_get_wake_per_bin_resolves_the_onset_jump(self):
        """The bin average must integrate the jump at ``t = 0`` exactly.

        Smoothing the jump into a one-bin ramp leaves an error of order
        the jump itself at the bins around it, which does not shrink with
        the bin width. Covers an onset on a bin centre and between two.
        """
        twc = TravelingWaveCavity(1e5, 200e6, 1e-6)
        onset_jump = 4 * 1e5 / (1e-6 / (2 * np.pi))
        for bin_step in (1e-10, 1e-11):
            for shift in (0.0, 0.3):
                with self.subTest(bin_step=bin_step, shift=shift):
                    time = backend.array((np.arange(-3, 6) + shift) * bin_step)
                    np.testing.assert_allclose(
                        copy_to_cpu(twc.get_wake_per_bin(time)),
                        self._exact_wake_per_bin(twc, time, bin_step),
                        rtol=0,
                        atol=2e-2 * onset_jump * (bin_step / 1e-10) ** 2,
                    )

    def test_get_wake_per_bin_is_exact_beyond_the_bin_nyquist(self):
        """The bin average must be exact at any frequency, not only low ones.

        Sampling the wake at the bin width and smoothing it with the
        5-point stencil cannot represent an oscillation the bins do not
        resolve: at 8 GHz with 100 ps bins that was off by about 30 times
        the exact average. Covers a mode below, near and far above the bin
        Nyquist frequency of 5 GHz, a filling time shorter than the kernel
        (``a_tilde < 1.5 * dt``) and several modes at once.
        """
        bin_step = 1e-10
        time = np.arange(-4, 60) * bin_step + 0.3 * bin_step
        for frequency, a_factor in (
            ([200e6], [2 * np.pi * 3e-9]),
            ([2e9], [2 * np.pi * 2e-9]),
            ([8e9], [2 * np.pi * 1e-9]),
            ([3e9], [2 * np.pi * 0.12e-9]),
            ([200e6, 8e9], [2 * np.pi * 3e-9, 2 * np.pi * 1e-9]),
        ):
            with self.subTest(frequency=frequency):
                twc = TravelingWaveCavity(
                    np.full(len(frequency), 1e5), frequency, a_factor
                )
                a_tilde = np.array(a_factor) / (2 * np.pi)
                frequency_R = copy_to_cpu(twc.frequency_R)

                def wake(t, frequency_R=frequency_R, a_tilde=a_tilde):
                    out = np.zeros_like(t)
                    for i in range(len(a_tilde)):
                        inside = (t > 0) & (t <= a_tilde[i])
                        out += np.where(
                            inside,
                            4e5
                            / a_tilde[i]
                            * (1 - t / a_tilde[i])
                            * np.cos(2 * np.pi * frequency_R[i] * t),
                            0.0,
                        )
                    return out

                expected = _gauss_legendre_bspline_average(
                    wake, time, bin_step, (0.0, *a_tilde)
                )
                np.testing.assert_allclose(
                    copy_to_cpu(twc.get_wake_per_bin(backend.array(time))),
                    expected,
                    rtol=0,
                    atol=1e-10 * np.max(np.abs(expected)),
                )

    def test_get_impedance_from_wake_follows_parameter_changes(self):
        """Changing a public parameter must not return a stale spectrum."""
        twc = TravelingWaveCavity(1e5, 200e6, 1e-6)
        time = backend.arange(16, dtype=backend.float) * 1e-10
        before = copy_to_cpu(
            twc.get_impedance_from_wake(time, None, None, n_fft=32)
        )
        twc.R_S = 2 * twc.R_S

        after = copy_to_cpu(
            twc.get_impedance_from_wake(time, None, None, n_fft=32)
        )

        np.testing.assert_allclose(after, 2 * before, rtol=1e-12)
