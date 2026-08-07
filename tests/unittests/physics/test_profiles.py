# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import unittest

import numpy as np
import pytest

from blond import (
    AllowPlotting,
    Beam,
    Cupy64Bit,
    backend,
    uranium_29,
)
from blond.acc_math.empiric.empiric import gauss_fit, multi_gauss_fit
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    ProfileBaseClass,
    StaticProfile,
)


class TestProfileBaseClass(unittest.TestCase):
    def setUp(self):
        self.profile_base_class = ProfileBaseClass()
        self.profile_base_class._hist_x = backend.linspace(-5, 5, 11)
        self.profile_base_class._hist_y = backend.linspace(5, 5, 11)

    def test___init__(self):
        pass

    def test_on_init_simulation(self):
        from blond.testing.mocks import simulation_mock

        self.profile_base_class.on_init_simulation(simulation=simulation_mock)

    def test_on_run_simulation(self):
        from blond.testing.mocks import beam_mock, simulation_mock

        self.profile_base_class.on_run_simulation(
            simulation=simulation_mock,
            beam=beam_mock,
            n_turns=1,
        )

    def test_plot(self):
        self.profile_base_class.plot()

    def test_hist_x(self):
        self.assertIsNotNone(self.profile_base_class.hist_x)

    def test_hist_y(self):
        self.assertIsNotNone(self.profile_base_class.hist_y)

    def test_n_bins(self):
        self.assertEqual(11, self.profile_base_class.n_bins)

    def test_diff_hist_y(self):
        self.assertEqual(11, len(self.profile_base_class.gradient_hist_y))

    def test_hist_step(self):
        self.assertEqual(1, self.profile_base_class.hist_step)

    def test_cut_left(self):
        self.assertEqual(-5.5, self.profile_base_class.cut_left)

    def test_cut_right(self):
        self.assertEqual(5.5, self.profile_base_class.cut_right)

    def test_bin_edges(self):
        with AllowPlotting():
            np.testing.assert_almost_equal(
                np.linspace(-5.5, 5.5, 12),
                copy_to_cpu(self.profile_base_class.bin_edges),
            )

    def test_track(self):
        from blond.testing.mocks import beam_mock

        with self.assertRaises(NotImplementedError):
            self.profile_base_class.track(beam=beam_mock)

    def test_track_empty_beam_zeros_hist(self):
        from unittest.mock import Mock

        from blond import Beam

        beam = Mock(Beam)
        beam.is_distributed = False
        beam.common_array_size = 0

        self.profile_base_class._hist_y[:] = 1.0
        self.profile_base_class.track(beam=beam)

        np.testing.assert_array_equal(
            copy_to_cpu(self.profile_base_class._hist_y),
            np.zeros(self.profile_base_class.n_bins),
        )
        self.assertEqual(self.profile_base_class.hist_y_to_density_factor, 0.0)

    def test_get_arrays(self):
        self.profile_base_class.get_arrays(
            cut_left=-5.5,
            cut_right=5.5,
            n_bins=11,
        )

    def test_cutoff_frequency(self):
        self.assertEqual(
            1 / (2 * self.profile_base_class.hist_step),
            self.profile_base_class.cutoff_frequency,
        )

    @unittest.skip("Not Implemented")
    def test__calc_gauss(self):
        self.profile_base_class._calc_gauss()

    @unittest.skip("Not Implemented")
    def test_gauss_fit_params(self):
        self.profile_base_class.gauss_fit_params()

    def test_beam_spectrum(self):
        beam_spectrum = self.profile_base_class.beam_spectrum(n_fft=None)
        with AllowPlotting():
            np.testing.assert_almost_equal(
                copy_to_cpu(beam_spectrum),
                np.fft.rfft(copy_to_cpu(self.profile_base_class.hist_y)),
            )

    def test_invalidate_cache(self):
        self.profile_base_class.invalidate_cache()

    def test_weighted_avg_dt(self):
        result = self.profile_base_class.weighted_avg_dt()
        expected = backend.average(
            self.profile_base_class.hist_x,
            weights=(self.profile_base_class.hist_y),
        )
        self.assertAlmostEqual(result, expected)

    def test_sigma_weighted_avg_dt(self):
        result = self.profile_base_class.sigma_weighted_avg_dt()
        average = backend.average(
            self.profile_base_class.hist_x,
            weights=(self.profile_base_class.hist_y),
        )
        variance = backend.average(
            (self.profile_base_class.hist_x - average) ** 2,
            weights=(self.profile_base_class.hist_y),
        )
        expected = backend.sqrt(variance)
        np.testing.assert_almost_equal(result, expected)

    def test_singlebunch_gauss_fit(self):
        result = self.profile_base_class.singlebunch_gauss_fit()
        with AllowPlotting():
            expected = gauss_fit(
                copy_to_cpu(self.profile_base_class.hist_x),
                copy_to_cpu(self.profile_base_class.hist_y),
            )
        np.testing.assert_allclose(result, expected)

    def test_multibunch_gauss_fit(self):
        result = self.profile_base_class.multibunch_gauss_fit(n_bunches=1)
        with AllowPlotting():
            expected = multi_gauss_fit(
                copy_to_cpu(self.profile_base_class.hist_x),
                copy_to_cpu(self.profile_base_class.hist_y),
                n_bunches=1,
            )
        np.testing.assert_allclose(result[0, :], expected[0, :])

    @pytest.mark.backend_mutation
    @pytest.mark.cupy
    def test_singlebunch_gauss_fit_gpu(self):
        try:
            import cupy as cp
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        backend.change_backend(Cupy64Bit)
        profile_base_class = ProfileBaseClass()
        profile_base_class._hist_x = backend.linspace(-5, 5, 11)
        profile_base_class._hist_y = backend.linspace(5, 5, 11)
        result = profile_base_class.singlebunch_gauss_fit()
        with AllowPlotting():
            expected = gauss_fit(
                copy_to_cpu(profile_base_class.hist_x),
                copy_to_cpu(profile_base_class.hist_y),
            )
        np.testing.assert_allclose(result, expected)

    @pytest.mark.backend_mutation
    @pytest.mark.cupy
    def test_multibunch_gauss_fit_gpu(self):
        try:
            import cupy as cp
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        backend.change_backend(Cupy64Bit)
        profile_base_class = ProfileBaseClass()
        profile_base_class._hist_x = backend.linspace(-5, 5, 11)
        profile_base_class._hist_y = backend.linspace(5, 5, 11)
        result = profile_base_class.multibunch_gauss_fit(n_bunches=1)
        with AllowPlotting():
            expected = multi_gauss_fit(
                copy_to_cpu(profile_base_class.hist_x),
                copy_to_cpu(profile_base_class.hist_y),
                n_bunches=1,
            )
        np.testing.assert_allclose(result[0, :], expected[0, :])


class TestStaticProfile(unittest.TestCase):
    def setUp(self):
        self.static_profile = StaticProfile(
            cut_left=-5.5,
            cut_right=5.5,
            n_bins=11,
            section_index=0,
            name="test",
        )

    def test___init__(self):
        pass

    def test_from_cutoff(self):
        profile = StaticProfile.from_cutoff(
            cut_left=-5.5,
            cut_right=5.5,
            cutoff_frequency=1.0 / 2.0,
        )
        self.assertEqual(11, len(profile.hist_x))

    def test_from_rad(self):
        profile = StaticProfile.from_rad(
            cut_left_rad=-np.pi,
            cut_right_rad=np.pi,
            n_bins=11,
            t_period=11,
        )
        np.testing.assert_almost_equal(
            copy_to_cpu(profile.hist_x),
            np.linspace(-5, 5, 11),
        )


class TestDynamicProfileConstCutoff(unittest.TestCase):
    def setUp(self):
        self.dynamic_profile_const_cutoff = DynamicProfileConstCutoff(
            timestep=0.1e-9,
            section_index=0,
            name="test",
        )

    def test___init__(self):
        pass

    def test_update_attributes(self):
        beam = Beam(
            intensity=1,
            particle_type=uranium_29,
        )
        beam.setup_beam(
            dt=np.linspace(0, 1e-9, 10),
            dE=np.linspace(0, 1e9, 10),
            reference_time=0,
            reference_total_energy=450e9,
        )
        self.dynamic_profile_const_cutoff.update_attributes(beam=beam)
        self.assertEqual(10, self.dynamic_profile_const_cutoff.n_bins)
        np.testing.assert_almost_equal(
            np.linspace(0 + 0.05e-9, 0 - 0.05e-9, 10),
            copy_to_cpu(self.dynamic_profile_const_cutoff.hist_x),
        )
        np.testing.assert_almost_equal(
            np.zeros(10),
            copy_to_cpu(self.dynamic_profile_const_cutoff.hist_y),
        )


class TestDynamicProfileConstNBins(unittest.TestCase):
    def setUp(self):
        self.dynamic_profile_const_cutoff = DynamicProfileConstNBins(
            n_bins=10,
            section_index=0,
            name="test",
        )

    def test___init__(self):
        pass

    def test_update_attributes(self):
        beam = Beam(
            intensity=1,
            particle_type=uranium_29,
        )
        beam.setup_beam(
            dt=np.linspace(0, 1e-9, 10),
            dE=np.linspace(0, 1e9, 10),
            reference_time=0,
            reference_total_energy=450e9,
        )
        self.dynamic_profile_const_cutoff.update_attributes(beam=beam)
        self.assertEqual(10, self.dynamic_profile_const_cutoff.n_bins)
        np.testing.assert_almost_equal(
            np.linspace(0 + 0.05e-9, 0 - 0.05e-9, 10),
            copy_to_cpu(self.dynamic_profile_const_cutoff.hist_x),
        )
        np.testing.assert_almost_equal(
            np.zeros(10),
            copy_to_cpu(self.dynamic_profile_const_cutoff.hist_y),
        )


class TestProfileWindowFitsInSpan(unittest.TestCase):
    """
    The single profile-window-vs-span guard on ProfileBaseClass.

    One check for every consumer that has to place the profile window
    inside a time span it does not control. Two consumers exist, and the
    span means the same thing for both: the interval between two
    consecutive passages of the consuming element.

    * A re-binning consumer (the cavity feedback's coarse grid) folds the
      window onto a fixed grid covering that interval. A window longer
      than the span puts two parts of the beam onto the same cell and the
      charge of one replaces the other.
    * A per-passage consumer (``MultiPassResonatorSolver``) shifts its
      stored deposits by that interval. A window longer than it overlaps
      the previous deposit, so the same charge is deposited twice and the
      overlap is lost at negative time.

    Both destroy charge, so the guard raises for both.
    """

    def setUp(self):
        """Set up a 5 t_rf profile window."""
        self.t_rf = 1.0e-9
        self.profile = StaticProfile(
            cut_left=0.0, cut_right=5.0 * self.t_rf, n_bins=100
        )

    def test_window_duration(self):
        """The window duration is cut_right - cut_left."""
        np.testing.assert_allclose(
            self.profile.profile_duration, 5.0 * self.t_rf
        )

    def test_window_duration_is_n_bins_times_hist_step(self):
        """
        The window is the outer-edge span, one bin wider than the centres.

        ``cut_left``/``cut_right`` sit half a bin outside the first/last
        bin centre, so
        ``cut_right - cut_left == n_bins * hist_step``, which is exactly
        one ``hist_step`` more than the first-to-last-centre distance
        ``(len(hist_x) - 1) * hist_step``. Pinned because the deleted
        module-level guard used the centre distance instead, making the
        two guards fire one bin apart -- the "one quantity, two names"
        hazard this class now closes.
        """
        for n_bins in (3, 21, 100, 1024):
            with self.subTest(n_bins=n_bins):
                profile = StaticProfile(
                    cut_left=0.0, cut_right=5.0 * self.t_rf, n_bins=n_bins
                )
                np.testing.assert_allclose(
                    profile.profile_duration,
                    profile.n_bins * profile.hist_step,
                    rtol=1e-15,
                )
                centre_span = (
                    len(copy_to_cpu(profile.hist_x)) - 1
                ) * profile.hist_step
                np.testing.assert_allclose(
                    profile.profile_duration - centre_span,
                    profile.hist_step,
                    rtol=1e-12,
                )

    def test_raises_when_window_longer_than_span(self):
        """A window longer than the span raises, naming the span."""
        with self.assertRaises(ValueError) as cm:
            self.profile.check_fits_in_span(
                3.0 * self.t_rf, span_description="the RF segment"
            )
        message = str(cm.exception)
        self.assertIn("longer than", message)
        self.assertIn("RF segment", message)

    def test_accepts_window_shorter_than_span(self):
        """The ordinary case, a window well inside the span, is silent."""
        self.profile.check_fits_in_span(6.0 * self.t_rf)

    def test_accepts_window_equal_to_span(self):
        """
        A window matching the span exactly is legal, not an overlap.

        A full-turn profile checked against exactly one turn must pass:
        ``MultiTurnWake`` builds exactly that geometry
        (``solvers.py``, ``_assert_profile_length_correct``).
        """
        self.profile.check_fits_in_span(5.0 * self.t_rf)

    def test_tolerance_defaults_to_one_bin(self):
        """
        A sub-bin overshoot is discretisation noise and stays silent.

        The window is derived from bin centres, so an equality case can
        miss by a fraction of a bin purely through float arithmetic.
        """
        self.profile.check_fits_in_span(
            5.0 * self.t_rf - 0.5 * self.profile.hist_step
        )

    def test_raises_when_overshoot_exceeds_the_tolerance(self):
        """Beyond the one-bin slack the guard still fires."""
        with self.assertRaises(ValueError):
            self.profile.check_fits_in_span(
                5.0 * self.t_rf - 3.0 * self.profile.hist_step
            )

    def test_message_names_both_durations_and_the_consumer(self):
        """
        The message carries the numbers and who complained.

        Ported from the deleted module-level guard, which took a
        ``consumer`` name so the user could tell which element is
        affected. A per-passage consumer has no ``span_description`` that
        means anything to the user, so the name is what identifies it.
        """
        with self.assertRaises(ValueError) as caught:
            self.profile.check_fits_in_span(
                2.0 * self.t_rf, consumer="MultiPassResonatorSolver"
            )
        message = str(caught.exception)
        self.assertIn("5e-09", message)
        self.assertIn("2e-09", message)
        self.assertIn("MultiPassResonatorSolver", message)

    def test_zero_span_is_not_judged(self):
        """
        A degenerate span is a different failure, reported elsewhere.

        ``span <= 0`` means the consumer has coincident passages (the
        two-beam meeting-azimuth case), which its own guard already
        reports -- this check must not pile a second failure on top.
        Ported from the deleted module-level guard, whose caller relies
        on it.
        """
        self.profile.check_fits_in_span(0.0)

    def test_sentinel_span_below_one_bin_is_not_judged(self):
        """
        An epsilon span carries no passage, so there is nothing to judge.

        Callers that must satisfy a strictly-positive clock assertion on
        a first deposit advance the reference by ``eps``. That is orders
        of magnitude below one bin, so it resolves no passage at all and
        must not be read as a span the window overshoots.
        """
        self.profile.check_fits_in_span(np.finfo(float).eps)

    def test_span_just_above_the_tolerance_is_judged_again(self):
        """
        The escape hatch stops at one bin -- it is not a blanket bypass.

        Pins the boundary of `test_sentinel_span_below_one_bin_is_not_
        judged`: a span above one bin is a real span, so a window longer
        than it must still be rejected.
        """
        with self.assertRaises(ValueError):
            self.profile.check_fits_in_span(1.001 * self.profile.hist_step)


if __name__ == "__main__":
    unittest.main()
