# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import dataclasses
import unittest
import warnings

import numpy as np
import pytest

from blond import (
    AllowPlotting,
    Beam,
    Cupy64Bit,
    backend,
    proton,
    uranium_29,
)
from blond.acc_math.empiric.empiric import gauss_fit, multi_gauss_fit
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    ProfileBaseClass,
    ProfileGeometry,
    StaticProfile,
)
from blond.testing.backend_testing import BLonDTestCase


class _CountingReads:
    """Wraps an array and counts element reads through ``[]``."""

    def __init__(self, array):
        self.array = array
        self.n_reads = 0

    def __getitem__(self, key):
        self.n_reads += 1
        return self.array[key]

    def __len__(self):
        return len(self.array)

    def __getattr__(self, name):  # e.g. shape, dtype
        return getattr(self.array, name)


def _beam_with_dt(dt: np.ndarray) -> Beam:
    beam = Beam(
        intensity=1,
        particle_type=uranium_29,
    )
    beam.setup_beam(
        dt=dt,
        dE=np.zeros_like(dt),
        reference_time=0,
        reference_total_energy=450e9,
    )
    return beam


class TestProfileBaseClass(BLonDTestCase):
    def setUp(self):
        self.profile_base_class = ProfileBaseClass()
        self.profile_base_class._set_window(
            cut_left=-5.5, cut_right=5.5, n_bins=11
        )
        self.profile_base_class.hist_y[:] = 5

    def test___init__(self):
        pass

    def test_geometry_is_read_only(self):
        """Writing one geometry attribute could not update the others
        consistently, so none of them may be writable."""
        for name in ("cut_left", "cut_right", "hist_step", "n_bins"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(self.profile_base_class, name, 1.0)
        with self.assertRaises(AttributeError):
            self.profile_base_class.bin_edges = backend.zeros(12)

    def test_geometry_fields_are_frozen(self):
        """Writing one geometry field could not update the others
        consistently, so the geometry is only replaced as a whole."""
        geometry = self.profile_base_class._geometry
        for name, value in (
            ("hist_x", backend.linspace(-5, 5, 11)),
            ("hist_y", backend.zeros(11)),
            ("cut_left", -4.0),
            ("cut_right", 4.0),
        ):
            with (
                self.subTest(name=name),
                self.assertRaises(dataclasses.FrozenInstanceError),
            ):
                setattr(geometry, name, value)

    def test_geometry_rejects_arrays_of_different_length(self):
        with self.assertRaises(AssertionError):
            ProfileGeometry(
                cut_left=-1.0,
                cut_right=1.0,
                hist_x=backend.zeros(3),
                hist_y=backend.zeros(4),
            )

    def test_geometry_rejects_hist_x_off_the_edges(self):
        """`hist_x` must be the bin centers of the window."""
        with self.assertRaises(AssertionError):
            ProfileGeometry(
                cut_left=-1.0,
                cut_right=1.0,
                hist_x=backend.linspace(-1.0, 1.0, 4),  # edges, not centers
                hist_y=backend.zeros(4),
            )

    def test_geometry_accepts_bin_centers(self):
        ProfileGeometry(
            cut_left=-1.0,
            cut_right=1.0,
            hist_x=backend.array([-0.75, -0.25, 0.25, 0.75]),
            hist_y=backend.zeros(4),
        )

    def test_geometry_rejects_inverted_window(self):
        with self.assertRaises(AssertionError):
            ProfileGeometry(
                cut_left=1.0,
                cut_right=-1.0,
                hist_x=backend.zeros(3),
                hist_y=backend.zeros(3),
            )

    def test_hist_y_writable_in_place(self):
        self.profile_base_class.hist_y[:] = 1.0
        self.profile_base_class.hist_y[:] *= 2.0
        np.testing.assert_array_equal(
            np.full(11, 2.0), copy_to_cpu(self.profile_base_class.hist_y)
        )

    def test_bind_arrays(self):
        hist_x = backend.linspace(-5, 5, 11)
        hist_y = backend.zeros(11)
        self.profile_base_class._bind_arrays(hist_x=hist_x, hist_y=hist_y)
        self.assertIs(hist_x, self.profile_base_class.hist_x)
        self.assertIs(hist_y, self.profile_base_class.hist_y)

    def test_bind_arrays_rejects_other_geometry(self):
        with self.assertRaises(AssertionError):
            self.profile_base_class._bind_arrays(
                hist_x=backend.linspace(-4, 4, 11), hist_y=backend.zeros(11)
            )

    def test_bind_arrays_rejects_hist_x_off_by_a_fraction_of_a_bin(self):
        """In [s], a default `atol` of 1e-8 would hide whole bins."""
        profile = StaticProfile(cut_left=0.0, cut_right=1e-9, n_bins=10)
        hist_x = copy_to_cpu(profile.hist_x)
        hist_x[5] += profile.hist_step / 2  # ends untouched
        with self.assertRaises(AssertionError):
            profile._bind_arrays(
                hist_x=backend.array(hist_x, dtype=backend.float),
                hist_y=backend.zeros(10, dtype=backend.float),
            )

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

    def test_gradient_hist_y_follows_in_place_writes(self):
        """`hist_y` is written in place from outside, e.g. by
        `EquidistantMultiProfile`, so a cached gradient would go stale."""
        profile = self.profile_base_class
        _ = profile.gradient_hist_y
        profile.hist_y[:] = backend.arange(11, dtype=backend.float) ** 2
        np.testing.assert_allclose(
            np.gradient(np.arange(11.0) ** 2, 1.0, edge_order=2),
            copy_to_cpu(profile.gradient_hist_y),
        )

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

        self.profile_base_class.hist_y[:] = 1.0
        self.profile_base_class.track(beam=beam)

        np.testing.assert_array_equal(
            copy_to_cpu(self.profile_base_class.hist_y),
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
        profile_base_class._set_window(cut_left=-5.5, cut_right=5.5, n_bins=11)
        profile_base_class.hist_y[:] = 5
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
        profile_base_class._set_window(cut_left=-5.5, cut_right=5.5, n_bins=11)
        profile_base_class.hist_y[:] = 5
        result = profile_base_class.multibunch_gauss_fit(n_bunches=1)
        with AllowPlotting():
            expected = multi_gauss_fit(
                copy_to_cpu(profile_base_class.hist_x),
                copy_to_cpu(profile_base_class.hist_y),
                n_bunches=1,
            )
        np.testing.assert_allclose(result[0, :], expected[0, :])


class TestStaticProfile(BLonDTestCase):
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

    def test_track_keeps_geometry_cache(self):
        """Tracking only refills `hist_y`; the fixed `hist_x` must not be
        rebuilt every turn.
        """
        beam = Beam(intensity=1, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.linspace(-4, 4, 100),
            dE=np.zeros(100),
            reference_time=0,
            reference_total_energy=450e9,
        )
        hist_x_before = self.static_profile.hist_x

        self.static_profile.track(beam=beam)

        self.assertIs(hist_x_before, self.static_profile.hist_x)

    def test_from_cutoff_whole_number_of_steps(self):
        """3.3 ns / 1.1 ns is 3.0000000000000004, which must stay 3 bins."""
        profile = StaticProfile.from_cutoff(
            cut_left=0.0,
            cut_right=3.3e-9,
            cutoff_frequency=1 / (2 * 1.1e-9),
        )
        self.assertEqual(3, profile.n_bins)

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

    def test_track_does_not_read_hist_x(self):
        """``hist_x`` of a static profile never changes, so ``_track`` must
        not re-derive the geometry from it.

        Every scalar read of ``hist_x`` is a device->host sync on GPU, so
        the reads are counted here as a CPU-side proxy for those syncs.
        """
        profile = self.static_profile
        beam = Beam(
            intensity=1,
            particle_type=uranium_29,
        )
        beam.setup_beam(
            dt=np.linspace(-4, 4, 10),
            dE=np.zeros(10),
            reference_time=0,
            reference_total_energy=450e9,
        )
        geometry = ("cut_left", "cut_right", "hist_step", "n_bins")
        expected = {name: getattr(profile, name) for name in geometry}
        expected_bin_edges = copy_to_cpu(profile.bin_edges)
        hist_x_reads = _CountingReads(profile.hist_x)
        # spy on reads, deliberately bypassing `_set_window`
        profile._geometry = dataclasses.replace(
            profile._geometry, hist_x=hist_x_reads
        )
        hist_x_reads.n_reads = 0  # `__post_init__` checks the ends

        for _ in range(2):
            profile.track(beam=beam)

        self.assertEqual(0, hist_x_reads.n_reads)
        for name in geometry:
            self.assertEqual(expected[name], getattr(profile, name), msg=name)
        np.testing.assert_array_equal(
            expected_bin_edges, copy_to_cpu(profile.bin_edges)
        )
        # hist_y changed, so its gradient must not be stale
        np.testing.assert_allclose(
            np.gradient(
                copy_to_cpu(profile.hist_y),
                expected["hist_step"],
                edge_order=2,
            ),
            copy_to_cpu(profile.gradient_hist_y),
        )


class TestDynamicProfileConstCutoff(BLonDTestCase):
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

    def test_update_attributes_n_bins_exact_multiple_of_timestep(self):
        """A window of exactly 10 timesteps needs 10 bins, not 11 from
        float rounding in ``ceil(width / timestep)``."""
        for dt_start in (0.0, 5e-9, -3e-9, 1e-6):
            beam = _beam_with_dt(np.linspace(dt_start, dt_start + 1e-9, 10))
            self.dynamic_profile_const_cutoff.update_attributes(beam=beam)
            with self.subTest(dt_start=dt_start):
                self.assertEqual(
                    10, len(self.dynamic_profile_const_cutoff.hist_x)
                )


class TestDynamicProfileConstNBins(BLonDTestCase):
    def setUp(self):
        self.dynamic_profile_const_cutoff = DynamicProfileConstNBins(
            n_bins=10,
            section_index=0,
            name="test",
        )

    def test___init__(self):
        pass

    def test_n_bins_fixed_after_init(self):
        """``n_bins`` sizes every window, so changing it would make
        ``n_bins`` and the current arrays disagree."""
        profile = self.dynamic_profile_const_cutoff
        n_bins = profile.n_bins
        with self.assertRaises(AttributeError):
            profile.n_bins = 5
        self.assertEqual(n_bins, profile.n_bins)

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

    def test_track_follows_moving_beam(self):
        """The cuts follow the beam each turn, so the cached geometry of
        the previous turn must not survive a track.
        """
        profile = self.dynamic_profile_const_cutoff
        beam = Beam(intensity=1, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.linspace(0, 1e-9, 10),
            dE=np.linspace(0, 1e9, 10),
            reference_time=0,
            reference_total_energy=450e9,
        )
        profile.track(beam=beam)
        cut_left_before = profile.cut_left

        beam.write_partial_dt()[:] += 1e-9
        profile.track(beam=beam)

        self.assertAlmostEqual(
            profile.cut_left - cut_left_before, 1e-9, delta=1e-15
        )


class TestDynamicProfile(BLonDTestCase):
    @staticmethod
    def _make_profiles():
        return (
            DynamicProfileConstCutoff(timestep=0.1e-9),
            DynamicProfileConstNBins(n_bins=10),
        )

    def test_track_after_reading_geometry_uses_new_window(self):
        """Reading the geometry between two turns (as e.g. a wake solver
        does) must not make the next histogram use the previous turn's
        window or bin count."""
        # second turn is shifted and wider, so window and n_bins change
        turns_dt = (
            np.linspace(0.0, 1e-9, 10),
            np.linspace(5e-9, 7e-9, 10),
        )
        for profile in self._make_profiles():
            for turn_i, dt in enumerate(turns_dt):
                profile.track(beam=_beam_with_dt(dt))
                with self.subTest(profile=type(profile).__name__, turn=turn_i):
                    # `track` leaves the geometry of the current window
                    expected, _ = np.histogram(
                        dt,
                        bins=len(profile.hist_x),
                        range=(profile.cut_left, profile.cut_right),
                    )
                    np.testing.assert_array_equal(
                        expected, copy_to_cpu(profile.hist_y)
                    )
                # fill the cache, as a consumer of the profile would
                _ = profile.cut_left, profile.cut_right, profile.hist_step
                _ = profile.n_bins, profile.bin_edges

    def test_track_counts_edge_particles(self):
        """The window spans ``beam.dt_min`` to ``beam.dt_max``, so the
        particles sitting exactly on the edges must be counted."""
        n_particles = 10
        for profile in self._make_profiles():
            for dt_start in (0.0, 5e-9, -3e-9, 1e-6):
                dt = np.linspace(dt_start, dt_start + 1e-9, n_particles)
                profile.track(beam=_beam_with_dt(dt))
                with self.subTest(
                    profile=type(profile).__name__, dt_start=dt_start
                ):
                    self.assertEqual(
                        n_particles, float(copy_to_cpu(profile.hist_y).sum())
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


class TestProfileCapturesWholeBeam(unittest.TestCase):
    """
    The profile warns when its window does not hold the whole beam.

    Charge outside ``[cut_left, cut_right]`` is dropped by the histogram
    without a trace: every consumer downstream then scales a profile that
    silently carries less charge than the beam does. The profile is the
    only object that can see this -- it owns both the window and the
    histogram -- so the check lives here rather than in any one consumer.

    The warning is latched per profile instance. A profile is tracked
    once per passage, and the condition is a property of the window, not
    of the turn: warning on every call would emit thousands of identical
    messages for one mistake.
    """

    def _beam(self, dt: list[float]) -> Beam:
        """Beam with the given ``dt`` coordinates and no energy offset."""
        beam = Beam(intensity=1e9, particle_type=proton)
        beam.setup_beam(
            dt=np.asarray(dt, dtype=float),
            dE=np.zeros(len(dt), dtype=float),
        )
        return beam

    @staticmethod
    def _capture_warnings(caught: list) -> list[str]:
        """Messages of the capture warning among ``caught``."""
        return [
            str(entry.message)
            for entry in caught
            if "inside the profile window" in str(entry.message)
        ]

    def setUp(self):
        """A window holding [0, 1] s in ten bins."""
        self.profile = StaticProfile(cut_left=0.0, cut_right=1.0, n_bins=10)

    def test_no_warning_when_every_particle_is_inside(self):
        """A window holding the whole beam is silent."""
        beam = self._beam([0.1, 0.5, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.profile._track(beam)
        self.assertEqual([], self._capture_warnings(caught))

    def test_warns_when_a_particle_is_outside(self):
        """A particle beyond cut_right is reported, with the fraction."""
        beam = self._beam([0.1, 0.5, 5.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.profile._track(beam)
        messages = self._capture_warnings(caught)
        self.assertEqual(1, len(messages))
        self.assertIn("0.666667", messages[0])

    def test_warns_when_a_particle_is_below_the_window(self):
        """The check is two-sided: cut_left is guarded as well."""
        beam = self._beam([-3.0, 0.5, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.profile._track(beam)
        self.assertEqual(1, len(self._capture_warnings(caught)))

    def test_warning_is_emitted_only_once_per_profile(self):
        """Ten tracked turns of the same mistake give one warning."""
        beam = self._beam([0.1, 0.5, 5.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(10):
                self.profile._track(beam)
        self.assertEqual(1, len(self._capture_warnings(caught)))

    def test_a_second_profile_warns_independently(self):
        """The latch is per instance, so each bad window is reported."""
        other = StaticProfile(cut_left=0.0, cut_right=1.0, n_bins=10)
        beam = self._beam([0.1, 0.5, 5.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.profile._track(beam)
            other._track(beam)
        self.assertEqual(2, len(self._capture_warnings(caught)))

    def test_empty_beam_does_not_warn(self):
        """
        A beam with no particles is not a window mistake.

        ``_track`` zeroes the histogram and sets the density factor to
        0.0 for an empty beam, which is a captured fraction of zero by
        arithmetic but says nothing about the window.
        """
        beam = self._beam([])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.profile._track(beam)
        self.assertEqual([], self._capture_warnings(caught))


if __name__ == "__main__":
    unittest.main()
