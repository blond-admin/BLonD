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
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    ProfileBaseClass,
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
        hist_x_reads = _CountingReads(profile._hist_x)
        profile._hist_x = hist_x_reads

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
                    hist_x = copy_to_cpu(profile.hist_x)
                    hist_step = float(hist_x[1] - hist_x[0])
                    expected, _ = np.histogram(
                        dt,
                        bins=len(hist_x),
                        range=(
                            float(hist_x[0] - hist_step / 2.0),
                            float(hist_x[-1] + hist_step / 2.0),
                        ),
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


if __name__ == "__main__":
    unittest.main()
