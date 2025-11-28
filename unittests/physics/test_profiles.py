# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import unittest

import numpy as np
from scipy.stats import norm

from blond import Beam, backend, uranium_29
from blond.acc_math.empiric.empiric import gauss_fit, multi_gauss_fit
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
            turn_i_init=0,
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
        np.testing.assert_almost_equal(
            np.linspace(-5.5, 5.5, 12), self.profile_base_class.bin_edges
        )

    def test_track(self):
        from blond.testing.mocks import beam_mock

        with self.assertRaises(NotImplementedError):
            self.profile_base_class.track(beam=beam_mock)

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
        np.testing.assert_almost_equal(
            beam_spectrum, np.fft.rfft(self.profile_base_class.hist_y)
        )

    def test_invalidate_cache(self):
        self.profile_base_class.invalidate_cache()

    def test_weighted_avg_dt(self):
        result = self.profile_base_class.weighted_avg_dt()
        expected = backend.average(self.profile_base_class.hist_x,
                                   weights=(self.profile_base_class.hist_y))
        self.assertAlmostEqual(result, expected)

    def test_sigma_weighted_avg_dt(self):
        result = self.profile_base_class.sigma_weighted_avg_dt()
        average = backend.average(self.profile_base_class.hist_x,
                                  weights=(self.profile_base_class.hist_y))
        variance = backend.average((self.profile_base_class.hist_x - average) ** 2,
                                   weights=(self.profile_base_class.hist_y))
        expected = backend.sqrt(variance)
        self.assertAlmostEqual(result, expected)

    def test_singlebunch_gauss_fit(self):
        result = self.profile_base_class.singlebunch_gauss_fit()
        expected = gauss_fit(self.profile_base_class.hist_x, self.profile_base_class.hist_y)
        np.testing.assert_almost_equal(result, expected)

    def test_multibunch_gauss_fit(self):
        result = self.profile_base_class.multibunch_gauss_fit(n_bunches =1)
        expected = multi_gauss_fit(self.profile_base_class.hist_x, self.profile_base_class.hist_y, n_bunches =1)
        np.testing.assert_almost_equal(result, expected)


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
            profile.hist_x,
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
            self.dynamic_profile_const_cutoff.hist_x,
        )
        np.testing.assert_almost_equal(
            np.zeros(10),
            self.dynamic_profile_const_cutoff.hist_y,
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
            self.dynamic_profile_const_cutoff.hist_x,
        )
        np.testing.assert_almost_equal(
            np.zeros(10),
            self.dynamic_profile_const_cutoff.hist_y,
        )

if __name__ == "__main__":
    unittest.main()
