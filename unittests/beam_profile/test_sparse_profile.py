# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Unit-tests for the self.SparseSlices  class.**

:Authors: **Markus Schwarz**, **Lina Valle**
"""

import copy

# General imports
# -----------------
import unittest

import numpy as np

# BLonD imports
# --------------
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import (
    _SparseProfileBaseClass,
    SparseBucket,
    SparseBatch,
)
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


class testProfileClass(unittest.TestCase):
    # Run before every test
    def setUp(self):
        """
        Slicing of the same Gaussian profile using four distinct settings to
        test different features.
        """

        np.random.seed(1984)

        intensity_pb = 1.0e11
        sigma = 0.2e-9  # Gauss sigma, [s]

        n_macroparticles_pb = int(1e4)
        n_bunches = 2

        # --- Ring and RF ----------------------------------------------
        intensity = n_bunches * intensity_pb  # total intensity SPS
        n_turns = 1
        # Ring parameters SPS
        circumference = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

        gamma_transition = 17.95142852  # Q20 Transition gamma
        momentum_compaction = (
            1.0 / gamma_transition**2
        )  # Momentum compaction array

        ring = Ring(
            circumference,
            momentum_compaction,
            sync_momentum,
            Proton(),
            n_turns=n_turns,
        )

        # RF parameters SPS
        harmonic_number = 4620  # harmonic number
        voltage = 3.5e6  # [V]
        phi_offsets = 0

        self.rf_station = RFStation(
            ring, harmonic_number, voltage, phi_offsets, n_rf=1
        )
        t_rf = self.rf_station.t_rf[0, 0]

        bunch_spacing = 5  # RF buckets

        n_macroparticles = n_bunches * n_macroparticles_pb
        self.beam = Beam(ring, n_macroparticles, intensity)

        for bunch in range(n_bunches):
            bunchBeam = Beam(ring, n_macroparticles_pb, intensity_pb)
            bigaussian(
                ring,
                self.rf_station,
                bunchBeam,
                sigma,
                reinsertion=True,
                seed=1984 + bunch,
            )

            self.beam.dt[
                bunch * n_macroparticles_pb : (bunch + 1) * n_macroparticles_pb
            ] = bunchBeam.dt + bunch * bunch_spacing * t_rf
            self.beam.dE[
                bunch * n_macroparticles_pb : (bunch + 1) * n_macroparticles_pb
            ] = bunchBeam.dE

        self.filling_pattern = np.zeros(bunch_spacing * (n_bunches - 1) + 1)
        self.filling_pattern[::bunch_spacing] = 1
        self.profile_length_in_buckets = 1

        # uniform profile

        profile_margin = 0 * t_rf

        t_batch_begin = 0 * t_rf
        t_batch_end = (bunch_spacing * (n_bunches - 1) + 1) * t_rf

        self.n_slices_rf = 32  # number of slices per RF-bucket

        cut_left = t_batch_begin - profile_margin
        cut_right = t_batch_end + profile_margin

        # number of rf-buckets of the self.beam
        # + rf-buckets before the self.beam + rf-buckets after the self.beam
        n_slices = self.n_slices_rf * (
            bunch_spacing * (n_bunches - 1)
            + 1
            + int(np.round((t_batch_begin - cut_left) / t_rf))
            + int(np.round((cut_right - t_batch_end) / t_rf))
        )

        self.uniform_profile = Profile(
            self.beam,
            cut_options=CutOptions(
                cut_left=cut_left, n_slices=n_slices, cut_right=cut_right
            ),
        )
        self.uniform_profile.track()

    def test_inputs(self):
        with self.assertRaises(ValueError):
            _SparseProfileBaseClass(
                self.rf_station,
                self.beam,
                self.n_slices_rf,
                np.concatenate(
                    (
                        self.filling_pattern,
                        np.ones(int(self.rf_station.harmonic[0][0])),
                    ),
                    axis=0,
                ),
                self.profile_length_in_buckets,
            )

        with self.assertRaises(TypeError):
            _SparseProfileBaseClass(
                self.rf_station,
                self.beam,
                self.n_slices_rf,
                self.filling_pattern,
                1.5,
            )

        with self.assertRaises(ValueError):
            _SparseProfileBaseClass(
                self.rf_station,
                self.beam,
                self.n_slices_rf,
                self.filling_pattern,
                int(-5),
            )
        with self.assertWarns(UserWarning):
            _SparseProfileBaseClass(
                self.rf_station,
                self.beam,
                self.n_slices_rf,
                self.filling_pattern,
                self.profile_length_in_buckets,
            )

    def test_WrongTrackingFunction(self):
        sparse_profile = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
            tracker_mode="something horribly wrong",
        )
        with self.assertRaises(RuntimeError):
            sparse_profile.track()

        nonuniform_profile = SparseBucket(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
        )

        self.assertEqual(
            nonuniform_profile.bin_centers_array.shape,
            (2, self.n_slices_rf),
            msg="Wrong shape of bin_centers_array!",
        )

    def test_onebyone(self):
        rtol = 1e-6  # relative tolerance
        atol = 0  # absolute tolerance

        nonuniform_profile = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
            tracker_mode="onebyone",
            do_track_on_init=True,
        )

        for bunch in range(2):
            indices = (
                self.uniform_profile.bin_centers
                > nonuniform_profile.cut_left_array[bunch]
            ) * (
                self.uniform_profile.bin_centers
                < nonuniform_profile.cut_right_array[bunch]
            )

            np.testing.assert_allclose(
                self.uniform_profile.bin_centers[indices],
                nonuniform_profile.bin_centers_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Bins for bunch {bunch} do not agree "
                + 'for tracker_mode="onebyone"',
            )

            np.testing.assert_allclose(
                self.uniform_profile.n_macroparticles[indices],
                nonuniform_profile.n_macroparticles_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Profiles for bunch {bunch} do not agree "
                + 'for tracker_mode="onebyone"',
            )

    def test_Ctracker(self):
        rtol = 1e-6  # relative tolerance
        atol = 0  # absolute tolerance

        nonuniform_profile = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
            tracker_mode="C",
            do_track_on_init=True,
        )

        for bunch in range(2):
            indices = (
                self.uniform_profile.bin_centers
                > nonuniform_profile.cut_left_array[bunch]
            ) * (
                self.uniform_profile.bin_centers
                < nonuniform_profile.cut_right_array[bunch]
            )

            np.testing.assert_allclose(
                self.uniform_profile.bin_centers[indices],
                nonuniform_profile.bin_centers_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Bins for bunch {bunch} do not agree "
                + 'for tracker_mode="C"',
            )

            np.testing.assert_allclose(
                self.uniform_profile.n_macroparticles[indices],
                nonuniform_profile.n_macroparticles_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Profiles for bunch {bunch} do not agree "
                + 'for tracker_mode="C"',
            )

    def test_tracker_consistency(self):
        rtol = 1e-6  # relative tolerance
        atol = 0  # absolute tolerance

        nonuniform_profile_python = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
            tracker_mode="onebyone",
            do_track_on_init=True,
        )

        nonuniform_profile_cpp = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
            tracker_mode="C",
            do_track_on_init=True,
        )
        for bunch in range(2):
            np.testing.assert_allclose(
                nonuniform_profile_python.bin_centers_array[bunch],
                nonuniform_profile_cpp.bin_centers_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Bins for bunch {bunch} do not agree "
                + "for both trackers",
            )

            np.testing.assert_allclose(
                nonuniform_profile_python.n_macroparticles_array[bunch],
                nonuniform_profile_cpp.n_macroparticles_array[bunch],
                rtol=rtol,
                atol=atol,
                err_msg=f"Profiles for bunch {bunch} do not agree "
                + "for both trackers",
            )

    def test_set_additional_cuts(self):
        updated_filling_pattern = np.array([1, 1, 0, 0, 0, 1])
        sparse_profile = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
        )

        sparse_profile_temoin = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            updated_filling_pattern,
            self.profile_length_in_buckets,
        )

        with self.assertRaises(ValueError):
            sparse_profile._set_additional_cuts(
                _updated_filling_pattern=np.ones(
                    len(sparse_profile._filling_pattern) + 1
                )
            )

        additional_filled_buckets = sparse_profile._set_additional_cuts(
            _updated_filling_pattern=updated_filling_pattern
        )

        np.testing.assert_equal(
            additional_filled_buckets, 1, err_msg="Expected 1"
        )
        np.testing.assert_equal(
            sparse_profile._filling_pattern,
            updated_filling_pattern,
        )
        np.testing.assert_equal(
            np.sort(sparse_profile.cut_left_array),
            sparse_profile_temoin.cut_left_array,
        )
        np.testing.assert_equal(
            np.sort(sparse_profile.cut_right_array),
            sparse_profile_temoin.cut_right_array,
        )

    def test_update_profile_lists(self):
        updated_filling_pattern = np.array([1, 1, 0, 0, 0, 1])
        sparse_profile = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            self.filling_pattern,
            self.profile_length_in_buckets,
        )

        sparse_profile_temoin = _SparseProfileBaseClass(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            updated_filling_pattern,
            self.profile_length_in_buckets,
        )

        with self.assertRaises(ValueError):
            sparse_profile._update_profile_lists(_additional_indices=5)

        additional_filled_buckets = sparse_profile._set_additional_cuts(
            _updated_filling_pattern=updated_filling_pattern
        )
        sparse_profile._update_profile_lists(
            _additional_indices=additional_filled_buckets
        )

        np.testing.assert_equal(
            sparse_profile.n_macroparticles_array[0],
            sparse_profile_temoin.n_macroparticles_array[0],
        )

        np.testing.assert_equal(
            sparse_profile.n_macroparticles_array[1],
            sparse_profile_temoin.n_macroparticles_array[-1],
        )

        np.testing.assert_equal(
            sparse_profile.bin_centers_array[0],
            sparse_profile_temoin.bin_centers_array[0],
        )

        np.testing.assert_equal(
            sparse_profile.bin_centers_array[1],
            sparse_profile_temoin.bin_centers_array[-1],
        )

        np.testing.assert_equal(
            sparse_profile._number_of_indices,
            sparse_profile_temoin._number_of_indices,
        )

        np.testing.assert_equal(
            np.sort(sparse_profile.n_macroparticles),
            np.sort(sparse_profile_temoin.n_macroparticles),
        )

        np.testing.assert_equal(
            sparse_profile.n_slices,
            sparse_profile_temoin.n_slices,
        )

        np.testing.assert_equal(
            sparse_profile._bucket_indices,
            sparse_profile_temoin._bucket_indices,
        )

        np.testing.assert_equal(
            np.sort(sparse_profile.bin_centers),
            np.sort(sparse_profile_temoin.bin_centers),
        )

    def test_properties_SparseBucket(self):
        sparse_profile = SparseBucket(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            np.array([0, 1, 0, 0, 0]),
        )
        np.testing.assert_equal(
            sparse_profile.bunch_list,
            np.array([0, 1, 0, 0, 0]),
        )

        np.testing.assert_equal(
            sparse_profile.total_number_of_filled_buckets,
            1,
        )

        np.testing.assert_equal(
            sparse_profile.bunch_indices,
            np.array([-1, 0, -1, -1, -1]),
        )

        # testing the update_bunch_list function
        with self.assertRaises(ValueError):
            sparse_profile.update_bunch_list(
                updated_bunch_list=np.ones(
                    len(sparse_profile._filling_pattern) + 1
                )
            )
        updated_bunch_list = np.array([0, 1, 0, 1, 0])
        sparse_profile.update_bunch_list(updated_bunch_list=updated_bunch_list)

        np.testing.assert_equal(updated_bunch_list, sparse_profile.bunch_list)

    def test_properties_SparseBatch(self):
        sparse_profile = SparseBatch(
            self.rf_station,
            self.beam,
            self.n_slices_rf,
            np.array([0, 1, 0, 0, 0]),
            self.profile_length_in_buckets * 2,
        )

        np.testing.assert_equal(
            sparse_profile.batch_list,
            np.array([0, 1, 0, 0, 0]),
        )

        np.testing.assert_equal(
            sparse_profile.number_of_slices_per_bucket,
            self.n_slices_rf / 2,
        )
        np.testing.assert_equal(
            sparse_profile.total_number_of_batches,
            1,
        )

        np.testing.assert_equal(
            sparse_profile.total_number_of_sliced_buckets,
            2,
        )

        np.testing.assert_equal(
            sparse_profile.batch_length,
            2,
        )

        np.testing.assert_equal(
            sparse_profile.batch_indices,
            np.array([-1, 0, -1, -1, -1]),
        )

        # testing the updated_batch_list function

        with self.assertRaises(ValueError):
            sparse_profile.update_batch_list(
                updated_batch_list=np.ones(
                    len(sparse_profile._filling_pattern) + 1
                )
            )
        updated_batch_list = np.array([0, 1, 0, 1, 0])
        sparse_profile.update_batch_list(updated_batch_list=updated_batch_list)

        np.testing.assert_equal(updated_batch_list, sparse_profile.batch_list)


if __name__ == "__main__":
    unittest.main()
