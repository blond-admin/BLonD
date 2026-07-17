# coding: utf8
"""
Test functions to compare standard and sparse profiles.

These test functions have been partially generated with the help of a LLM.

Author:
Lina Valle
"""

import unittest
import warnings

import numpy as np
from scipy.constants import e

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import SparseBatch
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring

# ---------------------------------------------------------------------------
# Shared machine parameters (LHC-like), reused by every test class below.
# These match the parameters used in BLonD's own TestLHCOpenDrive unittest,
# so the reference numbers in TestLHCCavityLoopStandardProfile are directly
# comparable to the values already verified upstream.
# ---------------------------------------------------------------------------
RING_CIRCUMFERENCE = 26658.883  # Machine circumference [m]
SYNCHRONOUS_MOMENTUM = 450e9  # [eV/c]
HARMONIC_NUMBER = 35640
RF_VOLTAGE = 4e6  # [V]
RF_PHASE = 0
GAMMA_TRANSITION = 53.8
MOM_COMPACTION = 1 / GAMMA_TRANSITION**2

N_MACROPARTICLES = int(1e5)
BUNCH_INTENSITY = 1e20
BUNCH_SIGMA_DT = 0.5e-9
# Warning: for a large number of bunches, the bin_size difference between
# the sparse profile and the standard profile induces slight mismatches
# between the indexes. Tests will artificially fail because of this difference.
number_of_batches = 10  # Length of the batch [number of batches]
batch_spacing = 5  # Number of empty buckets between each batch [number of rf
# buckets]

number_of_bunches_per_batch = 3  # number of bunches per batch i.e. per profile
bunch_spacing = 1  # Number of empty buckets between each bunch

total_length_batch = (number_of_bunches_per_batch +
                      (number_of_bunches_per_batch-1) * bunch_spacing)
assert(total_length_batch <= batch_spacing)
if number_of_batches > 50:
    warnings.warn(message="Warning: for a large number of bunches, the bin_size "
                          "difference between the sparse profile and the "
                          "standard profile induces slight mismatches between the indexes. "
                          "Tests might artificially fail because of this "
                          "difference.")

def build_ring_and_rf():
    """Build the Ring/RFStation pair shared by all tests."""
    ring = Ring(
        RING_CIRCUMFERENCE,
        MOM_COMPACTION,
        SYNCHRONOUS_MOMENTUM,
        particle=Proton(),
        n_turns=1,
    )
    rf_station = RFStation(
        ring, [HARMONIC_NUMBER], [RF_VOLTAGE], [RF_PHASE]
    )
    return ring, rf_station


def build_beam(ring, rf_station, seed=1234):
    """Build a Gaussian bunch so that Profile/SparseBatch slicing is
    well defined (an empty/point beam makes bin_size degenerate)."""
    # The beam

    # Beam object for the batch
    N_m = N_MACROPARTICLES * number_of_batches * number_of_bunches_per_batch
    N_p = BUNCH_INTENSITY * number_of_batches * number_of_bunches_per_batch
    beam = Beam(ring, N_m, N_p)
    # First generate a single gaussian bunch
    single_bunch = Beam(ring, N_MACROPARTICLES, BUNCH_INTENSITY)
    bigaussian(
        ring,
        rf_station,
        beam,
        sigma_dt=BUNCH_SIGMA_DT,
        seed=seed,
        reinsertion=True,
    )
    # Copy the bunch throughout the batch

    if number_of_bunches_per_batch >1:
        single_batch = Beam(ring,
                            number_of_bunches_per_batch*N_MACROPARTICLES,
                            number_of_bunches_per_batch*BUNCH_INTENSITY)
        for i in range(number_of_bunches_per_batch):
            single_batch.dE[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] \
                = single_bunch.dE
            single_batch.dt[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] = (
                    single_bunch.dt + i * bunch_spacing * rf_station.t_rf[0, 0]
            )
        for i in range(number_of_batches):
            N_MACROPARTICLES_PER_BATCH = number_of_bunches_per_batch * N_MACROPARTICLES
            beam.dE[i * N_MACROPARTICLES_PER_BATCH: (i + 1) * N_MACROPARTICLES_PER_BATCH] = (
                single_batch.dE
            )
            beam.dt[i * N_MACROPARTICLES_PER_BATCH: (i + 1) * N_MACROPARTICLES_PER_BATCH] = (
                    single_batch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
            )
    else:
        for i in range(number_of_batches):
            beam.dE[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dE
            )
            beam.dt[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] = (
                    single_bunch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
            )
    return beam


def build_standard_profile(beam, rf_station, n_slices):
    """A standard Profile covering the injected bunches and an extra bucket."""
    profile = Profile(
        beam,
        CutOptions(cut_left=0.0, cut_right=(batch_spacing * number_of_batches + 1)
            * rf_station.t_rf[
                0,
                0,
            ],
                   n_slices=n_slices * (batch_spacing * number_of_batches + 1)),
    )
    profile.track()
    return profile


def build_sparse_profile(beam, rf_station, n_slices):
    """A SparseBatch profile with a profile per number of bunches.
    """
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(number_of_batches):
        batch_list[k * batch_spacing] = 1
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(batch_spacing/2)+1) * n_slices,
        batch_list=batch_list,
        batch_length=int(batch_spacing/2)+1,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


class TestComparisonProfiles(unittest.TestCase):
    N_SLICES = 4 * HARMONIC_NUMBER // 5  # fine relative to the coarse (n_coarse) grid
    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_sparse_profile(
            self.beam, self.rf, self.N_SLICES
        )

        self.rtol = 1e-20
        self.atol = 1e-15

    def test_bin_centers_match(self):
        for p, profile in enumerate(
            self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(
                self.profile_std.bin_centers[index],
                self.profile_sparse.bin_centers[p * profile.n_slices],
                rtol = self.rtol, atol =  self.atol,
                err_msg="bin centers differ between "
                                               "standard Profile and SparseBatch for the same beam, profile number "
                                               f"{p}",
                                       )
    def test_histogram_equivalent(self):

        hist_std, bins_std = np.histogram(self.beam.dt,
                                          bins=len(
    self.profile_std.n_macroparticles), range = (self.profile_std.cut_left,
                                                 self.profile_std.cut_right))

        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )

            hist_sparse, bin_sparse = np.histogram(self.beam.dt,
                                                   bins=len(profile.n_macroparticles),
                                                   range = (profile.cut_left, profile.cut_right))
            np.testing.assert_allclose(
                bins_std[index:index + profile.n_slices + 1],
                bin_sparse,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bins for histograms differ between "
                        "standard Profile and SparseBatch for the same beam, profile number "
                        f"{p}",
            )
            np.testing.assert_allclose(
                hist_std[index:index + profile.n_slices],
                hist_sparse,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="histograms differ between "
                        "standard Profile and SparseBatch for the same beam, profile number "
                        f"{p}",
            )
    def test_initial_sparse_n_macroparticles_correct(self):
        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            hist_sparse, bin_sparse = np.histogram(self.beam.dt,
                                                   bins=len(
                                                       profile.n_macroparticles),
                                                   range=(profile.cut_left,
                                                          profile.cut_right))
            np.testing.assert_allclose(hist_sparse,
                                       profile.n_macroparticles,
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       err_msg="n_macroparticles differ "
                                               "between init and computation, profile number "
                                               f"{p}",
                                       )
    def test_n_macroparticles_match(self):
        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(
                self.profile_std.bin_centers[index:index +
                                                        profile.n_slices],
                profile.bin_centers,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bin_centers differ between "
                        "standard Profile and SparseBatch for the same beam, profile number "
                        f"{p}",
                )

            np.testing.assert_allclose(self.profile_std.n_macroparticles[index:index +
                                                         profile.n_slices],
                                       profile.n_macroparticles,
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       err_msg="n_macroparticles differ between "
                                               "standard Profile and SparseBatch for the same beam, profile number "
                                               f"{p}",
                                       )
    def test_bin_size_matches(self):
        self.assertAlmostEqual(self.profile_std.bin_size, self.profile_sparse.bin_size,
                               places=15)
