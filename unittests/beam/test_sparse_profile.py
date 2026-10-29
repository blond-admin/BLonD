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
from blond.beam.sparse_profiles import SparseBatch, SparseBunch
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

total_length_batch = (
    number_of_bunches_per_batch
    + (number_of_bunches_per_batch - 1) * bunch_spacing
)
assert total_length_batch <= batch_spacing
if number_of_batches > 50:
    warnings.warn(
        message="Warning: for a large number of bunches, the bin_size "
        "difference between the sparse profile and the "
        "standard profile induces slight mismatches between the indexes. "
        "Tests might artificially fail because of this "
        "difference."
    )


def build_ring_and_rf():
    """Build the Ring/RFStation pair shared by all tests."""
    ring = Ring(
        RING_CIRCUMFERENCE,
        MOM_COMPACTION,
        SYNCHRONOUS_MOMENTUM,
        particle=Proton(),
        n_turns=1,
    )
    rf_station = RFStation(ring, [HARMONIC_NUMBER], [RF_VOLTAGE], [RF_PHASE])
    return ring, rf_station


def build_beam(
    ring, rf_station, seed=1234, injected_batches=number_of_batches
):
    """Build a beam of injected_batches batches of Gaussian bunches, so
    that Profile/SparseBatch slicing is well defined (an empty/point beam
    makes bin_size degenerate)."""
    # The beam

    # Beam object for the batch
    N_m = N_MACROPARTICLES * injected_batches * number_of_bunches_per_batch
    N_p = BUNCH_INTENSITY * injected_batches * number_of_bunches_per_batch
    beam = Beam(ring, N_m, N_p)
    # First generate a single gaussian bunch
    single_bunch = Beam(ring, N_MACROPARTICLES, BUNCH_INTENSITY)
    bigaussian(
        ring,
        rf_station,
        single_bunch,
        sigma_dt=BUNCH_SIGMA_DT,
        seed=seed,
        reinsertion=True,
    )
    # Copy the bunch throughout the batch

    if number_of_bunches_per_batch > 1:
        single_batch = Beam(
            ring,
            number_of_bunches_per_batch * N_MACROPARTICLES,
            number_of_bunches_per_batch * BUNCH_INTENSITY,
        )
        for i in range(number_of_bunches_per_batch):
            single_batch.dE[
                i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
            ] = single_bunch.dE
            single_batch.dt[
                i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
            ] = single_bunch.dt + i * bunch_spacing * rf_station.t_rf[0, 0]
        for i in range(injected_batches):
            N_MACROPARTICLES_PER_BATCH = (
                number_of_bunches_per_batch * N_MACROPARTICLES
            )
            beam.dE[
                i * N_MACROPARTICLES_PER_BATCH : (i + 1)
                * N_MACROPARTICLES_PER_BATCH
            ] = single_batch.dE
            beam.dt[
                i * N_MACROPARTICLES_PER_BATCH : (i + 1)
                * N_MACROPARTICLES_PER_BATCH
            ] = single_batch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
    else:
        for i in range(injected_batches):
            beam.dE[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dE
            )
            beam.dt[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
            )
    return beam


def inject_batch(
    beam, ring, rf_station, sparse_profile, injected_batches, seed=1234
):
    """Inject batch number injected_batches into the beam and register it
    in the SparseBatch. Returns the updated number of injected batches.
    """
    batch = build_beam(ring, rf_station, seed, injected_batches=1)
    beam.add_particles(
        [
            batch.dt
            + injected_batches * batch_spacing * rf_station.t_rf[0, 0],
            batch.dE,
        ]
    )
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches + 1):
        batch_list[k * batch_spacing] = 1
    sparse_profile.update_batch_list(updated_batch_list=batch_list)
    return injected_batches + 1


def build_standard_profile(beam, rf_station, n_slices):
    """A standard Profile covering the injected bunches and an extra bucket."""
    profile = Profile(
        beam,
        CutOptions(
            cut_left=0.0,
            cut_right=(batch_spacing * number_of_batches + 1)
            * rf_station.t_rf[
                0,
                0,
            ],
            n_slices=n_slices * (batch_spacing * number_of_batches + 1),
        ),
    )
    profile.track()
    return profile


def build_sparse_profile(
    beam,
    rf_station,
    n_slices,
    tracker_mode="onebyone",
    injected_batches=number_of_batches,
):
    """A SparseBatch profile with a profile per injected batch."""
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
        batch_list[k * batch_spacing] = 1
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(batch_spacing / 2) + 1) * n_slices,
        batch_list=batch_list,
        batch_length=int(batch_spacing / 2) + 1,
        tracker_mode=tracker_mode,
    )
    sparse_profile.track()
    return sparse_profile


# SparseBunch windows: the margin keeps the window of a bunch (rms length
# BUNCH_SIGMA_DT = 0.5 ns, centred in the 2.5 ns bucket) inside its RF
# bucket. The grid is the one of the RF bucket (N_SLICES_PER_BUCKET bins
# per bucket, 2.5 ps); N_SLICES_PER_BUNCH is for the fixed-number-of-bins
# mode.
LENGTH_IN_SIGMA = 2
N_SLICES_PER_BUCKET = 1000
N_SLICES_PER_BUNCH = 2000


def bunch_buckets(injected_batches=number_of_batches):
    """RF bucket of every bunch of build_beam, in the order the bunches were
    appended to the beam (batch after batch, bunch after bunch)."""
    return np.array(
        [
            i * batch_spacing + j * bunch_spacing
            for i in range(injected_batches)
            for j in range(number_of_bunches_per_batch)
        ]
    )


def build_sparse_profile_bunch(
    beam,
    rf_station,
    n_slices_per_bucket=N_SLICES_PER_BUCKET,
    n_slices_per_window=None,
    length_in_sigma=LENGTH_IN_SIGMA,
    tracker_mode=None,
    injected_batches=number_of_batches,
    recenter_every=None,
    warn_outside_fraction=None,
    resize_windows=False,
    resize_tail_fraction=0.1,
    resize_factor=1.5,
):
    """A SparseBunch profile with a window around every bunch, on the RF
    bucket grid unless n_slices_per_window is given; histogrammed in one
    pass ("C") on the grid, window by window otherwise. The outside warning
    and the widening are off by default: a 2 sigma window leaves the tails
    (4.6 %) outside and holds 7 % of its particles in its outer tenths (16 %
    once the bunch is twice as long), hence the 0.1 threshold here."""
    if tracker_mode is None:
        tracker_mode = "onebyone" if n_slices_per_window is not None else "C"
    bunch_list = np.zeros(HARMONIC_NUMBER)
    bunch_list[bunch_buckets(injected_batches)] = 1
    sparse_profile = SparseBunch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=n_slices_per_window,
        bunch_list=bunch_list,
        length_in_sigma=length_in_sigma,
        tracker_mode=tracker_mode,
        number_of_slices_per_bucket=(
            None if n_slices_per_window is not None else n_slices_per_bucket
        ),
        recenter_every=recenter_every,
        warn_outside_fraction=warn_outside_fraction,
        resize_windows=resize_windows,
        resize_tail_fraction=resize_tail_fraction,
        resize_factor=resize_factor,
    )
    sparse_profile.track()
    return sparse_profile


class TestComparisonProfiles(unittest.TestCase):
    N_SLICES = (
        4 * HARMONIC_NUMBER // 5
    )  # fine relative to the coarse (n_coarse) grid

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
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                self.profile_std.bin_centers[index],
                self.profile_sparse.bin_centers[p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bin centers differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )

    def test_histogram_equivalent(self):
        hist_std, bins_std = np.histogram(
            self.beam.dt,
            bins=len(self.profile_std.n_macroparticles),
            range=(self.profile_std.cut_left, self.profile_std.cut_right),
        )

        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )

            hist_sparse, bin_sparse = np.histogram(
                self.beam.dt,
                bins=len(profile.n_macroparticles),
                range=(profile.cut_left, profile.cut_right),
            )
            np.testing.assert_allclose(
                bins_std[index : index + profile.n_slices + 1],
                bin_sparse,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bins for histograms differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )
            np.testing.assert_allclose(
                hist_std[index : index + profile.n_slices],
                hist_sparse,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="histograms differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )

    def test_initial_sparse_n_macroparticles_correct(self):
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            hist_sparse, bin_sparse = np.histogram(
                self.beam.dt,
                bins=len(profile.n_macroparticles),
                range=(profile.cut_left, profile.cut_right),
            )
            np.testing.assert_allclose(
                hist_sparse,
                profile.n_macroparticles,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="n_macroparticles differ "
                "between init and computation, profile number "
                f"{p}",
            )

    def test_n_macroparticles_match(self):
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                self.profile_std.bin_centers[index : index + profile.n_slices],
                profile.bin_centers,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bin_centers differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )

            np.testing.assert_allclose(
                self.profile_std.n_macroparticles[
                    index : index + profile.n_slices
                ],
                profile.n_macroparticles,
                rtol=self.rtol,
                atol=self.atol,
                err_msg="n_macroparticles differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )

    def test_bin_size_matches(self):
        self.assertAlmostEqual(
            self.profile_std.bin_size, self.profile_sparse.bin_size, places=15
        )

    def _assert_histograms_match(self, profile_sparse, profile_std=None):
        """_histogram_c (bm.sparse_histogram, all the windows at once),
        _histogram_one_by_one (Profile.track() window by window)
        and _histogram_one_by_one_numpy (np.histogram window by window) must give
        the same histograms for the same beam and the same windows, and the
        same as the standard profile over the same bins."""
        if profile_std is None:
            profile_std = self.profile_std
        profile_sparse._histogram_c()
        n_macroparticles_c = np.copy(profile_sparse.n_macroparticles_array)
        profile_sparse._histogram_one_by_one()
        n_macroparticles_one_by_one = np.copy(
            profile_sparse.n_macroparticles_array
        )
        profile_sparse._histogram_one_by_one_numpy()
        n_macroparticles_numpy = np.copy(profile_sparse.n_macroparticles_array)
        for p, profile in enumerate(profile_sparse.profiles_list):
            np.testing.assert_allclose(
                n_macroparticles_c[p],
                n_macroparticles_numpy[p],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="histograms differ between _histogram_c and "
                "_histogram_one_by_one_numpy for the same beam, profile number "
                f"{p}",
            )
            np.testing.assert_allclose(
                n_macroparticles_one_by_one[p],
                n_macroparticles_numpy[p],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="histograms differ between "
                "_histogram_one_by_one and "
                "_histogram_one_by_one_numpy for the same beam, profile number "
                f"{p}",
            )
            index = np.argmin(
                np.abs(profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                profile_std.n_macroparticles[index : index + profile.n_slices],
                n_macroparticles_numpy[p],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="histograms differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )

    def test_histogram_c_matches_one_by_one(self):
        profile_sparse = build_sparse_profile(
            self.beam, self.rf, self.N_SLICES, tracker_mode="C"
        )
        self._assert_histograms_match(profile_sparse)

    def test_multiturn_injection_histograms_match(self):
        """The same comparison after each injected batch: the beam grows and
        the SparseBatch gets a new profile per injection."""
        injected_batches = 1
        beam = build_beam(
            self.ring, self.rf, injected_batches=injected_batches
        )
        profile_std = build_standard_profile(beam, self.rf, self.N_SLICES)
        profile_sparse = build_sparse_profile(
            beam,
            self.rf,
            self.N_SLICES,
            tracker_mode="C",
            injected_batches=injected_batches,
        )
        while injected_batches < number_of_batches:
            injected_batches = inject_batch(
                beam, self.ring, self.rf, profile_sparse, injected_batches
            )
            profile_std.track()
            with self.subTest(injected_batches=injected_batches):
                self._assert_histograms_match(profile_sparse, profile_std)


class TestSparseBunch(unittest.TestCase):
    """SparseBunch: one window per filled bucket, centred on the bunch the
    bucket holds and length_in_sigma rms bunch lengths wide on each side,
    histogrammed window by window."""

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.profile = build_sparse_profile_bunch(self.beam, self.rf)
        self.buckets = bunch_buckets()
        self.t_rf = self.rf.t_rf[0, 0]
        self.rtol = 1e-20
        self.atol = 1e-15

    def bunch_dt(self, beam, k):
        """dt of the particles of bunch k: the k-th block appended to the
        beam, as the k-th window of the profile."""
        return beam.dt[k * N_MACROPARTICLES : (k + 1) * N_MACROPARTICLES]

    def bucket_dt(self, beam, bucket):
        """dt of the particles found in RF bucket `bucket`: the particles
        the window is built from. The tails of a 0.5 ns bunch reach into
        the neighbouring 2.5 ns bucket (about 0.6 % of its particles beyond
        2.5 rms lengths) and are attributed to that bucket."""
        dt = beam.dt
        return dt[(dt >= bucket * self.t_rf) & (dt < (bucket + 1) * self.t_rf)]

    def assert_windows_follow_the_bunches(
        self, beam, profile, buckets, length_in_sigma
    ):
        """Every window has the common width, 2 length_in_sigma rms lengths
        of the longest bunch (rounded up to whole bins on the bucket grid),
        the common bin size, and is centred on the centroid of the
        particles of its bucket (to half a bin on the bucket grid)."""
        sigmas = [np.std(self.bucket_dt(beam, b)) for b in buckets]
        expected_width = 2 * length_in_sigma * max(sigmas)
        bin_size = profile.window_bin_size
        on_grid = profile.number_of_slices_per_bucket is not None
        if on_grid:
            np.testing.assert_allclose(
                bin_size, self.t_rf / profile.number_of_slices_per_bucket,
                rtol=1e-12)
            self.assertGreaterEqual(
                profile.window_width, expected_width * (1 - 1e-12))
            self.assertLess(
                profile.window_width - expected_width, bin_size * (1 + 1e-9))
            centre_tolerance = 0.5 * bin_size * (1 + 1e-9)
        else:
            np.testing.assert_allclose(
                profile.window_width, expected_width, rtol=1e-12)
            centre_tolerance = expected_width * 1e-12
        for k, (bucket, window) in enumerate(
            zip(buckets, profile.profiles_list)
        ):
            center = np.mean(self.bucket_dt(beam, bucket))
            with self.subTest(bunch=k):
                np.testing.assert_allclose(
                    window.cut_right - window.cut_left,
                    profile.window_width,
                    rtol=1e-12,
                    err_msg=f"width of window {k}",
                )
                np.testing.assert_allclose(
                    window.bin_size,
                    bin_size,
                    rtol=1e-12,
                    err_msg=f"bin size of window {k}",
                )
                self.assertLessEqual(
                    abs(0.5 * (window.cut_left + window.cut_right) - center),
                    centre_tolerance,
                    msg=f"window {k} is not centred on its bunch",
                )
                if on_grid:
                    lattice = window.cut_left / bin_size
                    self.assertAlmostEqual(
                        lattice, round(lattice), delta=1e-6,
                        msg=f"window {k} is off the bin lattice")

    def assert_histograms_match_numpy(self, beam, profile):
        """_histogram_one_by_one (Profile.track per window) and
        _histogram_one_by_one_numpy must agree with np.histogram over the
        window, and count the particles of the bunch inside the window."""
        profile._histogram_one_by_one()
        one_by_one = np.copy(profile.n_macroparticles_array)
        profile._histogram_one_by_one_numpy()
        with_numpy = np.copy(profile.n_macroparticles_array)
        for k, window in enumerate(profile.profiles_list):
            hist, _ = np.histogram(
                beam.dt,
                bins=window.n_slices,
                range=(window.cut_left, window.cut_right),
            )
            dt = self.bunch_dt(beam, k)
            own_inside = int(
                np.sum((dt >= window.cut_left) & (dt <= window.cut_right))
            )
            with self.subTest(bunch=k):
                np.testing.assert_allclose(
                    one_by_one[k],
                    hist,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="_histogram_one_by_one differs from "
                    f"np.histogram, window {k}",
                )
                np.testing.assert_allclose(
                    with_numpy[k],
                    hist,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="_histogram_one_by_one_numpy differs from "
                    f"np.histogram, window {k}",
                )
                # the window holds the particles of its own bunch plus, at
                # most, the far tails of the neighbouring bunches
                self.assertGreaterEqual(int(hist.sum()), own_inside)
                self.assertLessEqual(
                    int(hist.sum()) - own_inside, 0.01 * N_MACROPARTICLES
                )

    def test_one_window_per_bunch(self):
        n_bunches = len(self.buckets)
        self.assertEqual(len(self.profile.profiles_list), n_bunches)
        self.assertEqual(
            self.profile.total_number_of_filled_buckets, n_bunches
        )
        self.assertEqual(
            self.profile.n_slices,
            self.profile.number_of_slices_per_profile * n_bunches,
        )
        # 2 sigma = 2 ns of 2.5 ps bins
        self.assertEqual(self.profile.number_of_slices_per_profile,
                         int(np.ceil(self.profile.window_width
                                     / self.profile.window_bin_size - 1e-9)))
        np.testing.assert_array_equal(
            np.flatnonzero(self.profile.bunch_list), np.sort(self.buckets)
        )

    def test_windows_follow_the_bunches(self):
        self.assert_windows_follow_the_bunches(
            self.beam, self.profile, self.buckets, LENGTH_IN_SIGMA
        )

    def test_windows_inside_the_rf_bucket(self):
        """With this margin the window of a bunch stays in its RF bucket, so
        no particle of a neighbouring bunch enters it."""
        for k, (bucket, window) in enumerate(
            zip(self.buckets, self.profile.profiles_list)
        ):
            with self.subTest(bunch=k):
                self.assertGreaterEqual(window.cut_left, bucket * self.t_rf)
                self.assertLessEqual(
                    window.cut_right, (bucket + 1) * self.t_rf
                )

    def test_histograms_match_numpy(self):
        self.assert_histograms_match_numpy(self.beam, self.profile)

    def test_window_holds_the_core_of_the_bunch(self):
        """length_in_sigma rms lengths on each side of a Gaussian bunch hold
        erf(length_in_sigma / sqrt(2)) of its particles (95.4 % for 2)."""
        from scipy.special import erf

        expected = erf(LENGTH_IN_SIGMA / np.sqrt(2))
        for k, window in enumerate(self.profile.profiles_list):
            dt = self.bunch_dt(self.beam, k)
            with self.subTest(bunch=k):
                fraction = np.mean(
                    (dt >= window.cut_left) & (dt <= window.cut_right)
                )
                self.assertAlmostEqual(fraction, expected, delta=0.01)

    def test_margin_scales_with_length_in_sigma(self):
        wider = build_sparse_profile_bunch(
            self.beam, self.rf, length_in_sigma=2 * LENGTH_IN_SIGMA
        )
        self.assert_windows_follow_the_bunches(
            self.beam, wider, self.buckets, 2 * LENGTH_IN_SIGMA
        )
        bin_size = self.profile.window_bin_size
        # twice the width, up to the rounding to whole bins
        self.assertLessEqual(
            abs(wider.window_width - 2 * self.profile.window_width), 2 * bin_size)
        for k, (narrow, wide) in enumerate(
            zip(self.profile.profiles_list, wider.profiles_list)
        ):
            with self.subTest(bunch=k):
                # same centre, up to the snapping of both to the lattice
                self.assertLessEqual(
                    abs((wide.cut_left + wide.cut_right)
                        - (narrow.cut_left + narrow.cut_right)),
                    2 * bin_size,
                )

    def test_fixed_number_of_bins_per_window(self):
        """number_of_slices_per_profile instead of the bucket grid: the
        windows have that many bins and the bin size follows from the
        common width."""
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, n_slices_per_window=N_SLICES_PER_BUNCH)
        self.assertIsNone(profile.number_of_slices_per_bucket)
        self.assertEqual(profile.number_of_slices_per_profile, N_SLICES_PER_BUNCH)
        np.testing.assert_allclose(
            profile.window_bin_size, profile.window_width / N_SLICES_PER_BUNCH,
            rtol=1e-12)
        self.assert_windows_follow_the_bunches(
            self.beam, profile, self.buckets, LENGTH_IN_SIGMA)
        self.assert_histograms_match_numpy(self.beam, profile)

    def test_grid_arguments_are_exclusive(self):
        bunch_list = np.zeros(HARMONIC_NUMBER)
        bunch_list[self.buckets] = 1
        with self.assertRaises(ValueError):
            SparseBunch(rf_station=self.rf, beam=self.beam,
                        bunch_list=bunch_list, tracker_mode="onebyone")
        with self.assertRaises(ValueError):
            SparseBunch(rf_station=self.rf, beam=self.beam,
                        bunch_list=bunch_list, tracker_mode="onebyone",
                        number_of_slices_per_profile=N_SLICES_PER_BUNCH,
                        number_of_slices_per_bucket=N_SLICES_PER_BUCKET)

    def shift_last_bunch(self, beam, shift):
        """Move the last bunch (no neighbour after it) by `shift` [s]."""
        k = len(self.buckets) - 1
        beam.dt[k * N_MACROPARTICLES : (k + 1) * N_MACROPARTICLES] += shift
        return k

    def test_outside_particles_are_counted(self):
        """n_macroparticles_outside is the number of macroparticles in no
        window: the tails beyond 2 sigma at rest, more once a bunch has
        drifted out of its fixed window."""
        profile = self.profile
        n_total = self.beam.n_macroparticles
        self.assertEqual(
            profile.n_macroparticles_outside,
            n_total - int(round(profile.n_macroparticles.sum())))
        at_rest = profile.n_macroparticles_outside
        self.assertAlmostEqual(at_rest / n_total, 0.046, delta=0.01)
        self.shift_last_bunch(self.beam, 0.3e-9)      # 0.6 sigma
        profile.track()
        self.assertEqual(
            profile.n_macroparticles_outside,
            n_total - int(round(profile.n_macroparticles.sum())))
        # about 8 % of the shifted bunch beyond its window instead of 4.6 %
        self.assertGreater(
            profile.n_macroparticles_outside - at_rest, 0.02 * N_MACROPARTICLES)

    def test_outside_particles_warn_above_the_fraction(self):
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, warn_outside_fraction=0.06)

        def outside_warnings(action):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                action()
            return [w for w in caught
                    if "outside the SparseBunch windows" in str(w.message)]

        self.assertEqual(outside_warnings(profile.track), [])   # 4.6 %
        # one bunch of 30 fully out of its window: 4.6 % + 3.3 % > 6 %
        self.shift_last_bunch(self.beam, 1.5e-9)
        self.assertEqual(len(outside_warnings(profile.track)), 1)

    def test_recentering_follows_a_drifting_bunch(self):
        """With recenter_every the window of a bunch that drifted is moved
        onto its centroid by whole bins, and the particles come back inside."""
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, recenter_every=1)
        k = self.shift_last_bunch(self.beam, 0.3e-9)
        window = profile.profiles_list[k]
        cut_left_before = window.cut_left
        profile.track()
        bin_size = profile.window_bin_size
        moved_by = (window.cut_left - cut_left_before) / bin_size
        self.assertAlmostEqual(moved_by, round(moved_by), delta=1e-6)
        # the whole 0.3 ns (120 bins) in one track(): the centroid of the
        # truncated histogram is biased towards the window, which the
        # repeated step of recenter_windows makes up for
        self.assertGreater(moved_by, 0.3e-9 / bin_size - 5)
        # centred on the bunch again: half a bin plus the statistical noise
        # of the centroid of 1e5 particles (1.6 ps) against 2.5 ps bins
        centroid = np.mean(self.bunch_dt(self.beam, k))
        self.assertLessEqual(
            abs(0.5 * (window.cut_left + window.cut_right) - centroid),
            4 * bin_size)
        np.testing.assert_allclose(
            self.cut_arrays_of(profile, k), (window.cut_left, window.cut_right),
            rtol=1e-12)
        # the histogram is the one of the moved window, and the tails are
        # the only particles outside again
        hist, _ = np.histogram(
            self.beam.dt, bins=window.n_slices,
            range=(window.cut_left, window.cut_right))
        np.testing.assert_allclose(window.n_macroparticles, hist,
                                   rtol=self.rtol, atol=self.atol)
        self.assertAlmostEqual(
            profile.n_macroparticles_outside / self.beam.n_macroparticles,
            0.046, delta=0.01)
        # the other windows did not move
        for j, other in enumerate(profile.profiles_list[:-1]):
            with self.subTest(bunch=j):
                np.testing.assert_allclose(
                    other.cut_left, self.profile.profiles_list[j].cut_left,
                    rtol=1e-12)

    def cut_arrays_of(self, profile, k):
        return profile.cut_left_array[k], profile.cut_right_array[k]

    def test_recentering_only_when_due(self):
        """The window moves on the track() call that is a multiple of
        recenter_every only (the build already tracked once or twice,
        depending on the mode)."""
        profile = build_sparse_profile_bunch(self.beam, self.rf)
        profile.recenter_every = profile._turns_tracked + 2
        k = self.shift_last_bunch(self.beam, 0.3e-9)
        cut_left_before = profile.profiles_list[k].cut_left
        profile.track()                                  # not due
        self.assertEqual(profile.profiles_list[k].cut_left, cut_left_before)
        profile.track()                                  # due
        self.assertGreater(profile.profiles_list[k].cut_left, cut_left_before)

    def lengthen_last_bunch(self, beam, factor, n_bunches=None):
        """Stretch the last bunch (no neighbour after it) about its median."""
        k = (len(self.buckets) if n_bunches is None else n_bunches) - 1
        block = slice(k * N_MACROPARTICLES, (k + 1) * N_MACROPARTICLES)
        median = np.median(beam.dt[block])
        beam.dt[block] = median + factor * (beam.dt[block] - median)
        return k

    def build_spaced_beam(self):
        """One bunch per batch (the first), batch_spacing buckets apart, so
        that widened windows have room: a Beam and its bunch list."""
        full = self.beam
        n_bunches = number_of_batches
        beam = Beam(self.ring, n_bunches * N_MACROPARTICLES,
                    n_bunches * BUNCH_INTENSITY)
        for i in range(n_bunches):
            source = i * number_of_bunches_per_batch * N_MACROPARTICLES
            beam.dt[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = \
                full.dt[source : source + N_MACROPARTICLES]
            beam.dE[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = \
                full.dE[source : source + N_MACROPARTICLES]
        bunch_list = np.zeros(HARMONIC_NUMBER)
        bunch_list[np.arange(n_bunches) * batch_spacing] = 1
        return beam, bunch_list

    def test_windows_widen_with_the_bunch_length(self):
        """A bunch twice as long as at the construction fills the outer bins
        of its window: at the next re-centring all windows are widened by
        the factor (whole bins, same lattice, same bin size), as often as
        needed, centred again and histogrammed again, and the bunch is back
        inside. Bunches one batch spacing apart, so the windows have room."""
        from scipy.special import erf

        beam, bunch_list = self.build_spaced_beam()
        profile = SparseBunch(
            rf_station=self.rf, beam=beam, bunch_list=bunch_list,
            number_of_slices_per_bucket=N_SLICES_PER_BUCKET,
            length_in_sigma=LENGTH_IN_SIGMA, tracker_mode="C",
            recenter_every=1, warn_outside_fraction=None,
            resize_windows=True, resize_tail_fraction=0.1)
        profile.track()
        n_bins_before = profile.number_of_slices_per_profile
        width_before = profile.window_width
        bin_size = profile.window_bin_size
        k = self.lengthen_last_bunch(beam, 2.0, n_bunches=number_of_batches)
        with self.assertWarns(UserWarning):
            profile.track()
        n_bins = profile.number_of_slices_per_profile
        # widened by the factor, as often as needed for the tails to fit
        self.assertGreaterEqual(n_bins, int(np.ceil(1.5 * n_bins_before)))
        self.assertLessEqual(np.nanmax(profile._tail_fractions()),
                             profile.resize_tail_fraction)
        self.assertLess(profile.window_width, batch_spacing * self.t_rf)
        np.testing.assert_allclose(profile.window_width, n_bins * bin_size, rtol=1e-12)
        np.testing.assert_allclose(profile.window_bin_size, bin_size, rtol=1e-12)
        self.assertEqual(profile.n_macroparticles_array.shape, (number_of_batches, n_bins))
        dt = beam.dt[k * N_MACROPARTICLES : (k + 1) * N_MACROPARTICLES]
        for j, window in enumerate(profile.profiles_list):
            with self.subTest(bunch=j):
                self.assertEqual(window.n_slices, n_bins)
                np.testing.assert_allclose(
                    window.cut_right - window.cut_left, profile.window_width, rtol=1e-12)
                lattice = window.cut_left / bin_size
                self.assertAlmostEqual(lattice, round(lattice), delta=1e-6)
                hist, _ = np.histogram(
                    beam.dt, bins=n_bins, range=(window.cut_left, window.cut_right))
                np.testing.assert_allclose(window.n_macroparticles, hist,
                                           rtol=self.rtol, atol=self.atol)
        window = profile.profiles_list[k]
        # the stretched bunch holds what a Gaussian holds in the new window
        half_width_in_sigma = 0.5 * profile.window_width / np.std(dt)
        inside = np.mean((dt >= window.cut_left) & (dt <= window.cut_right))
        self.assertGreater(inside, erf(half_width_in_sigma / np.sqrt(2)) - 0.03)
        self.assertLessEqual(
            abs(0.5 * (window.cut_left + window.cut_right) - np.mean(dt)), 4 * bin_size)
        self.assertGreater(profile.window_width, width_before)

    def test_widening_stops_at_the_neighbouring_window(self):
        """Adjacent bunches: the windows cannot grow past the bucket spacing
        and the class says so instead of overlapping them."""
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, recenter_every=1, resize_windows=True)
        self.lengthen_last_bunch(self.beam, 2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile.track()
        self.assertTrue(any("cannot be widened further" in str(w.message) for w in caught))
        self.assertLessEqual(profile.window_width, self.t_rf * (1 + 1e-9))

    def test_no_widening_at_rest(self):
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, recenter_every=1, resize_windows=True)
        n_bins = profile.number_of_slices_per_profile
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                profile.track()
        self.assertEqual(profile.number_of_slices_per_profile, n_bins)
        self.assertEqual(
            [w for w in caught if "widened" in str(w.message)], [])

    def test_widening_can_be_disabled(self):
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, recenter_every=1, resize_windows=False)
        n_bins = profile.number_of_slices_per_profile
        self.lengthen_last_bunch(self.beam, 2.0)
        profile.track()
        self.assertEqual(profile.number_of_slices_per_profile, n_bins)

    def test_resize_factor_must_exceed_one(self):
        with self.assertRaises(ValueError):
            build_sparse_profile_bunch(
                self.beam, self.rf, recenter_every=1, resize_windows=True,
                resize_factor=1.0)

    def test_tracker_mode_c_needs_the_bucket_grid(self):
        """The one-pass histogram locates the window of a particle from its
        bin index on the global lattice: not available with a fixed number
        of bins per window."""
        with self.assertRaises(ValueError):
            build_sparse_profile_bunch(
                self.beam, self.rf, n_slices_per_window=N_SLICES_PER_BUNCH,
                tracker_mode="C")

    def test_histogram_c_matches_one_by_one(self):
        """The one-pass histogram of all windows equals the window-by-window
        one (Profile.track) and np.histogram, at rest, after a drift with
        re-centring, and after an injection."""
        one_by_one = build_sparse_profile_bunch(
            self.beam, self.rf, tracker_mode="onebyone")
        np.testing.assert_allclose(
            self.profile.n_macroparticles_array,
            one_by_one.n_macroparticles_array,
            rtol=self.rtol, atol=self.atol)
        self.assert_histograms_match_numpy(self.beam, self.profile)
        self.assertEqual(self.profile.tracker_mode, "C")
        # a drifting bunch, followed by the windows
        profile = build_sparse_profile_bunch(
            self.beam, self.rf, recenter_every=1)
        self.shift_last_bunch(self.beam, 0.3e-9)
        profile.track()
        self.assert_histograms_match_numpy(self.beam, profile)
        self.assertEqual(profile.tracker_mode, "C")

    def test_injected_windows_centred_on_the_injected_particles(self):
        """With the injected blocks given, a new window is centred on its
        own bunch. From the bucket content alone it is not when a
        neighbouring bunch has drifted so that its tail fills part of the
        bucket: here the last stored bunch, moved by 0.8 ns towards the
        bucket of the first injected one."""
        beam = build_beam(self.ring, self.rf, injected_batches=1)
        last = number_of_bunches_per_batch - 1
        beam.dt[last * N_MACROPARTICLES : (last + 1) * N_MACROPARTICLES] += 0.8e-9
        with_blocks = build_sparse_profile_bunch(
            beam, self.rf, injected_batches=1)
        from_buckets = build_sparse_profile_bunch(
            beam, self.rf, injected_batches=1)
        # the new batch right after the stored one (buckets 3, 4, 5)
        batch = build_beam(self.ring, self.rf, injected_batches=1)
        shift_buckets = number_of_bunches_per_batch
        beam.add_particles([batch.dt + shift_buckets * self.t_rf, batch.dE])
        new_buckets = bunch_buckets(1) + shift_buckets
        bunch_list = np.zeros(HARMONIC_NUMBER)
        bunch_list[bunch_buckets(1)] = 1
        bunch_list[new_buckets] = 1
        n_new = batch.n_macroparticles
        blocks = [slice(beam.n_macroparticles - n_new + k * N_MACROPARTICLES,
                        beam.n_macroparticles - n_new + (k + 1) * N_MACROPARTICLES)
                  for k in range(number_of_bunches_per_batch)]
        with_blocks.update_bunch_list(
            updated_bunch_list=bunch_list, new_bunch_indices=new_buckets,
            new_particles=blocks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the tail makes it "longer"
            from_buckets.update_bunch_list(
                updated_bunch_list=bunch_list, new_bunch_indices=new_buckets)
        bin_size = with_blocks.window_bin_size
        for k, block in enumerate(blocks):
            centroid = np.mean(beam.dt[block])
            window = with_blocks.profiles_list[number_of_bunches_per_batch + k]
            other = from_buckets.profiles_list[number_of_bunches_per_batch + k]
            with self.subTest(bunch=k):
                self.assertLessEqual(
                    abs(0.5 * (window.cut_left + window.cut_right) - centroid),
                    0.5 * bin_size * (1 + 1e-9))
                if k == 0:
                    # the drifted neighbour's tail pulls the bucket-based
                    # centre by far more than a bin
                    self.assertGreater(
                        abs(0.5 * (other.cut_left + other.cut_right) - centroid),
                        10 * bin_size)
        # one block for the whole batch, split by bucket, is accepted too
        again = build_sparse_profile_bunch(beam, self.rf, injected_batches=1)
        again.update_bunch_list(
            updated_bunch_list=bunch_list, new_bunch_indices=new_buckets,
            new_particles=slice(beam.n_macroparticles - n_new, None))
        self.assertEqual(len(again.profiles_list), 2 * number_of_bunches_per_batch)
        with self.assertRaises(ValueError):
            build_sparse_profile_bunch(beam, self.rf, injected_batches=1).update_bunch_list(
                updated_bunch_list=bunch_list, new_bunch_indices=new_buckets,
                new_particles=blocks[:1])

    def test_empty_bucket_gets_the_bucket_centre_and_warns(self):
        bunch_list = np.zeros(HARMONIC_NUMBER)
        empty_bucket = (number_of_batches + 2) * batch_spacing
        bunch_list[self.buckets] = 1
        bunch_list[empty_bucket] = 1
        with self.assertWarns(UserWarning):
            profile = SparseBunch(
                rf_station=self.rf, beam=self.beam, bunch_list=bunch_list,
                number_of_slices_per_bucket=N_SLICES_PER_BUCKET,
                length_in_sigma=LENGTH_IN_SIGMA, tracker_mode="C",
                warn_outside_fraction=None)
        window = profile.profiles_list[-1]
        np.testing.assert_allclose(
            0.5 * (window.cut_left + window.cut_right),
            (empty_bucket + 0.5) * self.t_rf,
            atol=0.5 * profile.window_bin_size * (1 + 1e-9))
        np.testing.assert_allclose(
            window.cut_right - window.cut_left, profile.window_width, rtol=1e-12)
        profile.track()
        self.assertEqual(window.n_macroparticles.sum(), 0)

    def test_margin_below_one_rejected(self):
        with self.assertRaises(ValueError):
            build_sparse_profile_bunch(self.beam, self.rf, length_in_sigma=0)

    def test_multiturn_injection_adds_windows_around_new_bunches(self):
        """Batch after batch: the new bunches get their own windows, appended
        in injection order, and every window still follows its bunch and
        matches np.histogram."""
        injected_batches = 1
        beam = build_beam(
            self.ring, self.rf, injected_batches=injected_batches
        )
        profile = build_sparse_profile_bunch(
            beam, self.rf, injected_batches=injected_batches
        )
        while injected_batches < number_of_batches:
            batch = build_beam(self.ring, self.rf, injected_batches=1)
            beam.add_particles(
                [
                    batch.dt + injected_batches * batch_spacing * self.t_rf,
                    batch.dE,
                ]
            )
            new_buckets = injected_batches * batch_spacing + (
                bunch_spacing * np.arange(number_of_bunches_per_batch)
            )
            bunch_list = np.zeros(HARMONIC_NUMBER)
            bunch_list[bunch_buckets(injected_batches + 1)] = 1
            profile.update_bunch_list(
                updated_bunch_list=bunch_list, new_bunch_indices=new_buckets
            )
            injected_batches += 1
            profile.track()
            with self.subTest(injected_batches=injected_batches):
                self.assertEqual(
                    len(profile.profiles_list),
                    injected_batches * number_of_bunches_per_batch,
                )
                self.assert_windows_follow_the_bunches(
                    beam, profile, bunch_buckets(injected_batches),
                    LENGTH_IN_SIGMA
                )
                self.assert_histograms_match_numpy(beam, profile)


if __name__ == "__main__":
    unittest.main()
