# coding: utf8
"""
Test functions of the rf beam current with sparse profiles.

The rf beam current computed on sparse profiles (a SparseBatch and a
SparseBucket built on the same beam) is compared with the one computed on a
standard Profile covering all the buckets, for three injection schemes: all
the batches at once, one batch per turn in bucket order, and one batch per
turn out of bucket order.

These test functions have been partially generated with the help of a LLM.

Author:
Lina Valle
"""

import unittest

import numpy as np
from scipy.constants import e

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import SparseBatch, SparseBucket
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.signal_processing import (
    rf_beam_current,
    charges_from_fine_to_coarse,
)

# ---------------------------------------------------------------------------
# Shared machine parameters (LHC-like), reused by every test class below.
# These match the parameters used in BLonD's own TestLHCOpenDrive unittest.
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
BUNCH_SIGMA_DT = 0.25e-9  # [s]; the whole bunch stays inside its RF bucket,
# so the standard and sparse profiles see the same total charge
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

# Bucket of the first bunch of each batch, in injection order: either in
# bucket order, batch_spacing buckets apart, or out of bucket order, with
# bucket 20 injected twice
batch_buckets_sequential = [
    k * batch_spacing for k in range(number_of_batches)
]
batch_buckets_shuffled = [0, 45, 20, 35, 10, 30, 20, 40, 5, 15]


def bunch_buckets(batch_buckets, k):
    """Bucket indices of the bunches of batch k."""
    return [
        batch_buckets[k] + i * bunch_spacing
        for i in range(number_of_bunches_per_batch)
    ]


def batch_pattern(batch_buckets, injected_batches):
    """Filling pattern of the first bucket of each injected batch."""
    pattern = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
        pattern[batch_buckets[k]] = 1
    return pattern


def bunch_pattern(batch_buckets, injected_batches):
    """Filling pattern of every bunch of the injected batches."""
    pattern = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
        pattern[bunch_buckets(batch_buckets, k)] = 1
    return pattern


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


def build_batch(ring, rf_station, seed=1234):
    """One batch: number_of_bunches_per_batch copies of the same Gaussian
    bunch, bunch_spacing buckets apart, the first one in bucket 0."""
    single_bunch = Beam(ring, N_MACROPARTICLES, BUNCH_INTENSITY)
    bigaussian(
        ring,
        rf_station,
        single_bunch,
        sigma_dt=BUNCH_SIGMA_DT,
        seed=seed,
        reinsertion=True,
    )
    batch = Beam(
        ring,
        number_of_bunches_per_batch * N_MACROPARTICLES,
        number_of_bunches_per_batch * BUNCH_INTENSITY,
    )
    for i in range(number_of_bunches_per_batch):
        block = slice(i * N_MACROPARTICLES, (i + 1) * N_MACROPARTICLES)
        batch.dE[block] = single_bunch.dE
        batch.dt[block] = (
            single_bunch.dt + i * bunch_spacing * rf_station.t_rf[0, 0]
        )
    return batch


def build_beam(ring, rf_station, batch_buckets, injected_batches, seed=1234):
    """A beam holding the first injected_batches batches, each a copy of
    the same Gaussian batch placed at its batch bucket."""
    n_per_batch = number_of_bunches_per_batch * N_MACROPARTICLES
    beam = Beam(
        ring,
        n_per_batch * injected_batches,
        number_of_bunches_per_batch * BUNCH_INTENSITY * injected_batches,
    )
    batch = build_batch(ring, rf_station, seed)
    for k in range(injected_batches):
        block = slice(k * n_per_batch, (k + 1) * n_per_batch)
        beam.dE[block] = batch.dE
        beam.dt[block] = batch.dt + batch_buckets[k] * rf_station.t_rf[0, 0]
    return beam


def inject_batch(
    beam,
    ring,
    rf_station,
    batch_buckets,
    profiles_sparse,
    injected_batches,
    seed=1234,
):
    """Inject batch number injected_batches into the beam, register it in
    every sparse profile and re-track them. Returns the updated number of
    injected batches."""
    k = injected_batches
    batch = build_batch(ring, rf_station, seed)
    beam.add_particles(
        [batch.dt + batch_buckets[k] * rf_station.t_rf[0, 0], batch.dE]
    )
    for profile_sparse in profiles_sparse:
        if isinstance(profile_sparse, SparseBatch):
            profile_sparse.update_batch_list(
                updated_batch_list=batch_pattern(batch_buckets, k + 1)
            )
        else:
            # Several bunches are new at once: give them in injection order
            previous = profile_sparse.bunch_list
            new_bunches = [
                b for b in bunch_buckets(batch_buckets, k) if previous[b] == 0
            ]
            profile_sparse.update_bunch_list(
                updated_bunch_list=bunch_pattern(batch_buckets, k + 1),
                new_bunch_indices=new_bunches or None,
            )
        profile_sparse.track()
    return k + 1


def build_standard_profile(beam, rf_station, n_slices, extra_buckets):
    """A standard Profile covering all the batches and extra_buckets more,
    so that its last coarse-grid sample is complete."""
    n_buckets = batch_spacing * number_of_batches + extra_buckets
    profile = Profile(
        beam,
        CutOptions(
            cut_left=0.0,
            cut_right=n_buckets * rf_station.t_rf[0, 0],
            n_slices=n_slices * n_buckets,
        ),
    )
    profile.track()
    return profile


def build_sparse_profile(
    beam, rf_station, n_slices, batch_buckets, injected_batches
):
    """A SparseBatch profile with one profile per injected batch."""
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(batch_spacing / 2) + 1) * n_slices,
        batch_list=batch_pattern(batch_buckets, injected_batches),
        batch_length=int(batch_spacing / 2) + 1,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


def build_sparse_bucket_profile(
    beam, rf_station, n_slices, batch_buckets, injected_batches
):
    """A SparseBucket profile with one profile (one RF bucket) per bunch of
    the injected batches."""
    sparse_profile = SparseBucket(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=n_slices,
        bunch_list=bunch_pattern(batch_buckets, injected_batches),
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


class TestRFBeamCurrent(unittest.TestCase):
    """Compare the rf beam current computed on sparse profiles with the one
    computed on a standard Profile covering all the buckets.

    Two sparse profiles are built on the same beam, a SparseBatch (one
    profile per batch) and a SparseBucket (one profile per bunch), and
    every check runs on both of them. Here all the batches are injected at
    once, in bucket order; the subclasses inject them one per turn.
    """

    N_SLICES = (
        4 * HARMONIC_NUMBER // 5
    )  # fine relative to the coarse (n_coarse) grid
    batch_buckets = batch_buckets_sequential
    injected_batches_at_setup = number_of_batches
    extra_buckets = 1  # of the standard profile past the batches
    rf_periods_per_coarse_sample = 5
    n_points = 100  # coarse-grid samples
    coarse_extension = 1  # of the sparse fine grid past its last window [T_s]
    # Without that extension the last coarse sample of the sparse profile
    # is incomplete, so the bare sparse grid must give a different result
    bare_grid_differs = True

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.injected_batches = self.injected_batches_at_setup
        self.beam = build_beam(
            self.ring, self.rf, self.batch_buckets, self.injected_batches
        )
        self.omega = 2 * np.pi * 200.222e6

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES, self.extra_buckets
        )
        self.profile_sparse = build_sparse_profile(
            self.beam,
            self.rf,
            self.N_SLICES,
            self.batch_buckets,
            self.injected_batches,
        )
        self.profile_bucket = build_sparse_bucket_profile(
            self.beam,
            self.rf,
            self.N_SLICES,
            self.batch_buckets,
            self.injected_batches,
        )
        self.profiles_sparse = {
            "SparseBatch": self.profile_sparse,
            "SparseBucket": self.profile_bucket,
        }
        self.T_s = (
            self.rf_periods_per_coarse_sample
            * self.rf.t_rev[0]
            / self.rf.harmonic[0, 0]
        )
        self.rtol = 1e-15
        self.atol = 1e-12

    # Helpers -----------------------------------------------------------------

    def _windows(self, profile_sparse):
        """Yield (p, profile, index) for each window of a sparse profile:
        the profile number, its Profile object and the index of its first
        bin in the standard profile."""
        for p, profile in enumerate(profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            yield p, profile, index

    def _assert_windows_match(
        self, array_std, array_sparse, profile_sparse, name, what
    ):
        """Check an array in sparse layout against the same quantity in the
        standard profile layout, window by window."""
        for p, profile, index in self._windows(profile_sparse):
            np.testing.assert_allclose(
                array_std[index : index + profile.n_slices],
                array_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"{what} differ between standard Profile and {name} "
                f"for the same beam, profile number {p}",
            )

    @staticmethod
    def _charges(profile):
        """Charge per bin [C], as in rf_beam_current."""
        return (
            profile.beam.ratio
            * profile.beam.particle.charge
            * e
            * np.copy(profile.n_macroparticles)
        )

    def _demodulate(self, charges, bin_centers):
        """Demodulated charges at omega: I, Q and I + jQ, as in
        rf_beam_current."""
        I_f = 2.0 * charges * np.cos(self.omega * bin_centers)
        Q_f = -2.0 * charges * np.sin(self.omega * bin_centers)
        return I_f, Q_f, I_f + 1j * Q_f

    @staticmethod
    def _total_charge(profile):
        return (
            np.sum(profile.n_macroparticles)
            / profile.beam.n_macroparticles
            * profile.beam.intensity
        )

    def _charges_fine_extended(self, profile_sparse):
        """Demodulated charges of a sparse profile on its fine grid sorted
        in time and extended with empty bins past the last window, so that
        the last coarse sample is complete. Returns the charges and the
        extended grid."""
        order = np.argsort(profile_sparse.bin_centers)
        profile_bin_centers = profile_sparse.bin_centers[order]
        profile_n_macroparticles = profile_sparse.n_macroparticles[order]
        extra_bins = np.arange(
            profile_bin_centers[-1],
            profile_bin_centers[-1]
            + self.coarse_extension * self.T_s
            + np.pi / self.omega,
            step=profile_sparse.bin_size,
        )
        profile_bin_centers_for_coarse = np.concatenate(
            (profile_bin_centers, extra_bins)
        )
        profile_n_macroparticles_for_coarse = np.concatenate(
            (profile_n_macroparticles, np.zeros(len(extra_bins)))
        )
        charges = (
            profile_sparse.beam.ratio
            * profile_sparse.beam.particle.charge
            * e
            * profile_n_macroparticles_for_coarse
        )
        _, _, charges_fine_for_coarse_grid = self._demodulate(
            charges, profile_bin_centers_for_coarse
        )
        return charges_fine_for_coarse_grid, profile_bin_centers_for_coarse

    def _charges_coarse(self, charges_fine, bin_centers):
        return charges_from_fine_to_coarse(
            T_s=self.T_s,
            charges_fine=charges_fine,
            dT=0,
            n_points=self.n_points,
            omega_c=self.omega,
            profile_bin_centers=bin_centers,
        )

    # Checks, shared by the single-turn and multi-turn tests ------------------

    def _check_bin_centers(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                for p, profile, index in self._windows(profile_sparse):
                    np.testing.assert_allclose(
                        self.profile_std.bin_centers[index],
                        profile_sparse.bin_centers[p * profile.n_slices],
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg="bin centers differ between standard "
                        f"Profile and {name} for the same beam, profile "
                        f"number {p}",
                    )

    def _check_n_macroparticles(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                self._assert_windows_match(
                    self.profile_std.bin_centers,
                    profile_sparse.bin_centers,
                    profile_sparse,
                    name,
                    "bin_centers",
                )
                self._assert_windows_match(
                    self.profile_std.n_macroparticles,
                    profile_sparse.n_macroparticles,
                    profile_sparse,
                    name,
                    "n_macroparticles",
                )

    def _check_bin_size(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                self.assertAlmostEqual(
                    self.profile_std.bin_size,
                    profile_sparse.bin_size,
                    places=15,
                )

    def _check_charges_fine_grid(self):
        charges_std = self._charges(self.profile_std)
        I_f_std, Q_f_std, charges_fine_std = self._demodulate(
            charges_std, self.profile_std.bin_centers
        )
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                charges_sparse = self._charges(profile_sparse)
                self._assert_windows_match(
                    charges_std,
                    charges_sparse,
                    profile_sparse,
                    name,
                    "charges",
                )
                self.assertEqual(
                    self._total_charge(self.profile_std),
                    self._total_charge(profile_sparse),
                )
                I_f, Q_f, charges_fine = self._demodulate(
                    charges_sparse, profile_sparse.bin_centers
                )
                self._assert_windows_match(
                    I_f_std, I_f, profile_sparse, name, "I_f"
                )
                self._assert_windows_match(
                    Q_f_std, Q_f, profile_sparse, name, "Q_f"
                )
                self._assert_windows_match(
                    charges_fine_std,
                    charges_fine,
                    profile_sparse,
                    name,
                    "charges_fine",
                )

    def _check_charges_from_fine_to_coarse(self):
        charges_std = self._charges(self.profile_std)
        _, _, charges_fine_std = self._demodulate(
            charges_std, self.profile_std.bin_centers
        )
        charges_coarse_std = self._charges_coarse(
            charges_fine_std, self.profile_std.bin_centers
        )
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                if self.bare_grid_differs:
                    _, _, charges_fine_sparse = self._demodulate(
                        self._charges(profile_sparse),
                        profile_sparse.bin_centers,
                    )
                    charges_coarse_sparse = self._charges_coarse(
                        charges_fine_sparse, profile_sparse.bin_centers
                    )
                    with self.assertRaises(AssertionError):
                        np.testing.assert_allclose(
                            charges_coarse_std,
                            charges_coarse_sparse,
                            rtol=self.rtol,
                            atol=self.atol,
                        )
                # Extending the sparse grid with empty bins past the last
                # window recovers the standard result
                (
                    charges_fine_for_coarse_grid,
                    profile_bin_centers_for_coarse,
                ) = self._charges_fine_extended(profile_sparse)
                charges_coarse_sparse = self._charges_coarse(
                    charges_fine_for_coarse_grid,
                    profile_bin_centers_for_coarse,
                )
                np.testing.assert_allclose(
                    charges_coarse_std,
                    charges_coarse_sparse,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="coarse-grid charges differ between standard "
                    f"Profile and {name}",
                )

    def _check_rf_beam_current(self):
        rf_current_std = rf_beam_current(
            self.profile_std,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            external_reference=True,
            dT=0,
        )
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                rf_current_sparse = rf_beam_current(
                    profile_sparse,
                    self.omega,
                    self.ring.t_rev[0],
                    lpf=False,
                    external_reference=True,
                    dT=0,
                )
                self._assert_windows_match(
                    rf_current_std,
                    rf_current_sparse,
                    profile_sparse,
                    name,
                    "rf beam current",
                )

    def _check_downsampling(self):
        downsample_dict = {
            "Ts": self.T_s,
            "points": self.n_points,
        }
        rf_current_std, rf_current_coarse_std = rf_beam_current(
            self.profile_std,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            downsample=downsample_dict,
            external_reference=True,
            dT=0,
        )
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                rf_current_sparse, rf_current_coarse_sparse = rf_beam_current(
                    profile_sparse,
                    self.omega,
                    self.ring.t_rev[0],
                    lpf=False,
                    downsample=downsample_dict,
                    external_reference=True,
                    dT=0,
                )
                self._assert_windows_match(
                    rf_current_std,
                    rf_current_sparse,
                    profile_sparse,
                    name,
                    "rf beam current",
                )
                np.testing.assert_allclose(
                    rf_current_coarse_std,
                    rf_current_coarse_sparse,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="coarse-grid rf beam current differs between "
                    f"standard Profile and {name}",
                )

    # Tests -------------------------------------------------------------------

    def test_bin_centers_match(self):
        self._check_bin_centers()

    def test_n_macroparticles_match(self):
        self._check_n_macroparticles()

    def test_bin_size_matches(self):
        self._check_bin_size()

    def test_charges_fine_grid(self):
        self._check_charges_fine_grid()

    def test_charges_from_fine_to_coarse(self):
        self._check_charges_from_fine_to_coarse()

    def test_rf_beam_current(self):
        self._check_rf_beam_current()

    def test_downsampling(self):
        self._check_downsampling()


class TestRFBeamCurrentMultiTurnInjection(TestRFBeamCurrent):
    """Multi-turn injection: one batch at setUp, then the remaining batches
    injected one by one in bucket order. The test_* methods of the parent
    run on the single batch; the test_muliturn_injection_* methods repeat
    the same check after each injection.
    """

    injected_batches_at_setup = 1
    extra_buckets = 10
    rf_periods_per_coarse_sample = 10
    n_points = 3654
    coarse_extension = 2
    # With a single batch the bare sparse grid does not reach a coarse-grid
    # sample boundary, so only the extended grid is checked
    bare_grid_differs = False

    def _inject_all(self, check):
        """Inject the remaining batches one at a time; after each injection
        re-track the standard profile and run check()."""
        while self.injected_batches < number_of_batches:
            self.injected_batches = inject_batch(
                self.beam,
                self.ring,
                self.rf,
                self.batch_buckets,
                self.profiles_sparse.values(),
                self.injected_batches,
            )
            self.profile_std.track()
            with self.subTest(injected_batches=self.injected_batches):
                check()

    def test_muliturn_injection_bin_centers_match(self):
        self._inject_all(self._check_bin_centers)

    def test_muliturn_injection_n_macroparticles_match(self):
        self._inject_all(self._check_n_macroparticles)

    def test_muliturn_injection_charges_fine_grid(self):
        self._inject_all(self._check_charges_fine_grid)

    def test_muliturn_injection_from_fine_to_coarse(self):
        self._inject_all(self._check_charges_from_fine_to_coarse)

    def test_muliturn_injection_rf_beam_current(self):
        self._inject_all(self._check_rf_beam_current)

    def test_muliturn_injection_downsampling(self):
        self._inject_all(self._check_downsampling)


class TestRFBeamCurrentMultiTurnInjectionBatchList(
    TestRFBeamCurrentMultiTurnInjection
):
    """Multi-turn injection out of bucket order, following
    batch_buckets_shuffled, with bucket 20 injected twice."""

    batch_buckets = batch_buckets_shuffled


if __name__ == "__main__":
    unittest.main()
