# coding: utf8
"""
Test functions of the rf beam current with sparse profiles.

These test functions have been partially generated with the help of a LLM.

Author:
Lina Valle
"""

import unittest

import numpy as np
import warnings
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
BUNCH_SIGMA_DT = 0.25e-9  # [s]; the whole bunch stays inside its RF bucket,
# so the standard and sparse profiles see the same total charge
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
        message="Warning: for a large number of batches, "
        "the bin_size "
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
        for i in range(number_of_batches):
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
        for i in range(number_of_batches):
            beam.dE[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dE
            )
            beam.dt[i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
            )
    return beam


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


def build_sparse_profile(beam, rf_station, n_slices):
    """A SparseBatch profile with a profile per number of bunches."""
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(number_of_batches):
        batch_list[k * batch_spacing] = 1
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(batch_spacing / 2) + 1) * n_slices,
        batch_list=batch_list,
        batch_length=int(batch_spacing / 2) + 1,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


def build_sparse_bucket_profile(beam, rf_station, n_slices):
    """A SparseBucket profile with one profile (one RF bucket) per bunch."""
    bunch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(number_of_batches):
        for i in range(number_of_bunches_per_batch):
            bunch_list[k * batch_spacing + i * bunch_spacing] = 1
    sparse_profile = SparseBucket(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=n_slices,
        bunch_list=bunch_list,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


class TestRFBeamCurrent(unittest.TestCase):
    """Compare the rf beam current computed on sparse profiles with the one
    computed on a standard Profile covering all the buckets.

    Two sparse profiles are built on the same beam, a SparseBatch (one
    profile per batch) and a SparseBucket (one profile per bunch), and
    every check runs on both of them.
    """

    N_SLICES = (
        4 * HARMONIC_NUMBER // 5
    )  # fine relative to the coarse (n_coarse) grid

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.omega = 2 * np.pi * 200.222e6

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_sparse_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_bucket = build_sparse_bucket_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profiles_sparse = {
            "SparseBatch": self.profile_sparse,
            "SparseBucket": self.profile_bucket,
        }
        self.T_s = 5 * self.rf.t_rev[0] / self.rf.harmonic[0, 0]
        self.n_points = 100
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
        """Demodulated charges of a sparse profile on a fine grid extended
        with empty bins past the last window, so that the last coarse
        sample is complete. Returns the charges and the extended grid."""
        profile_bin_centers = profile_sparse.bin_centers
        profile_n_macroparticles = profile_sparse.n_macroparticles
        extra_bins = np.arange(
            profile_bin_centers[-1],
            profile_bin_centers[-1] + self.T_s + np.pi / self.omega,
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
        _, _, charges_fine = self._demodulate(
            charges, profile_bin_centers_for_coarse
        )
        return charges_fine, profile_bin_centers_for_coarse

    def _charges_coarse(self, charges_fine, bin_centers):
        return charges_from_fine_to_coarse(
            T_s=self.T_s,
            charges_fine=charges_fine,
            dT=0,
            n_points=self.n_points,
            omega_c=self.omega,
            profile_bin_centers=bin_centers,
        )

    # Tests -------------------------------------------------------------------

    def test_bin_centers_match(self):
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

    def test_n_macroparticles_match(self):
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

    def test_bin_size_matches(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                self.assertAlmostEqual(
                    self.profile_std.bin_size,
                    profile_sparse.bin_size,
                    places=15,
                )

    def test_charges_fine_grid(self):
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

    def test_charges_from_fine_to_coarse(self):
        charges_std = self._charges(self.profile_std)
        _, _, charges_fine_std = self._demodulate(
            charges_std, self.profile_std.bin_centers
        )
        charges_coarse_std = self._charges_coarse(
            charges_fine_std, self.profile_std.bin_centers
        )
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                # On the bare sparse grid the last coarse sample is
                # incomplete, so the coarse charges cannot match
                _, _, charges_fine_sparse = self._demodulate(
                    self._charges(profile_sparse), profile_sparse.bin_centers
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
                )

    def test_rf_beam_current(self):
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

    def test_downsampling(self):
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
                profile_sparse.track()
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


if __name__ == "__main__":
    unittest.main()
