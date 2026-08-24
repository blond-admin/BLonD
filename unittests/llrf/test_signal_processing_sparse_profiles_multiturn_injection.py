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
from blond.beam.sparse_profiles import SparseBatch
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
BUNCH_SIGMA_DT = 0.25e-9
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


def build_beam(ring, rf_station, injected_batches, seed=1234):
    """Build a Gaussian bunch so that Profile/SparseBatch slicing is
    well defined (an empty/point beam makes bin_size degenerate)."""
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
    return beam, injected_batches


def update_beam(
    beam, ring, rf_station, sparse_profile, injected_batches, seed=1234
):
    """Build a Gaussian bunch so that Profile/SparseBatch slicing is
    well defined (an empty/point beam makes bin_size degenerate)."""

    if injected_batches < number_of_batches:
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
        single_batch = Beam(
            ring,
            number_of_bunches_per_batch * N_MACROPARTICLES,
            number_of_bunches_per_batch * BUNCH_INTENSITY,
        )
        if number_of_bunches_per_batch > 1:
            for i in range(number_of_bunches_per_batch):
                single_batch.dE[
                    i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
                ] = single_bunch.dE
                single_batch.dt[
                    i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
                ] = single_bunch.dt + i * bunch_spacing * rf_station.t_rf[0, 0]
        else:
            for i in range(number_of_batches):
                single_batch.dE[
                    i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
                ] = single_bunch.dE
                single_batch.dt[
                    i * N_MACROPARTICLES : (i + 1) * N_MACROPARTICLES
                ] = single_bunch.dt + i * batch_spacing * rf_station.t_rf[0, 0]
        injected_batches += 1
        updated_batch_list = np.zeros(HARMONIC_NUMBER)
        for k in range(injected_batches):
            updated_batch_list[k * batch_spacing] = 1
        index = np.where(updated_batch_list == 1)[0][-1]
        beam.add_particles(
            [single_batch.dt + index * rf_station.t_rf[0, 0], single_batch.dE]
        )
        sparse_profile.update_batch_list(updated_batch_list=updated_batch_list)
        return beam, sparse_profile, injected_batches
    else:
        return beam, sparse_profile, injected_batches


def build_standard_profile(beam, rf_station, n_slices):
    """A standard Profile covering the injected bunches and an extra bucket."""
    profile = Profile(
        beam,
        CutOptions(
            cut_left=0.0,
            cut_right=(batch_spacing * number_of_batches + 10)
            * rf_station.t_rf[
                0,
                0,
            ],
            n_slices=n_slices * (batch_spacing * number_of_batches + 10),
        ),
    )
    profile.track()
    return profile


def build_sparse_profile(beam, rf_station, n_slices, injected_batches):
    """A SparseBatch profile with a profile per number of bunches."""
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
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


class TestRFBeamCurrent(unittest.TestCase):
    N_SLICES = (
        4 * HARMONIC_NUMBER // 5
    )  # fine relative to the coarse (n_coarse) grid

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam, injected_batches = build_beam(
            self.ring, self.rf, injected_batches=1
        )
        self.omega = 2 * np.pi * 200.222e6

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_sparse_profile(
            self.beam,
            self.rf,
            self.N_SLICES,
            injected_batches=1,
        )
        self.T_s = 10 * self.rf.t_rev[0] / self.rf.harmonic[0, 0]
        self.n_points = 3654
        self.rtol = 1e-15
        self.atol = 1e-12

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

    def test_charges_fine_grid(self):
        charges_std = (
            self.profile_std.beam.ratio
            * self.profile_std.beam.particle.charge
            * e
            * np.copy(self.profile_std.n_macroparticles)
        )

        charges_sparse = (
            self.profile_sparse.beam.ratio
            * self.profile_sparse.beam.particle.charge
            * e
            * np.copy(self.profile_sparse.n_macroparticles)
        )
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                charges_std[index : index + profile.n_slices],
                charges_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="charges differ between "
                "standard Profile and SparseBatch for the same beam, profile number "
                f"{p}",
            )
        tot_charges = (
            np.sum(self.profile_std.n_macroparticles)
            / self.profile_std.beam.n_macroparticles
            * self.profile_std.beam.intensity
        )
        tot_charges_sparse = (
            np.sum(self.profile_sparse.n_macroparticles)
            / self.profile_sparse.beam.n_macroparticles
            * self.profile_sparse.beam.intensity
        )
        self.assertEqual(tot_charges, tot_charges_sparse)

        I_f_std = (
            2.0
            * charges_std
            * np.cos(self.omega * self.profile_std.bin_centers)
        )
        Q_f_std = (
            -2.0
            * charges_std
            * np.sin(self.omega * self.profile_std.bin_centers)
        )
        charges_fine_std = I_f_std + 1j * Q_f_std

        I_f_sparse = (
            2.0
            * charges_sparse
            * np.cos(self.omega * self.profile_sparse.bin_centers)
        )
        Q_f_sparse = (
            -2.0
            * charges_sparse
            * np.sin(self.omega * self.profile_sparse.bin_centers)
        )
        charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                I_f_std[index : index + profile.n_slices],
                I_f_sparse[p * profile.n_slices : (p + 1) * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                Q_f_std[index : index + profile.n_slices],
                Q_f_sparse[p * profile.n_slices : (p + 1) * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                charges_fine_std[index : index + profile.n_slices],
                charges_fine_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol=self.rtol,
                atol=self.atol,
            )

    def test_charges_from_fine_to_coarse(self):
        charges_std = (
            self.profile_std.beam.ratio
            * self.profile_std.beam.particle.charge
            * e
            * np.copy(self.profile_std.n_macroparticles)
        )

        charges_sparse = (
            self.profile_sparse.beam.ratio
            * self.profile_sparse.beam.particle.charge
            * e
            * np.copy(self.profile_sparse.n_macroparticles)
        )
        I_f_std = (
            2.0
            * charges_std
            * np.cos(self.omega * self.profile_std.bin_centers)
        )
        Q_f_std = (
            -2.0
            * charges_std
            * np.sin(self.omega * self.profile_std.bin_centers)
        )
        charges_fine_std = I_f_std + 1j * Q_f_std

        I_f_sparse = (
            2.0
            * charges_sparse
            * np.cos(self.omega * self.profile_sparse.bin_centers)
        )
        Q_f_sparse = (
            -2.0
            * charges_sparse
            * np.sin(self.omega * self.profile_sparse.bin_centers)
        )
        charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse

        charges_coarse_std = charges_from_fine_to_coarse(
            T_s=self.T_s,
            charges_fine=charges_fine_std,
            dT=0,
            n_points=self.n_points,
            omega_c=self.omega,
            profile_bin_centers=self.profile_std.bin_centers,
        )

        order = np.argsort(self.profile_sparse.bin_centers)
        profile_bin_centers = self.profile_sparse.bin_centers[order]
        profile_n_macroparticles = self.profile_sparse.n_macroparticles[order]
        extra_bins = np.arange(
            profile_bin_centers[-1],
            profile_bin_centers[-1] + 2 * self.T_s + 0 + np.pi / self.omega,
            step=self.profile_sparse.bin_size,
        )
        profile_bin_centers_for_coarse = np.concatenate(
            (profile_bin_centers, extra_bins)
        )
        profile_n_macroparticles_for_coarse = np.concatenate(
            (profile_n_macroparticles, np.zeros(len(extra_bins)))
        )
        charges = (
            self.profile_sparse.beam.ratio
            * self.profile_sparse.beam.particle.charge
            * e
            * np.copy(profile_n_macroparticles_for_coarse)
        )
        I_f = (
            2.0 * charges * np.cos(self.omega * profile_bin_centers_for_coarse)
        )
        Q_f = (
            -2.0
            * charges
            * np.sin(self.omega * profile_bin_centers_for_coarse)
        )
        charges_fine_for_coarse_grid = I_f + 1j * Q_f

        charges_coarse_sparse = charges_from_fine_to_coarse(
            T_s=self.T_s,
            charges_fine=charges_fine_for_coarse_grid,
            dT=0,
            n_points=self.n_points,
            omega_c=self.omega,
            profile_bin_centers=profile_bin_centers_for_coarse,
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

        rf_current_sparse = rf_beam_current(
            self.profile_sparse,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            external_reference=True,
            dT=0,
        )
        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                rf_current_std[index : index + profile.n_slices],
                rf_current_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol=self.rtol,
                atol=self.atol,
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

        rf_current_sparse, rf_current_coarse_sparse = rf_beam_current(
            self.profile_sparse,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            downsample=downsample_dict,
            external_reference=True,
            dT=0,
        )

        for p, profile in enumerate(self.profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.profile_std.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                rf_current_std[index : index + profile.n_slices],
                rf_current_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol=self.rtol,
                atol=self.atol,
            )

        np.testing.assert_allclose(
            rf_current_coarse_std,
            rf_current_coarse_sparse,
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_muliturn_injection_bin_centers_match(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()
            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
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

    def test_muliturn_injection_n_macroparticles_match(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()
            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    self.profile_std.bin_centers[
                        index : index + profile.n_slices
                    ],
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

    def test_muliturn_injection_charges_fine_grid(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()

            charges_std = (
                self.profile_std.beam.ratio
                * self.profile_std.beam.particle.charge
                * e
                * np.copy(self.profile_std.n_macroparticles)
            )

            charges_sparse = (
                self.profile_sparse.beam.ratio
                * self.profile_sparse.beam.particle.charge
                * e
                * np.copy(self.profile_sparse.n_macroparticles)
            )
            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    charges_std[index : index + profile.n_slices],
                    charges_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="charges differ between "
                    "standard Profile and SparseBatch for the same beam, profile number "
                    f"{p}",
                )
            tot_charges = (
                np.sum(self.profile_std.n_macroparticles)
                / self.profile_std.beam.n_macroparticles
                * self.profile_std.beam.intensity
            )
            tot_charges_sparse = (
                np.sum(self.profile_sparse.n_macroparticles)
                / self.profile_sparse.beam.n_macroparticles
                * self.profile_sparse.beam.intensity
            )
            self.assertEqual(tot_charges, tot_charges_sparse)

            I_f_std = (
                2.0
                * charges_std
                * np.cos(self.omega * self.profile_std.bin_centers)
            )
            Q_f_std = (
                -2.0
                * charges_std
                * np.sin(self.omega * self.profile_std.bin_centers)
            )
            charges_fine_std = I_f_std + 1j * Q_f_std

            I_f_sparse = (
                2.0
                * charges_sparse
                * np.cos(self.omega * self.profile_sparse.bin_centers)
            )
            Q_f_sparse = (
                -2.0
                * charges_sparse
                * np.sin(self.omega * self.profile_sparse.bin_centers)
            )
            charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse
            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    I_f_std[index : index + profile.n_slices],
                    I_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )
                np.testing.assert_allclose(
                    Q_f_std[index : index + profile.n_slices],
                    Q_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )
                np.testing.assert_allclose(
                    charges_fine_std[index : index + profile.n_slices],
                    charges_fine_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )

    def test_muliturn_injection_from_fine_to_coarse(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()
            self.profile_sparse.track()

            self.test_n_macroparticles_match()

            charges_std = (
                self.profile_std.beam.ratio
                * self.profile_std.beam.particle.charge
                * e
                * np.copy(self.profile_std.n_macroparticles)
            )

            charges_sparse = (
                self.profile_sparse.beam.ratio
                * self.profile_sparse.beam.particle.charge
                * e
                * np.copy(self.profile_sparse.n_macroparticles)
            )
            I_f_std = (
                2.0
                * charges_std
                * np.cos(self.omega * self.profile_std.bin_centers)
            )
            Q_f_std = (
                -2.0
                * charges_std
                * np.sin(self.omega * self.profile_std.bin_centers)
            )
            charges_fine_std = I_f_std + 1j * Q_f_std

            I_f_sparse = (
                2.0
                * charges_sparse
                * np.cos(self.omega * self.profile_sparse.bin_centers)
            )
            Q_f_sparse = (
                -2.0
                * charges_sparse
                * np.sin(self.omega * self.profile_sparse.bin_centers)
            )
            charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse

            charges_coarse_std = charges_from_fine_to_coarse(
                T_s=self.T_s,
                charges_fine=charges_fine_std,
                dT=0,
                n_points=self.n_points,
                omega_c=self.omega,
                profile_bin_centers=self.profile_std.bin_centers,
            )

            order = np.argsort(self.profile_sparse.bin_centers)
            profile_bin_centers = self.profile_sparse.bin_centers[order]
            profile_n_macroparticles = self.profile_sparse.n_macroparticles[
                order
            ]

            extra_bins = np.arange(
                profile_bin_centers[-1],
                profile_bin_centers[-1]
                + 2 * self.T_s
                + 0
                + np.pi / self.omega,
                step=self.profile_sparse.bin_size,
            )
            profile_bin_centers_for_coarse = np.concatenate(
                (profile_bin_centers, extra_bins)
            )
            profile_n_macroparticles_for_coarse = np.concatenate(
                (profile_n_macroparticles, np.zeros(len(extra_bins)))
            )
            charges = (
                self.profile_sparse.beam.ratio
                * self.profile_sparse.beam.particle.charge
                * e
                * np.copy(profile_n_macroparticles_for_coarse)
            )
            I_f = (
                2.0
                * charges
                * np.cos(self.omega * profile_bin_centers_for_coarse)
            )
            Q_f = (
                -2.0
                * charges
                * np.sin(self.omega * profile_bin_centers_for_coarse)
            )
            charges_fine_for_coarse_grid = I_f + 1j * Q_f

            charges_coarse_sparse = charges_from_fine_to_coarse(
                T_s=self.T_s,
                charges_fine=charges_fine_for_coarse_grid,
                dT=0,
                n_points=self.n_points,
                omega_c=self.omega,
                profile_bin_centers=profile_bin_centers_for_coarse,
            )

            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    I_f_std[index : index + profile.n_slices],
                    I_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )
                np.testing.assert_allclose(
                    Q_f_std[index : index + profile.n_slices],
                    Q_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )
                np.testing.assert_allclose(
                    charges_fine_std[index : index + profile.n_slices],
                    charges_fine_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )
                np.testing.assert_allclose(
                    np.sum(charges_fine_std[index : index + profile.n_slices]),
                    np.sum(
                        charges_fine_sparse[
                            p * profile.n_slices : (p + 1) * profile.n_slices
                        ]
                    ),
                    rtol=self.rtol,
                    atol=self.atol,
                )
            np.testing.assert_allclose(
                charges_coarse_std,
                charges_coarse_sparse,
                rtol=self.rtol,
                atol=self.atol,
            )

    def test_muliturn_injection_rf_beam_current(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()

            rf_current_std = rf_beam_current(
                self.profile_std,
                self.omega,
                self.ring.t_rev[0],
                lpf=False,
                external_reference=True,
                dT=0,
            )

            rf_current_sparse = rf_beam_current(
                self.profile_sparse,
                self.omega,
                self.ring.t_rev[0],
                lpf=False,
                external_reference=True,
                dT=0,
            )
            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    rf_current_std[index : index + profile.n_slices],
                    rf_current_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )

    def test_muliturn_injection_downsampling(self):
        injected_batches = 1
        for k in range(number_of_batches - injected_batches):
            self.beam, self.profile_sparse, injected_batches = update_beam(
                beam=self.beam,
                ring=self.ring,
                rf_station=self.rf,
                sparse_profile=self.profile_sparse,
                injected_batches=injected_batches,
                seed=1234,
            )
            print(f"Injection #{k + 1}")

            self.assertEqual(injected_batches, k + 2)

            self.profile_std.track()
            # self.profile_sparse.track()
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

            rf_current_sparse, rf_current_coarse_sparse = rf_beam_current(
                self.profile_sparse,
                self.omega,
                self.ring.t_rev[0],
                lpf=False,
                downsample=downsample_dict,
                external_reference=True,
                dT=0,
            )

            for p, profile in enumerate(self.profile_sparse.profiles_list):
                index = np.argmin(
                    np.abs(
                        self.profile_std.bin_centers - profile.bin_centers[0]
                    )
                )
                np.testing.assert_allclose(
                    rf_current_std[index : index + profile.n_slices],
                    rf_current_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                    rtol=self.rtol,
                    atol=self.atol,
                )

            np.testing.assert_allclose(
                rf_current_coarse_std,
                rf_current_coarse_sparse,
                rtol=self.rtol,
                atol=self.atol,
            )


if __name__ == "__main__":
    unittest.main()
