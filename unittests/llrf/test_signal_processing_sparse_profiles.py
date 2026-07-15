# coding: utf8
"""
Test functions of the rf beam current with sparse profiles.

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
from blond.llrf.cavity_feedback import LHCCavityLoopCommissioning
from blond.llrf.signal_processing import rf_beam_current,charges_from_fine_to_coarse

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
BUNCH_INTENSITY = 1e11
BUNCH_SIGMA_DT = 0.5e-9
# Warning: for a large number of bunches, the bin_size difference between
# the sparse profile and the standard profile induces slight mismatches
# between the indexes. Tests will artificially fail because of this difference.
number_of_bunches = 10  # Length of the batch [number of bunches]
if number_of_bunches > 50:
    warnings.warn(message="Warning: for a large number of bunches, the bin_size "
                          "difference between the sparse profile and the "
                          "standard profile induces slight mismatches between the indexes. "
                          "Tests might artificially fail because of this "
                          "difference.")
bunch_spacing = 10  # Bunch spacing [number of rf buckets]

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
    N_m = N_MACROPARTICLES * number_of_bunches
    N_p = BUNCH_INTENSITY * number_of_bunches
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
    for i in range(number_of_bunches):
        beam.dE[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] = (
            single_bunch.dE
        )
        beam.dt[i * N_MACROPARTICLES: (i + 1) * N_MACROPARTICLES] = (
                single_bunch.dt + i * bunch_spacing * rf_station.t_rf[0, 0]
        )
    return beam


def build_standard_profile(beam, rf_station, n_slices):
    """A standard Profile covering the injected bunches and an extra bucket."""
    profile = Profile(
        beam,
        CutOptions(cut_left=0.0, cut_right=(bunch_spacing * number_of_bunches + 1)
            * rf_station.t_rf[
                0,
                0,
            ],
                   n_slices=n_slices * (bunch_spacing * number_of_bunches + 1)),
    )
    profile.track()
    return profile


def build_full_turn_sparse_profile(beam, rf_station, n_slices):
    """A SparseBatch profile with a profile per number of bunches.
    """
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(number_of_bunches):
        batch_list[k * bunch_spacing] = 1
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(bunch_spacing/2)+1) * n_slices,
        batch_list=batch_list,
        batch_length=int(bunch_spacing/2)+1,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


def build_open_drive_commissioning():
    # enable_klystron=False to match the reference values below, which were
    # captured without the klystron bandwidth-limiting FIR filter.
    return LHCCavityLoopCommissioning(open_drive=True, enable_klystron=False)


class TestRFBeamCurrent(unittest.TestCase):
    N_SLICES = 4 * HARMONIC_NUMBER // 5  # fine relative to the coarse (n_coarse) grid
    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.omega = 2 * np.pi * 200.222e6

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_full_turn_sparse_profile(
            self.beam, self.rf, self.N_SLICES
        )

        self.T_s = 5 * self.rf.t_rev[0] / self.rf.harmonic[0, 0]
        self.n_points = 1000
        self.rtol = 1e-10
        self.atol = 1e-10

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

        I_f_std = 2.0 * charges_std * np.cos(self.omega *
                                             self.profile_std.bin_centers)
        Q_f_std = -2.0 * charges_std * np.sin(self.omega *
                                              self.profile_std.bin_centers)
        self.charges_fine_std = I_f_std + 1j * Q_f_std

        I_f_sparse = 2.0 * charges_sparse * np.cos(self.omega *
                                                   self.profile_sparse.bin_centers)
        Q_f_sparse = -2.0 * charges_sparse * np.sin(self.omega *
                                                    self.profile_sparse.bin_centers)
        self.charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse
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
        for p, profile in enumerate(
            self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(charges_std[index:index +
                                                      profile.n_slices ],
                                          charges_sparse[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                                          rtol = self.rtol,
                                          atol= self.atol,
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

        I_f_std = 2.0 * charges_std * np.cos(self.omega *
                                             self.profile_std.bin_centers)
        Q_f_std = -2.0 * charges_std * np.sin(self.omega *
                                             self.profile_std.bin_centers)
        self.charges_fine_std = I_f_std + 1j * Q_f_std

        I_f_sparse = 2.0 * charges_sparse * np.cos(self.omega *
                                             self.profile_sparse.bin_centers)
        Q_f_sparse = -2.0 * charges_sparse * np.sin(self.omega *
                                              self.profile_sparse.bin_centers)
        self.charges_fine_sparse = I_f_sparse + 1j * Q_f_sparse
        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(I_f_std[index:index +
                                                          profile.n_slices ],
                                       I_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       )
            np.testing.assert_allclose(Q_f_std[index:index +
                                                          profile.n_slices ],
                                       Q_f_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       )
            np.testing.assert_allclose(self.charges_fine_std[index:index +
                                                          profile.n_slices ],
                                       self.charges_fine_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       )

    def test_charges_from_fine_to_coarse(self):
        charges_coast_std = charges_from_fine_to_coarse(T_s = self.T_s,
                                                        charges_fine =
                                                        self.charges_fine_std,
                                                        dT = 0,
                                                        n_points=self.n_points,
                                                        omega_c=self.omega,
                                                        profile=self.profile_std,
                                                        )

        charges_coast_sparse = charges_from_fine_to_coarse(T_s = self.T_s,
                                                        charges_fine =
                                                        self.charges_fine_sparse,
                                                        dT = 0,
                                                        n_points=self.n_points,
                                                        omega_c=self.omega,
                                                        profile=self.profile_sparse,
                                                        )

        np.testing.assert_allclose(charges_coast_std,
                                   charges_coast_sparse,
                                   rtol = self.rtol,
                                   atol = self.atol,
                                   )
    def test_rf_beam_current(self):
        rf_current_std = (
            rf_beam_current(
                self.profile_std,
                self.omega,
                self.ring.t_rev[0],
                lpf=False,
                external_reference=True,
                dT=0,
            ))

        rf_current_sparse = rf_beam_current(
            self.profile_sparse,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            external_reference=True,
            dT=0,
        )
        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(rf_current_std[index:index +
                                                          profile.n_slices ],
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

        rf_current_std, rf_current_coarse_std = (
            rf_beam_current(
            self.profile_std,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            downsample=downsample_dict,
            external_reference=True,
            dT=0,
        ))

        rf_current_sparse, rf_current_coarse_sparse = rf_beam_current(
            self.profile_sparse,
            self.omega,
            self.ring.t_rev[0],
            lpf=False,
            downsample=downsample_dict,
            external_reference=True,
            dT=0,
        )

        for p, profile in enumerate(
                self.profile_sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.profile_std.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(rf_current_std[index:index +
                                                          profile.n_slices ],
                                       rf_current_sparse[
                        p * profile.n_slices : (p + 1) * profile.n_slices
                    ],
                                       rtol=self.rtol,
                                       atol=self.atol,
                                       )

        np.testing.assert_allclose(rf_current_coarse_std,
                                   rf_current_coarse_sparse,
                                   rtol=self.rtol,
                                   atol=self.atol,
                                   )

if __name__ == "__main__":
    unittest.main()
