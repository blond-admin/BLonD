# coding: utf8
"""
Test functions of the LHC Cavity Loop.

These tests ensure the LHC cavity loop provides the same results and outputs
with a standard beam profile and a sparse profile.

To do so, the tests ensure the standard profile and the sparse profile have
equivalent bin_centers and other parameters. Then, the COARSE grid and the
FINE grid for both profiles are compared, for different functions, towards
the full .track() method.

The current implementation of the CAvity Loops with sparse profile ensure a
max relative error of 1e-3 on the fine grid antenna voltage.

These test functions have been partially generated with the help of a LLM.

Author:
Lina Valle
"""

import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import SparseBatch
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCCavityLoopCommissioning


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
number_of_batches = 10  # Length of the batch [number of batches]
batch_spacing = 5  # Number of empty buckets between each batch [number of rf
# buckets]

number_of_bunches_per_batch = 3  # number of bunches per batch i.e. per profile
bunch_spacing = 1  # Number of empty buckets between each bunch

total_length_batch = (number_of_bunches_per_batch +
                      (number_of_bunches_per_batch-1) * bunch_spacing)
assert(total_length_batch <= batch_spacing)

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


def build_open_drive_commissioning():
    # enable_klystron=False to match the reference values below, which were
    # captured without the klystron bandwidth-limiting FIR filter.
    return LHCCavityLoopCommissioning(open_drive=True, enable_klystron=False)


class TestLHCCavityLoopStandardProfile(unittest.TestCase):
    """Regression tests for LHCCavityLoop built with a standard Profile.

    These reproduce the reference values from BLonD's own
    `TestLHCOpenDrive` unittest, to confirm that the no-beam / open-drive
    behaviour of LHCCavityLoop was not altered while adding SparseBatch
    support.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.profile = build_standard_profile(self.beam, self.rf, n_slices=1000)
        self.RFFB = build_open_drive_commissioning()
        self.f_c = self.rf.omega_rf[0, 0] / (2 * np.pi)

    def _make_loop(self, **overrides):
        kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.2778,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        kwargs.update(overrides)
        return LHCCavityLoop(self.rf, self.profile, **kwargs)

    def test_open_drive_default_Q_L(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)

    def test_open_drive_higher_Q_L(self):
        CL = self._make_loop(Q_L=60000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)

    def test_open_drive_higher_R_over_Q(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=90)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)

    def test_fine_grid_array_shapes(self):
        CL = self._make_loop()
        self.assertEqual(CL.V_ANT_FINE.shape[0], self.profile.n_slices + 1)
        self.assertEqual(CL.I_GEN_FINE.shape[0], self.profile.n_slices + 1)


class TestLHCCavityLoopSparseProfile(unittest.TestCase):
    """The same regression tests as above, but built with a SparseBatch
    profile. Since `track_one_turn` / the open-drive path never touches
    the profile object, these must reproduce exactly the same reference
    numbers as the standard-Profile case.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.profile = build_sparse_profile(
            self.beam, self.rf, n_slices=1000
        )
        self.RFFB = build_open_drive_commissioning()
        self.f_c = self.rf.omega_rf[0, 0] / (2 * np.pi)

    def _make_loop(self, **overrides):
        kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.2778,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        kwargs.update(overrides)
        return LHCCavityLoop(self.rf, self.profile, **kwargs)

    def test_sparse_profile_is_recognised(self):
        self.assertIsInstance(self.profile, SparseBatch)
        self.assertEqual(len(self.profile.profiles_list), number_of_batches)

    def test_open_drive_default_Q_L(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)

    def test_open_drive_higher_Q_L(self):
        CL = self._make_loop(Q_L=60000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)

    def test_open_drive_higher_R_over_Q(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=90)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)

    def test_fine_grid_array_shapes(self):
        CL = self._make_loop()
        self.assertEqual(CL.V_ANT_FINE.shape[0], self.profile.n_slices + 1)
        self.assertEqual(CL.I_GEN_FINE.shape[0], self.profile.n_slices + 1)


class TestProfileEquivalence(unittest.TestCase):
    """Sanity check the premise the comparison tests below rely on: a
    single-batch, full-turn SparseBatch must slice the same beam into
    exactly the same bins as a standard Profile with the same cuts.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.n_slices = 2000
        self.standard = build_standard_profile(self.beam, self.rf, self.n_slices)
        self.sparse = build_sparse_profile(
            self.beam, self.rf, self.n_slices
        )

        self.assertAlmostEqual(self.standard.bin_centers[0],
                         self.sparse.bin_centers[0],
                               places = 12)
    def test_bin_centers_match(self):
        for p, profile in enumerate(
            self.sparse.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.standard.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(
                self.standard.bin_centers[index],
                self.sparse.bin_centers[
                    p * profile.n_slices],
                rtol = 1e-12, atol =  1e-12,
            )

    def test_bin_size_matches(self):
        self.assertAlmostEqual(self.standard.bin_size, self.sparse.bin_size,
                               places=15)

class TestLHCCavityLoopConsistencyBetweenProfileTypes(unittest.TestCase):
    """Core comparison requested: for the *same* beam and the *same* time
    grid, LHCCavityLoop should produce the same coarse- and fine-grid
    signals whether it is fed a standard Profile or an equivalent SparseBatch.
    """

    N_SLICES = 4 * HARMONIC_NUMBER // 5  # fine relative to the coarse (n_coarse) grid

    def setUp(self,
              show_plot : bool = False):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_sparse_profile(
            self.beam, self.rf, self.N_SLICES
        )

        self.RFFB = LHCCavityLoopCommissioning()  # default, closed loop
        self.f_c = self.rf.omega_rf[0, 0] / (2 * np.pi)

        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std, **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse, **common_kwargs)

        self.rtol = 1e-10
        self.atol = 1e-10
        if show_plot:
            fig, ax = plt.subplots(nrows=3, figsize=(10, 5))
            ax[0].plot(
                self.profile_std.bin_centers * 1e6,
                self.profile_std.n_macroparticles,
                label="Standard profile",
            )
            for profile_ind in self.profile_sparse.profiles_list:
                ax[0].plot(
                    profile_ind.bin_centers * 1e6,
                    profile_ind.n_macroparticles,
                    ls="--",
                    label="Sparse profile",
                )
            ax[0].set_xlabel(r"$\Delta t$ [$\mu$s]")
            ax[0].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
            ax[0].legend()
            ax[0].set_yticks([])

            ax[1].plot(
                self.profile_std.bin_centers * 1e6,
                self.profile_std.n_macroparticles,
                label="Standard profile",
            )
            for profile_ind in self.profile_sparse.profiles_list:
                ax[1].plot(
                    profile_ind.bin_centers * 1e6,
                    profile_ind.n_macroparticles,
                    ls="--",
                    label="Sparse profile",
                )
            ax[1].set_xlabel(r"$\Delta t$ [$\mu$s]")
            ax[1].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
            ax[1].legend()
            ax[1].set(xlim=[24.95, 24.954])
            ax[1].set_yticks([])

            ax[2].plot(
                self.profile_std.bin_centers * 1e6,
                self.profile_std.n_macroparticles,
                label="Standard profile",
            )
            for profile_ind in self.profile_sparse.profiles_list:
                ax[2].plot(
                    profile_ind.bin_centers * 1e6,
                    profile_ind.n_macroparticles,
                    ls="--",
                    label="Sparse profile",
                )
            ax[2].set_xlabel(r"$\Delta t$ [$\mu$s]")
            ax[2].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
            ax[2].legend()
            ax[2].set(xlim=[27.445, 27.449])
            ax[2].set_yticks([])
            plt.show()
    def test_rf_beam_current_consistent(self):
        """The beam current seen on the coarse and fine grids must not
        depend on which profile representation was used."""
        self.CL_standard.rf_beam_current(lpf=self.CL_standard.lpf)
        self.CL_sparse.rf_beam_current(lpf=self.CL_sparse.lpf)

        np.testing.assert_allclose(
            self.CL_standard.I_BEAM_COARSE[
                -self.CL_standard.n_coarse:
            ],
            self.CL_sparse.I_BEAM_COARSE[
                -self.CL_sparse.n_coarse:
            ],
            rtol= self.rtol,
            atol= self.atol,
            err_msg="I_BEAM_COARSE differs between standard Profile and "
                    "single-batch SparseBatch for the same beam.",
        )
        for p, profile in enumerate(
            self.CL_sparse.profile.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                self.CL_sparse.profile.bin_centers[
                    p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                self.CL_standard.I_BEAM_FINE[index:index + profile.n_slices],
                self.CL_sparse.I_BEAM_FINE[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_BEAM_FINE differs between standard Profile and "
                "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )

    def test_coarse_antenna_voltage_consistent_after_one_track(self):
        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std,
                                         **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse,
                                       **common_kwargs)
        self.CL_standard.track()
        self.CL_sparse.track()

        np.testing.assert_allclose(
            self.CL_standard.V_ANT_COARSE,
            self.CL_sparse.V_ANT_COARSE,
            rtol= self.rtol,
            atol= self.atol,
            err_msg="V_ANT_COARSE differs between standard Profile and "
            "single-batch SparseBatch for the same beam.",
        )
        np.testing.assert_allclose(
            self.CL_standard.I_GEN_COARSE,
            self.CL_sparse.I_GEN_COARSE,
            rtol= self.rtol,
            atol= self.atol,
            err_msg="I_GEN_COARSE differs between standard Profile and "
            "single-batch SparseBatch for the same beam.",
        )
    def test_fine_grid_antenna_voltage_consistent_track_one_turn(self):
        """The track_one_turn() function should provide the same output for
        a standard and a sparse profile.
        """
        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std,
                                         **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse,
                                       **common_kwargs)

        self.CL_standard.track_one_turn()
        self.CL_sparse.track_one_turn()

        for p, profile in enumerate(
            self.CL_sparse.profile.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )
            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                self.CL_sparse.profile.bin_centers[
                    p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                self.CL_standard.I_BEAM_FINE[index:index + profile.n_slices],
                self.CL_sparse.I_BEAM_FINE[
                    p * profile.n_slices: (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_BEAM_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )
            np.testing.assert_allclose(
                self.CL_standard.I_GEN_FINE[index:index + profile.n_slices],
                self.CL_sparse.I_GEN_FINE[
                    p * profile.n_slices: (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_GEN_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )
            np.testing.assert_allclose(
                self.CL_standard.V_ANT_FINE[index:index + profile.n_slices],
                self.CL_sparse.V_ANT_FINE[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="V_ANT_FINE differs between standard Profile and "
                "single-batch SparseBatch for the same beam. Profile "
                        f"{p}",
            )

    def test_fine_grid_generator_current_consistent_fine_grid_disabled(self):
        """Interpolation of the generator fine grid current should provide
        the same output for the standard profile and the sparse profile.
        Since the first element of the I_GEN_FINE is common, the test only
        covers the the n_slices per profile (hence the  [
                    p * profile.n_slices + 1: (p + 1) * profile.n_slices +1
                ]).
        """
        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std,
                                         **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse,
                                       **common_kwargs)

        self.CL_standard.disable_fine_grid = True
        self.CL_sparse.disable_fine_grid = True
        self.CL_standard.track()
        self.CL_sparse.track()

        for p, profile in enumerate(
            self.CL_sparse.profile.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )

            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                self.CL_sparse.profile.bin_centers[
                    p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                self.CL_standard.I_BEAM_FINE[index:index +
                                                      profile.n_slices ],
                self.CL_sparse.I_BEAM_FINE[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_BEAM_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )

            np.testing.assert_allclose(
                self.CL_standard.I_GEN_FINE[index + 1:index +
                                                      profile.n_slices + 1],
                self.CL_sparse.I_GEN_FINE[
                    p * profile.n_slices + 1: (p + 1) * profile.n_slices +1
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_GEN_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )

            np.testing.assert_allclose(
                self.CL_standard.V_ANT_FINE[index + 1:index +
                                                      profile.n_slices + 1],
                self.CL_sparse.V_ANT_FINE[
                    p * profile.n_slices + 1: (p + 1) * profile.n_slices +1
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="V_ANT_FINE differs between standard Profile and "
                "single-batch SparseBatch for the same beam -- see the "
                "docstring of this test for the likely cause. Profile "
                        f"{p}",
            )

    def test_fine_grid_cavity_response_inputs(self):
        """The input to the cavity_response_fine_matrix function should be
        identical for the standard profile and the first sparse profile.
        """
        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std,
                                         **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse,
                                       **common_kwargs)

        self.CL_standard.track()
        self.CL_sparse.track()
        np.testing.assert_equal(self.CL_standard.samples_fine,
                                self.CL_sparse.samples_fine,
                                )
        t_at_init = (self.CL_standard.profile.bin_centers[0] -
                     self.CL_standard.profile.bin_size)
        t_at_init_sparse = (self.CL_sparse.profile.bin_centers[0] -
                     self.CL_sparse.profile.bin_size)
        np.testing.assert_equal(t_at_init,
            t_at_init_sparse
        )

        V_A_init = interp1d(
            np.concatenate(
                (
                    self.CL_standard.rf_centers - self.CL_standard.T_s *
                    self.CL_standard.n_coarse,
                    self.CL_standard.rf_centers,
                )
            ),
            self.CL_standard.V_ANT_COARSE,
            fill_value="extrapolate",
        )(t_at_init)
        V_A_init_sparse = interp1d(
            np.concatenate(
                (
                    self.CL_sparse.rf_centers - self.CL_sparse.T_s *
                    self.CL_sparse.n_coarse,
                    self.CL_sparse.rf_centers,
                )
            ),
            self.CL_sparse.V_ANT_COARSE,
            fill_value="extrapolate",
        )(t_at_init_sparse)

        np.testing.assert_equal(V_A_init,
                                V_A_init_sparse
                                )

        I_gen_init = interp1d(
            np.concatenate(
                (
                    self.CL_standard.rf_centers - self.CL_standard.T_s *
                    self.CL_standard.n_coarse,
                    self.CL_standard.rf_centers,
                )
            ),
            self.CL_standard.I_GEN_COARSE,
            fill_value="extrapolate",
        )(t_at_init)
        I_gen_init_sparse = interp1d(
            np.concatenate(
                (
                    self.CL_sparse.rf_centers - self.CL_sparse.T_s *
                    self.CL_sparse.n_coarse,
                    self.CL_sparse.rf_centers,
                )
            ),
            self.CL_sparse.I_GEN_COARSE,
            fill_value="extrapolate",
        )(t_at_init_sparse)
        np.testing.assert_equal(I_gen_init,
                                I_gen_init_sparse
                                )
        np.testing.assert_allclose(
            self.CL_standard.I_GEN_FINE[0],
            self.CL_sparse.I_GEN_FINE[0],
            rtol=self.rtol,
            atol=self.atol,
            err_msg="I_GEN_FINE first element differs between standard "
                    "Profile and "
                    "single-batch SparseBatch for the same beam",
        )
        for p, profile in enumerate(
            self.CL_sparse.profile.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )

            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                self.CL_sparse.profile.bin_centers[
                    p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                self.CL_standard.I_BEAM_FINE[index:index +
                                                      profile.n_slices ],
                self.CL_sparse.I_BEAM_FINE[
                    p * profile.n_slices : (p + 1) * profile.n_slices
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_BEAM_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )

            np.testing.assert_allclose(
                self.CL_standard.I_GEN_FINE[index + 1:index +
                                                      profile.n_slices + 1],
                self.CL_sparse.I_GEN_FINE[
                    p * profile.n_slices + 1: (p + 1) * profile.n_slices +1
                ],
                rtol= self.rtol,
                atol= self.atol,
                err_msg="I_GEN_FINE differs between standard Profile and "
                        "single-batch SparseBatch for the same beam, profile number "
                        f"{p}",
            )

    def test_fine_grid_generator_current_consistent(self):
        """The fine-grid antenna voltage computed by
        `cavity_response_fine_matrix` should be numerically the same for
        a standard Profile and an equivalent SparseBatch.
        Since the first element of the V_ANT_FINE is common, the test only
        covers the the n_slices per profile (hence the  [
                    p * profile.n_slices + 1: (p + 1) * profile.n_slices +1
                ]).
        """
        common_kwargs = dict(
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.0,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        self.CL_standard = LHCCavityLoop(self.rf, self.profile_std,
                                         **common_kwargs)
        self.CL_sparse = LHCCavityLoop(self.rf, self.profile_sparse,
                                       **common_kwargs)

        self.CL_standard.track()
        self.CL_sparse.track()

        np.testing.assert_allclose(
            self.CL_standard.V_ANT_FINE[0],
            self.CL_sparse.V_ANT_FINE[0],
            rtol=self.rtol,
            atol=self.atol,
            err_msg="V_ANT_FINE first element differs between standard "
                    "Profile and "
                    "single-batch SparseBatch for the same beam",
        )
        for p, profile in enumerate(
            self.CL_sparse.profile.profiles_list
        ):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )

            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                self.CL_sparse.profile.bin_centers[
                    p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
            )


            np.testing.assert_allclose(
                self.CL_standard.V_ANT_FINE[index + 1:index +
                                                      profile.n_slices + 1],
                self.CL_sparse.V_ANT_FINE[
                    p * profile.n_slices + 1 : (p + 1) * profile.n_slices +1
                ],
                rtol= 1e-3,
                atol= 1e-7,
                err_msg="V_ANT_FINE differs between standard Profile and "
                "single-batch SparseBatch for the same beam -- see the "
                "docstring of this test for the likely cause. Profile "
                        f"{p}",
            )


if __name__ == "__main__":
    unittest.main()
