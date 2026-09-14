# coding: utf8
"""
Test functions of the LHC Cavity Loop.

These tests ensure the LHC cavity loop provides the same results and outputs
with a standard beam profile and a sparse profile.

To do so, the tests ensure the standard profile and the sparse profile have
equivalent bin_centers and other parameters. Then, the COARSE grid and the
FINE grid for both profiles are compared, for different functions, towards
the full .track() method.

The current implementation of the Cavity Loops with sparse profile
reproduces the fine grid antenna voltage of the standard profile to
rounding precision (relative error of 1e-10 in these tests).

These test functions have been partially generated with the help of a LLM.

Author:
Lina Valle
"""

import unittest
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import (
    SparseBatch,
    SparseBucket,
    SparseProfileBaseClass,
)
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
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
BUNCH_INTENSITY = 1e11
BUNCH_SIGMA_DT = 0.25e-9
number_of_batches = 10  # Length of the batch [number of batches]
batch_spacing = 5  # Number of empty buckets between each batch [number of rf
# buckets]

batch_list = [0, 45, 20, 35, 10, 30, 20, 40, 5, 15]
number_of_bunches_per_batch = 3  # number of bunches per batch i.e. per profile
bunch_spacing = 1  # Number of empty buckets between each bunch

total_length_batch = (
    number_of_bunches_per_batch
    + (number_of_bunches_per_batch - 1) * bunch_spacing
)
assert total_length_batch <= batch_spacing
if number_of_batches > 10:
    warnings.warn(
        message="Warning: for a large number of batches, "
        "the bin_size "
        "difference between the sparse profile and the "
        "standard profile induces slight mismatches between the indexes. "
        "Tests might artificially fail because of this "
        "difference."
    )


def batch_start(k):
    """Bucket index of the first bunch of batch k, from batch_list: batches
    are injected out of bucket order, and bucket 20 is injected twice."""
    return batch_list[k]


def bunch_buckets(k):
    """Bucket indices of the bunches of batch k."""
    return [
        batch_start(k) + i * bunch_spacing
        for i in range(number_of_bunches_per_batch)
    ]


def batch_pattern(injected_batches):
    """Filling pattern of the first bucket of each injected batch."""
    pattern = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
        pattern[batch_start(k)] = 1
    return pattern


def bunch_pattern(injected_batches):
    """Filling pattern of every bunch of the injected batches."""
    pattern = np.zeros(HARMONIC_NUMBER)
    for k in range(injected_batches):
        pattern[bunch_buckets(k)] = 1
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


def build_beam(ring, rf_station, injected_batches, seed=1234):
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
        beam.dt[block] = batch.dt + batch_start(k) * rf_station.t_rf[0, 0]
    return beam


def inject_batch(
    beam, ring, rf_station, profiles_sparse, injected_batches, seed=1234
):
    """Inject batch number injected_batches into the beam, register it in
    every sparse profile and re-track them. Returns the updated number of
    injected batches."""
    k = injected_batches
    batch = build_batch(ring, rf_station, seed)
    beam.add_particles(
        [batch.dt + batch_start(k) * rf_station.t_rf[0, 0], batch.dE]
    )
    for profile_sparse in profiles_sparse:
        if isinstance(profile_sparse, SparseBatch):
            profile_sparse.update_batch_list(
                updated_batch_list=batch_pattern(k + 1)
            )
        else:
            # Several bunches are new at once: give them in injection order
            previous = profile_sparse.bunch_list
            new_bunches = [b for b in bunch_buckets(k) if previous[b] == 0]
            profile_sparse.update_bunch_list(
                updated_bunch_list=bunch_pattern(k + 1),
                new_bunch_indices=new_bunches or None,
            )
        profile_sparse.track()
    return k + 1


def build_standard_profile(beam, rf_station, n_slices):
    """A standard Profile covering all the batches and some extra buckets."""
    n_buckets = batch_spacing * number_of_batches + 10
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


def build_sparse_profile(beam, rf_station, n_slices, injected_batches):
    """A SparseBatch profile with one profile per injected batch."""
    sparse_profile = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=(int(batch_spacing / 2) + 1) * n_slices,
        batch_list=batch_pattern(injected_batches),
        batch_length=int(batch_spacing / 2) + 1,
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


def build_sparse_bucket_profile(beam, rf_station, n_slices, injected_batches):
    """A SparseBucket profile with one profile (one RF bucket) per bunch of
    the injected batches."""
    sparse_profile = SparseBucket(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=n_slices,
        bunch_list=bunch_pattern(injected_batches),
        tracker_mode="onebyone",
    )
    sparse_profile.track()
    return sparse_profile


def build_open_drive_commissioning():
    # enable_klystron=False to match the reference values below, which were
    # captured without the klystron bandwidth-limiting FIR filter.
    return LHCCavityLoopCommissioning(open_drive=True, enable_klystron=False)


def build_cavity_loop(rf_station, profile, RFFB, **overrides):
    """An LHCCavityLoop on the given profile with the parameters shared by
    all tests; overrides replace individual parameters."""
    kwargs = dict(
        f_c=rf_station.omega_rf[0, 0] / (2 * np.pi),
        G_gen=1,
        I_gen_offset=0.0,
        n_cavities=8,
        n_pretrack=0,
        Q_L=20000,
        R_over_Q=45,
        tau_loop=650e-9,
        tau_otfb=1472e-9,
        RFFB=RFFB,
    )
    kwargs.update(overrides)
    return LHCCavityLoop(rf_station, profile, **kwargs)


class TestLHCCavityLoopStandardProfile(unittest.TestCase):
    """Regression tests for LHCCavityLoop built with a standard Profile.

    These reproduce the reference values from BLonD's own
    `TestLHCOpenDrive` unittest, to confirm that the no-beam / open-drive
    behaviour of LHCCavityLoop was not altered while adding sparse profile
    support.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf, 1)
        self.profile = build_standard_profile(
            self.beam, self.rf, n_slices=1000
        )
        self.RFFB = build_open_drive_commissioning()

    def _make_loop(self, **overrides):
        return build_cavity_loop(
            self.rf, self.profile, self.RFFB, I_gen_offset=0.2778, **overrides
        )

    def test_open_drive_default_Q_L(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)

    def test_open_drive_higher_Q_L(self):
        CL = self._make_loop(Q_L=60000, R_over_Q=45)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)

    def test_open_drive_higher_R_over_Q(self):
        CL = self._make_loop(Q_L=20000, R_over_Q=90)
        CL.track_one_turn()

        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        P_gen = CL.generator_power()[-1] * 1e-3

        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)

    def test_fine_grid_array_shapes(self):
        CL = self._make_loop()
        self.assertEqual(CL.V_ANT_FINE.shape[0], self.profile.n_slices + 1)
        self.assertEqual(CL.I_GEN_FINE.shape[0], self.profile.n_slices + 1)


class TestLHCCavityLoopSparseProfile(unittest.TestCase):
    """The same regression tests as above, but built with a SparseBatch and
    with a SparseBucket profile. Since `track_one_turn` / the open-drive
    path never touches the profile object, these must reproduce exactly the
    same reference numbers as the standard-Profile case.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf, 1)
        self.profile_sparse = build_sparse_profile(self.beam, self.rf, 1000, 1)
        self.profile_bucket = build_sparse_bucket_profile(
            self.beam, self.rf, 1000, 1
        )
        self.profiles_sparse = {
            "SparseBatch": self.profile_sparse,
            "SparseBucket": self.profile_bucket,
        }
        self.RFFB = build_open_drive_commissioning()

    def _make_loop(self, profile, **overrides):
        return build_cavity_loop(
            self.rf, profile, self.RFFB, I_gen_offset=0.2778, **overrides
        )

    def _check_open_drive(
        self, Q_L, R_over_Q, V_ant_ref, I_gen_ref, P_gen_ref
    ):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                CL = self._make_loop(
                    profile_sparse, Q_L=Q_L, R_over_Q=R_over_Q
                )
                CL.track_one_turn()

                V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
                I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
                P_gen = CL.generator_power()[-1] * 1e-3

                self.assertAlmostEqual(V_ant, V_ant_ref, places=7)
                self.assertAlmostEqual(I_gen, I_gen_ref, places=10)
                self.assertAlmostEqual(P_gen, P_gen_ref, places=10)

    def test_sparse_profile_is_recognised(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                self.assertIsInstance(profile_sparse, SparseProfileBaseClass)
        self.assertEqual(len(self.profile_sparse.profiles_list), 1)
        self.assertEqual(
            len(self.profile_bucket.profiles_list),
            1 * number_of_bunches_per_batch,
        )

    def test_open_drive_default_Q_L(self):
        self._check_open_drive(
            20000, 45, 0.49817991, 0.2778000000, 34.7277780000
        )

    def test_open_drive_higher_Q_L(self):
        self._check_open_drive(
            60000, 45, 1.26745787, 0.2778000000, 104.1833340000
        )

    def test_open_drive_higher_R_over_Q(self):
        self._check_open_drive(
            20000, 90, 0.99635982, 0.2778000000, 69.4555560000
        )

    def test_fine_grid_array_shapes(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                CL = self._make_loop(profile_sparse)
                self.assertEqual(
                    CL.V_ANT_FINE.shape[0], profile_sparse.n_slices + 1
                )
                self.assertEqual(
                    CL.I_GEN_FINE.shape[0], profile_sparse.n_slices + 1
                )


class TestProfileEquivalence(unittest.TestCase):
    """Sanity check the premise the comparison tests below rely on: a
    SparseBatch and a SparseBucket must slice the same beam into exactly the
    same bins as a standard Profile with the same cuts.
    """

    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf, 1)
        self.n_slices = 2000
        self.standard = build_standard_profile(
            self.beam, self.rf, self.n_slices
        )
        self.sparse = build_sparse_profile(
            self.beam, self.rf, self.n_slices, 1
        )
        self.bucket = build_sparse_bucket_profile(
            self.beam, self.rf, self.n_slices, 1
        )
        self.profiles_sparse = {
            "SparseBatch": self.sparse,
            "SparseBucket": self.bucket,
        }
        for profile_sparse in self.profiles_sparse.values():
            self.assertAlmostEqual(
                self.standard.bin_centers[0],
                profile_sparse.bin_centers[0],
                places=12,
            )
        self.rtol = 1e-12
        self.atol = 1e-12

    def _windows(self, profile_sparse):
        """Yield (p, profile, index) for each window of a sparse profile:
        the profile number, its Profile object and the index of its first
        bin in the standard profile."""
        for p, profile in enumerate(profile_sparse.profiles_list):
            index = np.argmin(
                np.abs(self.standard.bin_centers - profile.bin_centers[0])
            )
            yield p, profile, index

    def test_bin_centers_match(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                for p, profile, index in self._windows(profile_sparse):
                    np.testing.assert_allclose(
                        self.standard.bin_centers[index],
                        profile_sparse.bin_centers[p * profile.n_slices],
                        rtol=self.rtol,
                        atol=self.atol,
                    )

    def test_bin_size_matches(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                self.assertAlmostEqual(
                    self.standard.bin_size, profile_sparse.bin_size, places=15
                )

    def test_n_macroparticles_match(self):
        for name, profile_sparse in self.profiles_sparse.items():
            with self.subTest(profile=name):
                for p, profile, index in self._windows(profile_sparse):
                    np.testing.assert_allclose(
                        self.standard.bin_centers[
                            index : index + profile.n_slices
                        ],
                        profile.bin_centers,
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg="bin_centers differ between standard "
                        f"Profile and {name} for the same beam, profile "
                        f"number {p}",
                    )
                    np.testing.assert_allclose(
                        self.standard.n_macroparticles[
                            index : index + profile.n_slices
                        ],
                        profile.n_macroparticles,
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg="n_macroparticles differ between standard "
                        f"Profile and {name} for the same beam, profile "
                        f"number {p}",
                    )


class TestLHCCavityLoopConsistencyBetweenProfileTypesMultiTurnInjection(
    unittest.TestCase
):
    """Core comparison requested: for the *same* beam and the *same* time
    grid, LHCCavityLoop should produce the same coarse- and fine-grid
    signals whether it is fed a standard Profile or an equivalent sparse
    profile, a SparseBatch (one profile per batch) or a SparseBucket (one
    profile per bunch).

    The test_* methods run with a single injected batch; the
    test_muliturn_* methods inject the remaining batches one by one and
    repeat the same check after each injection.
    """

    N_SLICES = (
        4 * HARMONIC_NUMBER // 5
    )  # fine relative to the coarse (n_coarse) grid

    def setUp(self, show_plot: bool = False):
        self.ring, self.rf = build_ring_and_rf()
        self.injected_batches = 1
        self.beam = build_beam(self.ring, self.rf, self.injected_batches)

        self.profile_std = build_standard_profile(
            self.beam, self.rf, self.N_SLICES
        )
        self.profile_sparse = build_sparse_profile(
            self.beam, self.rf, self.N_SLICES, self.injected_batches
        )
        self.profile_bucket = build_sparse_bucket_profile(
            self.beam, self.rf, self.N_SLICES, self.injected_batches
        )
        self.profiles_sparse = {
            "SparseBatch": self.profile_sparse,
            "SparseBucket": self.profile_bucket,
        }

        self.RFFB = LHCCavityLoopCommissioning()  # default, closed loop
        self._make_loops()

        self.rtol = 1e-10
        self.atol = 1e-10
        if show_plot:
            fig, ax = plt.subplots(nrows=3, figsize=(10, 5))
            for axis, xlim in zip(
                ax, (None, [24.95, 24.954], [27.445, 27.449])
            ):
                axis.plot(
                    self.profile_std.bin_centers * 1e6,
                    self.profile_std.n_macroparticles,
                    label="Standard profile",
                )
                for name, profile_sparse in self.profiles_sparse.items():
                    for profile_ind in profile_sparse.profiles_list:
                        axis.plot(
                            profile_ind.bin_centers * 1e6,
                            profile_ind.n_macroparticles,
                            ls="--",
                            label=name,
                        )
                axis.set_xlabel(r"$\Delta t$ [$\mu$s]")
                axis.set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
                axis.legend()
                axis.set_yticks([])
                if xlim is not None:
                    axis.set(xlim=xlim)
            plt.show()

    # Helpers -----------------------------------------------------------------

    def _make_loops(self):
        """Fresh cavity loops on the current profiles: CL_standard, and one
        per sparse profile in loops_sparse."""
        self.CL_standard = build_cavity_loop(
            self.rf, self.profile_std, self.RFFB
        )
        self.CL_sparse = build_cavity_loop(
            self.rf, self.profile_sparse, self.RFFB
        )
        self.CL_bucket = build_cavity_loop(
            self.rf, self.profile_bucket, self.RFFB
        )
        self.loops_sparse = {
            "SparseBatch": self.CL_sparse,
            "SparseBucket": self.CL_bucket,
        }

    def _track_loops(self):
        self.CL_standard.track()
        for CL_sparse in self.loops_sparse.values():
            CL_sparse.track()

    def _disable_fine_grid(self):
        self.CL_standard.disable_fine_grid = True
        for CL_sparse in self.loops_sparse.values():
            CL_sparse.disable_fine_grid = True

    def _windows(self, CL_sparse):
        """Yield (p, profile, index) for each window of the sparse profile
        of a cavity loop: the profile number, its Profile object and the
        index of its first bin in the standard profile."""
        for p, profile in enumerate(CL_sparse.profile.profiles_list):
            index = np.argmin(
                np.abs(
                    self.CL_standard.profile.bin_centers
                    - profile.bin_centers[0]
                )
            )
            yield p, profile, index

    def _assert_bin_centers_match(self, CL_sparse, name):
        for p, profile, index in self._windows(CL_sparse):
            np.testing.assert_allclose(
                self.CL_standard.profile.bin_centers[index],
                CL_sparse.profile.bin_centers[p * profile.n_slices],
                rtol=self.rtol,
                atol=self.atol,
                err_msg="bin_centers differ between standard Profile and "
                f"{name} for the same beam, profile number {p}",
            )

    def _assert_fine_grid_match(self, array, CL_sparse, name, offset=0):
        """Compare a fine-grid array of CL_sparse with the one of
        CL_standard window by window. offset=1 skips the first element of
        the arrays of length n_slices + 1 (I_GEN_FINE, V_ANT_FINE), which
        is common to all windows."""
        array_std = getattr(self.CL_standard, array)
        array_sparse = getattr(CL_sparse, array)
        for p, profile, index in self._windows(CL_sparse):
            n = profile.n_slices
            np.testing.assert_allclose(
                array_std[index + offset : index + n + offset],
                array_sparse[p * n + offset : (p + 1) * n + offset],
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"{array} differs between standard Profile and "
                f"{name} for the same beam, profile number {p}",
            )

    @staticmethod
    def _coarse_state_at(CL, array, t):
        """Interpolate a coarse-grid array of a cavity loop (previous and
        current turn) at time t, as cavity_response_fine_matrix does."""
        return interp1d(
            np.concatenate(
                (CL.rf_centers - CL.T_s * CL.n_coarse, CL.rf_centers)
            ),
            array,
            fill_value="extrapolate",
        )(t)

    # Checks, shared by the single-turn and multi-turn tests ------------------

    def _check_rf_beam_current(self):
        """The beam current seen on the coarse and fine grids must not
        depend on which profile representation was used."""
        self.CL_standard.rf_beam_current(lpf=self.CL_standard.lpf)
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.rf_beam_current(lpf=CL_sparse.lpf)
                np.testing.assert_allclose(
                    self.CL_standard.I_BEAM_COARSE[
                        -self.CL_standard.n_coarse :
                    ],
                    CL_sparse.I_BEAM_COARSE[-CL_sparse.n_coarse :],
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="I_BEAM_COARSE differs between standard Profile "
                    f"and {name} for the same beam.",
                )
                self._assert_bin_centers_match(CL_sparse, name)
                self._assert_fine_grid_match("I_BEAM_FINE", CL_sparse, name)

    def _check_coarse_antenna_voltage_after_one_track(self):
        self.CL_standard.track()
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track()
                for array in ("V_ANT_COARSE", "I_GEN_COARSE"):
                    np.testing.assert_allclose(
                        getattr(self.CL_standard, array),
                        getattr(CL_sparse, array),
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg=f"{array} differs between standard Profile "
                        f"and {name} for the same beam.",
                    )

    def _check_fine_grid_track_one_turn(self):
        """The track_one_turn() function should provide the same output for
        a standard and a sparse profile."""
        self.CL_standard.track_one_turn()
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track_one_turn()
                self._assert_bin_centers_match(CL_sparse, name)
                for array in ("I_BEAM_FINE", "I_GEN_FINE", "V_ANT_FINE"):
                    self._assert_fine_grid_match(array, CL_sparse, name)

    def _check_fine_grid_generator_current_fine_grid_disabled(self):
        """Interpolation of the generator fine grid current should provide
        the same output for the standard profile and the sparse profiles.
        Since the first element of I_GEN_FINE and V_ANT_FINE is common, the
        test only covers the n_slices per profile (offset=1)."""
        self._disable_fine_grid()
        self.CL_standard.track()
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track()
                self._assert_bin_centers_match(CL_sparse, name)
                self._assert_fine_grid_match("I_BEAM_FINE", CL_sparse, name)
                self._assert_fine_grid_match(
                    "I_GEN_FINE", CL_sparse, name, offset=1
                )
                self._assert_fine_grid_match(
                    "V_ANT_FINE", CL_sparse, name, offset=1
                )

    def _check_fine_grid_cavity_response_inputs(self):
        """The input to the cavity_response_fine_matrix function should be
        identical for the standard profile and the sparse profiles."""
        self.CL_standard.track()
        t_at_init = (
            self.CL_standard.profile.bin_centers[0]
            - self.CL_standard.profile.bin_size
        )
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track()
                np.testing.assert_equal(
                    self.CL_standard.samples_fine, CL_sparse.samples_fine
                )
                t_at_init_sparse = (
                    CL_sparse.profile.bin_centers[0]
                    - CL_sparse.profile.bin_size
                )
                np.testing.assert_equal(t_at_init, t_at_init_sparse)
                np.testing.assert_array_equal(
                    self.CL_standard.rf_centers, CL_sparse.rf_centers
                )
                np.testing.assert_allclose(
                    self.CL_standard.V_ANT_COARSE,
                    CL_sparse.V_ANT_COARSE,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="V_ANT_COARSE differs between standard Profile "
                    f"and {name} for the same beam.",
                )
                for array in ("V_ANT_COARSE", "I_GEN_COARSE"):
                    np.testing.assert_allclose(
                        self._coarse_state_at(
                            self.CL_standard,
                            getattr(self.CL_standard, array),
                            t_at_init,
                        ),
                        self._coarse_state_at(
                            CL_sparse,
                            getattr(CL_sparse, array),
                            t_at_init_sparse,
                        ),
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg=f"initial value from {array} differs between "
                        f"standard Profile and {name} for the same beam.",
                    )
                np.testing.assert_allclose(
                    self.CL_standard.I_GEN_FINE[0],
                    CL_sparse.I_GEN_FINE[0],
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="I_GEN_FINE first element differs between "
                    f"standard Profile and {name} for the same beam",
                )
                self._assert_bin_centers_match(CL_sparse, name)
                self._assert_fine_grid_match("I_BEAM_FINE", CL_sparse, name)
                self._assert_fine_grid_match(
                    "I_GEN_FINE", CL_sparse, name, offset=1
                )

    def _check_fine_grid_antenna_voltage(self):
        """The fine-grid antenna voltage computed by
        `cavity_response_fine_matrix` should be numerically the same for
        a standard Profile and an equivalent sparse profile. Since the
        first element of V_ANT_FINE is common, the test only covers the
        n_slices per profile (offset=1)."""
        self.CL_standard.track()
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track()
                np.testing.assert_allclose(
                    self.CL_standard.V_ANT_FINE[0],
                    CL_sparse.V_ANT_FINE[0],
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="V_ANT_FINE first element differs between "
                    f"standard Profile and {name} for the same beam",
                )
                self._assert_bin_centers_match(CL_sparse, name)
                self._assert_fine_grid_match(
                    "V_ANT_FINE", CL_sparse, name, offset=1
                )

    def _check_generator_power(self):
        self.CL_standard.track()
        for name, CL_sparse in self.loops_sparse.items():
            with self.subTest(profile=name):
                CL_sparse.track()
                np.testing.assert_allclose(
                    self.CL_standard.generator_power(),
                    CL_sparse.generator_power(),
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg="generator_power differs between standard "
                    f"Profile and {name} for the same beam.",
                )

    # Tests -------------------------------------------------------------------

    def test_rf_beam_current_consistent(self):
        self._check_rf_beam_current()

    def test_coarse_antenna_voltage_consistent_after_one_track(self):
        self._check_coarse_antenna_voltage_after_one_track()

    def test_fine_grid_antenna_voltage_consistent_track_one_turn(self):
        self._check_fine_grid_track_one_turn()

    def test_fine_grid_generator_current_consistent_fine_grid_disabled(self):
        self._check_fine_grid_generator_current_fine_grid_disabled()

    def test_fine_grid_cavity_response_inputs(self):
        self._check_fine_grid_cavity_response_inputs()

    def test_fine_grid_antenna_voltage_consistent(self):
        self._check_fine_grid_antenna_voltage()

    def test_generator_power_consistent(self):
        self._check_generator_power()

    # Multi-turn injection: same checks after every injection -----------------

    def _inject_all(self, check):
        """Inject the remaining batches one at a time; after each injection
        re-track the standard profile and run check() on the cavity loops,
        which follow their profiles."""
        while self.injected_batches < number_of_batches:
            self.injected_batches = inject_batch(
                self.beam,
                self.ring,
                self.rf,
                self.profiles_sparse.values(),
                self.injected_batches,
            )
            self.profile_std.track()
            with self.subTest(injected_batches=self.injected_batches):
                check()

    def test_muliturn_injection_rf_beam_current(self):
        self._track_loops()
        self._inject_all(self._check_rf_beam_current)

    def test_muliturn_injection_coarse_antenna_voltage_consistent_after_one_track(
        self,
    ):
        self._track_loops()
        self._inject_all(self._check_coarse_antenna_voltage_after_one_track)

    def test_muliturn_injection_fine_grid_generator_current_consistent_fine_grid_disabled(
        self,
    ):
        self._disable_fine_grid()
        self._track_loops()
        self._inject_all(
            self._check_fine_grid_generator_current_fine_grid_disabled
        )

    def test_muliturn_fine_grid_cavity_response_inputs(self):
        self._track_loops()
        self._inject_all(self._check_fine_grid_cavity_response_inputs)

    def test_muliturn_fine_grid_antenna_voltage_consistent(self):
        self._track_loops()
        self._inject_all(self._check_fine_grid_antenna_voltage)

    def test_multiturn_injection_generator_power(self):
        self._track_loops()
        self._inject_all(self._check_generator_power)


if __name__ == "__main__":
    unittest.main()
