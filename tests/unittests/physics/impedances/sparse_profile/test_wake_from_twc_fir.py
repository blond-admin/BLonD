import unittest
from contextlib import contextmanager

import numpy as np
import pytest
from scipy.signal import fftconvolve

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Cupy64Bit, Numpy64Bit
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.impedances.sources import TravelingWaveCavity

pytestmark = pytest.mark.backend_mutation

try:
    import cupy  # NOQA: F401

    cupy_available = True
except ModuleNotFoundError:
    cupy_available = False

# modes with an implementation of `wake_from_twc_fir`
SPECIAL_MODES = ["python", "numba", "cpp"]
if cupy_available:
    SPECIAL_MODES.append("cuda")


@contextmanager
def specials_mode(mode):
    """Switch to `mode` and to its backend, cupy for ``"cuda"``.

    Restores the backend active before, which need not be numpy.
    """
    original_backend = type(backend)
    backend.change_backend(Cupy64Bit if mode == "cuda" else Numpy64Bit)
    try:
        with backend.temporary_specials_mode(mode):
            yield
    finally:
        backend.change_backend(original_backend)


def get_test_profile(n: int, bin_dt: float):
    rng = np.random.default_rng(42)
    samples = np.concatenate(
        (
            rng.standard_normal(int(1e5)) * 20 * bin_dt,
            rng.standard_normal(int(1e5)) * 20 * bin_dt + n * bin_dt / 3,
        )
    )
    hist_y, edges = np.histogram(
        samples, bins=n, range=(-(n * bin_dt) / 4, 3 * (n * bin_dt) / 4)
    )
    centers = edges[:-1] + bin_dt / 2
    return centers, np.asarray(hist_y, dtype=float)


def run_kernel(hist_y, r_shunt, a_tilde, frequency_r, bin_dt, grid_index=None):
    n = len(hist_y)
    if grid_index is None:
        grid_index = np.arange(n, dtype=np.int32)
    # NaN, so that a kernel that adds to `voltage` instead of overwriting
    # it is caught
    voltage = backend.array(np.full(n, np.nan), dtype=backend.float)
    backend.specials.wake_from_twc_fir(
        profile=backend.array(hist_y, dtype=backend.float),
        grid_index=backend.array(grid_index, dtype=np.int32),
        r_shunt=backend.array(r_shunt, dtype=backend.float),
        a_tilde=backend.array(a_tilde, dtype=backend.float),
        omega_r=backend.array(2 * np.pi * frequency_r, dtype=backend.float),
        bin_dt=bin_dt,
        factor=1.0,
        voltage=voltage,
        voltage_threaded=backend.zeros(
            (backend.specials.get_max_threads(), n), dtype=backend.float
        ),
    )
    return copy_to_cpu(voltage)


def _proton_velocity():
    """Velocity of the 25.92 GeV/c protons of the test ring, in [m/s]."""
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=25.92e9, in_unit="momentum"
    )
    return 6911.56 / magnetic_cycle.get_t_rev_init(
        6911.56, particle_type=proton
    )


class TestWakeFromTwcFir(unittest.TestCase):
    def test_matches_wake_convolution(self):
        """Kernel must match fftconvolve(profile, wake_calc) for two modes."""
        n = 4096
        bin_dt = 1e-10
        centers, hist_y = get_test_profile(n, bin_dt)

        # wake supports of ~150 and ~400 bins: termination happens in-window
        a_tilde = np.array([150.5 * bin_dt, 400.25 * bin_dt])
        frequency_r = np.array([2.0e8, 5.3e8])
        r_shunt = np.array([1.38e6, 0.88e6])

        twc = TravelingWaveCavity(
            R_S=r_shunt,
            frequency_R=frequency_r,
            a_factor=2 * np.pi * a_tilde,
        )
        wake = copy_to_cpu(twc.wake_calc(backend.array(centers - centers[0])))
        ref = fftconvolve(hist_y, wake)[:n]
        atol = 0.01 * np.max(np.abs(ref))

        for mode in SPECIAL_MODES:
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    voltage = run_kernel(
                        hist_y, r_shunt, a_tilde, frequency_r, bin_dt
                    )
                np.testing.assert_allclose(
                    voltage,
                    ref,
                    rtol=0.05,
                    atol=atol,
                    err_msg=f"[{mode}] must match fftconvolve reference",
                )

    def test_wake_terminates_after_a_tilde(self):
        """A delta profile's wake must be exactly zero after the filling time."""
        n = 512
        bin_dt = 1e-10
        hist_y = np.zeros(n, dtype=float)
        hist_y[3] = 1.0
        n_wake = 100  # bins

        for mode in SPECIAL_MODES:
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    voltage = run_kernel(
                        hist_y,
                        np.array([1e6]),
                        np.array([n_wake * bin_dt]),
                        np.array([2e8]),
                        bin_dt,
                    )
                peak = np.max(np.abs(voltage))
                # wake present within the filling time ...
                self.assertGreater(
                    np.max(np.abs(voltage[3 : 3 + n_wake])), 0.1 * peak
                )
                # ... numerically dead beyond it (allow the ceil-rounded
                # edge bin)
                np.testing.assert_allclose(
                    voltage[3 + n_wake + 2 :], 0.0, atol=1e-9 * peak
                )

    def test_gapped_grid_matches_dense_grid(self):
        """Gapped-lattice processing must equal a dense grid with empty gaps.

        The profile bins of a gapped multi-bunch grid sit on a common
        equidistant lattice (gaps are integer multiples of the bin width).
        Convolving only the occupied bins while jumping the gaps must give
        the same voltage as running the dense lattice with zero charge in
        the gaps.
        """
        bin_dt = 1e-10
        bins_per_bucket = 16
        # occupied buckets on a 21-bucket lattice: gaps of 4, 1 and 12
        # buckets between them
        bucket_starts = np.array([0, 5, 7, 20])
        rng = np.random.default_rng(7)
        chunks = rng.random((len(bucket_starts), bins_per_bucket)) + 0.1
        hist_gapped = chunks.ravel()
        grid_index = np.concatenate(
            [
                np.arange(s, s + bins_per_bucket, dtype=np.int32)
                for s in bucket_starts * bins_per_bucket
            ]
        )
        n_dense = (bucket_starts[-1] + 1) * bins_per_bucket
        hist_dense = np.zeros(n_dense)
        hist_dense[grid_index] = hist_gapped

        # wake supports of ~10.3 and ~3.4 buckets: one bridges the small
        # gaps and dies before the last bucket, one ends inside a gap
        a_tilde = np.array([10.3, 3.4]) * bins_per_bucket * bin_dt
        frequency_r = np.array([2.0e8, 5.3e8])
        r_shunt = np.array([1.38e6, 0.88e6])

        for mode in SPECIAL_MODES:
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    v_dense = run_kernel(
                        hist_dense, r_shunt, a_tilde, frequency_r, bin_dt
                    )
                    v_gapped = run_kernel(
                        hist_gapped,
                        r_shunt,
                        a_tilde,
                        frequency_r,
                        bin_dt,
                        grid_index=grid_index,
                    )
                np.testing.assert_allclose(
                    v_gapped,
                    v_dense[grid_index],
                    rtol=1e-9,
                    atol=1e-9 * np.max(np.abs(v_dense)),
                    err_msg=f"[{mode}] gapped run must equal dense run",
                )

    def test_backends_match_python_reference(self):
        """Every backend must agree with the python reference."""
        n = 2048
        bin_dt = 5e-11
        _, hist_y = get_test_profile(n, bin_dt)
        r_shunt = np.array([1.38e6, 0.88e6, 2.2e5])
        a_tilde = np.array([120.5, 333.25, 47.0]) * bin_dt
        frequency_r = np.array([2.0e8, 5.3e8, 1.1e9])

        with specials_mode("python"):
            v_python = run_kernel(
                hist_y, r_shunt, a_tilde, frequency_r, bin_dt
            )
        for mode in SPECIAL_MODES[1:]:
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    voltage = run_kernel(
                        hist_y, r_shunt, a_tilde, frequency_r, bin_dt
                    )
                np.testing.assert_allclose(
                    voltage,
                    v_python,
                    rtol=1e-8,
                    atol=1e-8 * np.max(np.abs(v_python)),
                )

    def test_many_modes_match_cpp(self):
        """Every mode must add its wake once, also with many modes.

        On CUDA each mode is one thread that adds into the shared
        `voltage`; with more modes than the device runs concurrently,
        threads start after others have finished, which exposes any
        write to shared memory that is not an atomic add (like the race on
        ``t_start`` once fixed in ``wake_from_pole_residue``).
        """
        n = 64
        bin_dt = 1e-10
        n_modes = 10000
        rng = np.random.default_rng(3)
        _, hist_y = get_test_profile(n, bin_dt)
        r_shunt = rng.uniform(1e3, 1e4, n_modes)
        a_tilde = rng.uniform(5, 50, n_modes) * bin_dt
        frequency_r = rng.uniform(1e8, 1e9, n_modes)

        with specials_mode("cpp"):
            v_cpp = run_kernel(hist_y, r_shunt, a_tilde, frequency_r, bin_dt)
        for mode in SPECIAL_MODES:
            if mode in ("python", "cpp"):
                continue  # python: too slow for this many modes
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    for _ in range(2):  # reusing the device buffers
                        voltage = run_kernel(
                            hist_y, r_shunt, a_tilde, frequency_r, bin_dt
                        )
                        np.testing.assert_allclose(
                            voltage,
                            v_cpp,
                            rtol=1e-8,
                            atol=1e-8 * np.max(np.abs(v_cpp)),
                        )


def _long_gapped_case():
    """A multi-bunch lattice long enough for a backend to split it in time.

    64 bunches of 64 bins in every third bucket, and wake supports that
    end inside a bunch, inside a gap, and bridge several bunches.
    """
    bin_dt = 1e-10
    bins_per_bucket = 64
    bucket_starts = np.arange(0, 3 * 64, 3) * bins_per_bucket
    grid_index = (
        (bucket_starts[:, None] + np.arange(bins_per_bucket))
        .ravel()
        .astype(np.int32)
    )
    rng = np.random.default_rng(11)
    hist_y = rng.random(len(grid_index)) + 0.1
    a_tilde = np.array([40.5, 150.0, 700.25]) * bin_dt
    frequency_r = np.array([2.0e8, 5.3e8, 1.1e9])
    r_shunt = np.array([1.38e6, 0.88e6, 2.2e5])
    return hist_y, grid_index, r_shunt, a_tilde, frequency_r, bin_dt


class TestWakeFromTwcFirLongGappedProfile(unittest.TestCase):
    def test_backends_match_python_reference(self):
        """Every backend agrees with the python reference on a long lattice.

        Long enough that a backend may split the bins into chunks run in
        parallel; the stitching must reproduce the serial recursion.
        """
        hist_y, grid_index, r_shunt, a_tilde, frequency_r, bin_dt = (
            _long_gapped_case()
        )
        with specials_mode("python"):
            v_python = run_kernel(
                hist_y, r_shunt, a_tilde, frequency_r, bin_dt, grid_index
            )
        for mode in SPECIAL_MODES[1:]:
            with self.subTest(mode=mode):
                with specials_mode(mode):
                    voltage = run_kernel(
                        hist_y,
                        r_shunt,
                        a_tilde,
                        frequency_r,
                        bin_dt,
                        grid_index,
                    )
                np.testing.assert_allclose(
                    voltage,
                    v_python,
                    rtol=1e-8,
                    atol=1e-8 * np.max(np.abs(v_python)),
                )


def _sps_scale_case():
    """An SPS-like lattice: 288 bunches, sites up to ~2e5, long supports.

    Four batches of 72 bunches of 128 bins, every fifth bucket, with
    eight empty buckets between batches; the supports bridge many
    bunches, and the profile spans many chunks of the CUDA scan.
    """
    bin_dt = 3.9e-11
    bins_per_bucket = 128
    buckets = np.concatenate(
        [batch * (72 * 5 + 8) + 5 * np.arange(72) for batch in range(4)]
    )
    grid_index = (
        (buckets[:, None] * bins_per_bucket + np.arange(bins_per_bucket))
        .ravel()
        .astype(np.int32)
    )
    rng = np.random.default_rng(12)
    hist_y = rng.poisson(50, len(grid_index)).astype(float)
    a_tilde = np.array([567e-9, 401e-9, 61e-9])
    frequency_r = np.array([200.222e6, 200.222e6, 799.2e6])
    r_shunt = np.array([2.71e4, 3.48e4, 1.9e5])
    return hist_y, grid_index, r_shunt, a_tilde, frequency_r, bin_dt


@pytest.mark.cupy
@unittest.skipUnless(cupy_available, "requires cupy")
class TestScannedTwcKernel(unittest.TestCase):
    """CUDA: the wake as window sums of prefix sums over the bins."""

    def test_sps_scale_matches_cpp(self):
        """Many scan chunks, large lattice sites: no loss of precision."""
        from blond.core.backends.cuda.callables import _TWC_SCAN_CHUNK

        hist_y, grid_index, r_shunt, a_tilde, frequency_r, bin_dt = (
            _sps_scale_case()
        )
        self.assertGreater(len(hist_y), 30 * _TWC_SCAN_CHUNK)
        results = {}
        for mode in ("cpp", "cuda"):
            with specials_mode(mode):
                results[mode] = run_kernel(
                    hist_y, r_shunt, a_tilde, frequency_r, bin_dt, grid_index
                )
        np.testing.assert_allclose(
            results["cuda"],
            results["cpp"],
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(results["cpp"])),
        )


class TestTwcInMultiPoleSparseSolve(unittest.TestCase):
    """TWC sources handled by the sparse solver via the FIR kernel."""

    @staticmethod
    def _twc():
        """A TWC whose wake support is ~40 of the 256 profile bins."""
        t_rf = 6911.56 / 4620 / _proton_velocity()
        return TravelingWaveCavity(
            R_S=[1.38e6],
            frequency_R=[200.222e6],
            a_factor=[2 * np.pi * 40.5 * t_rf / 256],
        )

    @classmethod
    def _run_one_turn(cls, solver, sources=None):
        circumference = 6911.56
        harmonic = 4620
        ring = Ring(circumference=circumference)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton, value=25.92e9, in_unit="momentum"
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=22.82177322938192
            ),
            orbit_length=circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=harmonic, voltage=0.9e6, phi_rf=0.0
        )
        t_rev = magnetic_cycle.get_t_rev_init(
            circumference, particle_type=proton
        )
        t_rf = t_rev / harmonic
        profile = StaticProfile(cut_left=0.0, cut_right=t_rf, n_bins=256)
        wakefield = WakeField(
            sources=(cls._twc(),) if sources is None else sources,
            solver=solver,
            profile=profile,
        )
        ring.add_elements((wakefield, drift, rf_station))
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        bunch = Beam(intensity=1e10, particle_type=proton)
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=t_rf / 10, seed=1, n_macroparticles=int(1e4)
            ),
            beam=bunch,
        )
        sim.run_simulation(beams=(bunch,), n_turns=1)
        return copy_to_cpu(wakefield.induced_voltage)

    def test_matches_time_domain_solver(self):
        v_sparse = self._run_one_turn(MultiPoleSparseSolve())
        v_reference = self._run_one_turn(
            TimeDomainFftSolver(allow_next_fast_len=False)
        )
        np.testing.assert_allclose(
            v_sparse,
            v_reference,
            rtol=0.05,
            atol=0.01 * np.max(np.abs(v_reference)),
            err_msg="sparse solver TWC FIR must match TimeDomainFftSolver",
        )

    @unittest.skipUnless(cupy_available, "requires cupy")
    def test_cuda_matches_cpu(self):
        """The whole sparse solver with TWC must run on the GPU."""
        from unittest.mock import patch

        from blond.core.backends.cuda.callables import CudaSpecials

        v_cpu = self._run_one_turn(MultiPoleSparseSolve())
        with (
            specials_mode("cuda"),
            patch.object(
                CudaSpecials,
                "wake_from_twc_fir",
                wraps=CudaSpecials.wake_from_twc_fir,
            ) as kernel,
        ):
            v_cuda = self._run_one_turn(MultiPoleSparseSolve())
        kernel.assert_called()
        np.testing.assert_allclose(
            v_cuda, v_cpu, rtol=1e-8, atol=1e-8 * np.max(np.abs(v_cpu))
        )

    @staticmethod
    def _run_one_turn_multiprofile(filled_buckets):
        """One-turn sparse-solver run on an ``EquidistantMultiProfile``.

        A single bunch sits in bucket 0; the TWC wake support spans ~4.6
        buckets, so its wake reaches every bucket in `filled_buckets`.
        """
        from blond.physics.profiles_sparse import EquidistantMultiProfile

        circumference = 6911.56
        harmonic = 4620
        ring = Ring(circumference=circumference)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton, value=25.92e9, in_unit="momentum"
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=22.82177322938192
            ),
            orbit_length=circumference,
        )
        rf_station = SingleHarmonicRFStation(
            harmonic=harmonic, voltage=0.9e6, phi_rf=0.0
        )
        t_rev = magnetic_cycle.get_t_rev_init(
            circumference, particle_type=proton
        )
        t_rf = t_rev / harmonic
        mask = np.zeros(harmonic, dtype=bool)
        mask[list(filled_buckets)] = True
        profile = EquidistantMultiProfile(
            filling_pattern=mask, bins_per_profile=64
        )
        twc = TravelingWaveCavity(
            R_S=[1.38e6],
            frequency_R=[200.222e6],
            a_factor=[2 * np.pi * 4.6 * t_rf],
        )
        wakefield = WakeField(
            sources=(twc,), solver=MultiPoleSparseSolve(), profile=profile
        )
        ring.add_elements((wakefield, drift, rf_station))
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        bunch = Beam(intensity=1e10, particle_type=proton)
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=t_rf / 10, seed=1, n_macroparticles=int(1e4)
            ),
            beam=bunch,
        )
        sim.run_simulation(beams=(bunch,), n_turns=1)
        return copy_to_cpu(wakefield.induced_voltage)

    def test_gapped_multiprofile_matches_dense(self):
        """Wake reaching bucket 4 across empty buckets 1-3 must match the
        contiguous (gap-free) reference run of the same beam."""
        n = 64  # bins per bucket
        v_gapped = self._run_one_turn_multiprofile({0, 4})
        v_dense = self._run_one_turn_multiprofile({0, 1, 2, 3, 4})
        # bucket 0 occupies bins 0:n in both runs; bucket 4 is bins n:2n
        # of the gapped run and bins 4n:5n of the dense run
        atol = 1e-9 * np.max(np.abs(v_dense))
        np.testing.assert_allclose(
            v_gapped[:n], v_dense[:n], atol=atol, err_msg="bucket 0"
        )
        np.testing.assert_allclose(
            v_gapped[n : 2 * n],
            v_dense[4 * n : 5 * n],
            atol=atol,
            err_msg="bucket 4 (across the gap)",
        )

    def test_unsupported_source_raises(self):
        from blond.physics.impedances.sources import InductiveImpedance

        with self.assertRaisesRegex(TypeError, "SupportsTWCFIRModel"):
            self._run_one_turn(
                MultiPoleSparseSolve(),
                sources=(InductiveImpedance(28.0),),
            )

    def test_warns_that_the_wake_is_point_sampled(self):
        """The FIR samples the TWC wake instead of bin-averaging it.

        Every other time-domain solver applies the bin-averaged wake, so
        the sparse solver differs from them at the onset and at the end of
        the wake by an amount that vanishes for bins much shorter than the
        filling time. Until the FIR is bin-averaged, the user is told.
        """
        with self.assertWarnsRegex(UserWarning, "point-sampled"):
            self._run_one_turn(MultiPoleSparseSolve())

    def test_twc_adds_to_poles_and_direct_term(self):
        """TWC, poles and the direct term of a vector fit superpose."""
        from blond.physics.impedances.base import (
            SupportsVectorFittedModel,
            VectorFit,
            WakeFieldSource,
        )
        from blond.physics.impedances.sources import Resonators

        resonators = Resonators(1e5, 1.2e9, 5.0)
        fit = resonators.get_vectorfit()

        class _Fitted(WakeFieldSource, SupportsVectorFittedModel):
            def __init__(self, direct_term):
                super().__init__(is_dynamic=False)
                self.direct_term = direct_term

            def get_vectorfit(self):
                return VectorFit(
                    fit.poles,
                    fit.residues,
                    fit.counterrotation_signs,
                    direct_term=self.direct_term,
                )

        twc = self._run_one_turn(MultiPoleSparseSolve())
        fitted = self._run_one_turn(
            MultiPoleSparseSolve(), sources=(_Fitted(2.5e3),)
        )
        both = self._run_one_turn(
            MultiPoleSparseSolve(),
            sources=(self._twc(), _Fitted(2.5e3)),
        )
        np.testing.assert_allclose(
            both, twc + fitted, rtol=0, atol=1e-9 * np.max(np.abs(both))
        )


class TestTwcAcrossCalls(unittest.TestCase):
    """`wake_from_twc_fir` keeps no state from one call to the next."""

    BIN_DT = 1e-10

    def _solver_and_beam(self, a_tilde):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from scipy.constants import e

        solver = MultiPoleSparseSolve()
        parent = Mock(WakeField)
        parent.sources = (
            TravelingWaveCavity(
                R_S=[1e6], frequency_R=[2e8], a_factor=[2 * np.pi * a_tilde]
            ),
        )
        profile = Mock(spec=StaticProfile)
        profile.hist_x = backend.array(
            np.arange(16) * self.BIN_DT, dtype=backend.float
        )
        profile.hist_y = backend.ones(16, dtype=backend.float)
        profile.hist_step = self.BIN_DT
        profile.hist_y_to_density_factor = 1.0
        parent.profile = profile
        solver._parent_wakefield = parent
        solver._profile = profile
        beam = SimpleNamespace(
            reference=SimpleNamespace(time=0.0),
            particle_type=SimpleNamespace(charge=1.0),
            intensity=1.0 / e,
            is_counter_rotating=False,
        )
        return solver, beam

    def test_warns_when_the_previous_wake_reaches_this_call(self):
        solver, beam = self._solver_and_beam(a_tilde=50 * self.BIN_DT)
        with self.assertWarnsRegex(UserWarning, "point-sampled"):
            solver.calc_induced_voltage(beam=beam)
        beam.reference.time = 20 * self.BIN_DT  # 5 bins after the last
        with self.assertWarnsRegex(UserWarning, "previous call"):
            solver.calc_induced_voltage(beam=beam)

    def test_silent_when_the_previous_wake_has_ended(self):
        import warnings

        solver, beam = self._solver_and_beam(a_tilde=3 * self.BIN_DT)
        with self.assertWarnsRegex(UserWarning, "point-sampled"):
            solver.calc_induced_voltage(beam=beam)
        beam.reference.time = 20 * self.BIN_DT
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solver.calc_induced_voltage(beam=beam)


if __name__ == "__main__":
    unittest.main()
