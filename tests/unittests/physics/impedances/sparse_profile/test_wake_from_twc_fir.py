import unittest

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
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.impedances.sources import TravelingWaveCavity

pytestmark = pytest.mark.backend_mutation

# modes with an implementation of `wake_from_twc_fir`
SPECIAL_MODES = ["python", "cpp"]


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
    voltage = backend.zeros(n, dtype=backend.float)
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
                with backend.temporary_specials_mode(mode):
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
                with backend.temporary_specials_mode(mode):
                    voltage = run_kernel(
                        hist_y,
                        np.array([1e6]),
                        np.array([n_wake * bin_dt]),
                        np.array([2e8]),
                        bin_dt,
                    )
                peak = np.max(np.abs(voltage))
                # wake present within the filling time ...
                assert np.max(np.abs(voltage[3 : 3 + n_wake])) > 0.1 * peak
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
                with backend.temporary_specials_mode(mode):
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

    def test_cpp_matches_python_reference(self):
        """cpp and python implementations must agree to float64 accuracy."""
        n = 2048
        bin_dt = 5e-11
        _, hist_y = get_test_profile(n, bin_dt)
        r_shunt = np.array([1.38e6, 0.88e6, 2.2e5])
        a_tilde = np.array([120.5, 333.25, 47.0]) * bin_dt
        frequency_r = np.array([2.0e8, 5.3e8, 1.1e9])

        with backend.temporary_specials_mode("python"):
            v_python = run_kernel(
                hist_y, r_shunt, a_tilde, frequency_r, bin_dt
            )
        with backend.temporary_specials_mode("cpp"):
            v_cpp = run_kernel(hist_y, r_shunt, a_tilde, frequency_r, bin_dt)
        np.testing.assert_allclose(
            v_cpp,
            v_python,
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(v_python)),
        )


class TestTwcInMultiPoleSparseSolve(unittest.TestCase):
    """TWC sources handled by the sparse solver via the FIR kernel."""

    @staticmethod
    def _run_one_turn(solver, sources=None):
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
        # wake support ~40 bins, well inside the profile window
        twc = TravelingWaveCavity(
            R_S=[1.38e6],
            frequency_R=[200.222e6],
            a_factor=[2 * np.pi * 40.5 * t_rf / 256],
        )
        wakefield = WakeField(
            sources=(twc,) if sources is None else sources,
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

        with pytest.raises(TypeError, match="MultiPoleSparseSolve"):
            self._run_one_turn(
                MultiPoleSparseSolve(),
                sources=(InductiveImpedance(28.0),),
            )


if __name__ == "__main__":
    unittest.main()
