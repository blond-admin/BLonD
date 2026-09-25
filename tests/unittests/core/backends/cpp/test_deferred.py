# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The deferred C++ specials must reproduce the eager C++ specials.

``DeferredCppSpecials`` queues per-particle kernels and executes the queue
chunk by chunk (all queued kernels on one cache-sized chunk, then the
next) once a result is needed. It shares the formulas with the eager
kernels, so the results must agree with ``CppSpecials`` to rounding.
"""

from __future__ import annotations

import gc
import inspect
import threading

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp.callables import reload_cpp_backend
from blond.testing.backend_testing import BLonDTestCase

N_BINS = 64
CUT_LEFT = -1e-9
CUT_RIGHT = 1e-9


def _make_beam(n_macroparticles: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dt = rng.normal(0, 3e-10, n_macroparticles)
    dE = rng.normal(0, 1e6, n_macroparticles)
    return dt, dE


def _assert_close_to_scale(actual, desired):
    scale = np.max(np.abs(desired))
    np.testing.assert_allclose(actual, desired, rtol=1e-13, atol=1e-13 * scale)


def _make_voltage():
    bin_centers = np.linspace(CUT_LEFT, CUT_RIGHT, N_BINS)
    voltage = 1e3 * np.sin(np.linspace(0, 3, N_BINS))
    return voltage, bin_centers


def _track_one_turn(specials, dt, dE, voltage, bin_centers, profile):
    """Every queueable kernel once, ending in the histogram (a flush).

    All parameters differ, so a packing order mistake changes results.
    """
    specials.kick_interpolated(
        dt=dt,
        dE=dE,
        voltage=voltage,
        bin_centers=bin_centers,
        charge=0.7,
        acceleration_kick=12.0,
    )
    specials.kick_single_harmonic(
        dt=dt,
        dE=dE,
        voltage=6e6,
        omega_rf=2.1e9,
        phi_rf=0.3,
        charge=-1.3,
        acceleration_kick=-4.0,
    )
    specials.kick_multi_harmonic(
        dt=dt,
        dE=dE,
        voltage=np.array([1e6, 2e5, 1e5]),
        omega_rf=np.array([2.1e9, 4.2e9, 8.4e9]),
        phi_rf=np.array([0.1, 0.2, 0.3]),
        charge=0.9,
        n_rf=3,
        acceleration_kick=1.5,
    )
    specials.drift_simple(
        dt=dt, dE=dE, T=8.9e-5, eta_0=3e-4, beta=0.99, energy=450e9
    )
    specials.drift_like_line_segment(
        dt=dt, dE=dE, T=8.9e-5, eta_0=3e-4, beta=0.99, energy=450e9
    )
    specials.histogram(
        array_read=dt, array_write=profile, start=CUT_LEFT, stop=CUT_RIGHT
    )


class TestDeferredCppSpecials(BLonDTestCase):
    def setUp(self):
        self.eager = reload_cpp_backend(np.float64)
        self.deferred = reload_cpp_backend(np.float64, deferred=True)
        self.chunk_size_org = self.deferred.get_chunk_size()

    def tearDown(self):
        self.deferred.flush()
        self.deferred.set_chunk_size(self.chunk_size_org)

    def _compare_turns(self, n_macroparticles, chunk_size, n_turns=3):
        self.deferred.set_chunk_size(chunk_size)
        voltage, bin_centers = _make_voltage()
        dt_eager, dE_eager = _make_beam(n_macroparticles)
        dt_deferred, dE_deferred = _make_beam(n_macroparticles)
        profile_eager = np.zeros(N_BINS)
        profile_deferred = np.zeros(N_BINS)
        for _ in range(n_turns):
            _track_one_turn(
                self.eager,
                dt_eager,
                dE_eager,
                voltage,
                bin_centers,
                profile_eager,
            )
            _track_one_turn(
                self.deferred,
                dt_deferred,
                dE_deferred,
                voltage,
                bin_centers,
                profile_deferred,
            )
            # the histogram flushed the queue, the profile is up to date
            self.assertEqual(self.deferred.n_pending(), 0)
            np.testing.assert_array_equal(profile_deferred, profile_eager)
        # A particle on a bin edge may round into either neighbouring bin
        # (vector part or scalar tail of a tile, which chunking moves);
        # the interpolation is continuous there, so the kicks agree to
        # rounding -- relative to the coordinates' scale, not to a
        # coordinate that happens to be near zero.
        _assert_close_to_scale(dt_deferred, dt_eager)
        _assert_close_to_scale(dE_deferred, dE_eager)

    def test_matches_eager(self):
        for n_macroparticles in (1, 7, 1000, 100_003):
            for chunk_size in (1, 64, 4096, 10**9):
                with self.subTest(
                    n_macroparticles=n_macroparticles, chunk_size=chunk_size
                ):
                    self._compare_turns(n_macroparticles, chunk_size)

    def test_kernels_are_queued_until_flush(self):
        dt, dE = _make_beam(100)
        dt_org = dt.copy()
        self.deferred.drift_simple(
            dt=dt, dE=dE, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        self.assertEqual(self.deferred.n_pending(), 1)
        np.testing.assert_array_equal(dt, dt_org)
        self.deferred.flush()
        self.assertEqual(self.deferred.n_pending(), 0)
        np.testing.assert_allclose(dt, dt_org + dE)

    def test_flush_of_empty_queue_is_noop(self):
        self.deferred.flush()
        self.assertEqual(self.deferred.n_pending(), 0)

    def test_unqueued_kernel_flushes_first(self):
        dt, dE = _make_beam(100)
        self.deferred.kick_single_harmonic(
            dt=dt,
            dE=dE,
            voltage=0.0,
            omega_rf=1.0,
            phi_rf=0.0,
            charge=1.0,
            acceleration_kick=5.0,
        )
        expected = float(np.sum(dE + 5.0))
        # `sum_1d_array` is not queued; it must see the kicked energies
        self.assertAlmostEqual(
            self.deferred.sum_1d_array(dE), expected, delta=1e-6
        )
        self.assertEqual(self.deferred.n_pending(), 0)

    def test_kernel_on_other_beam_flushes_first(self):
        dt_a, dE_a = _make_beam(100, seed=1)
        dt_b, dE_b = _make_beam(50, seed=2)
        dt_a_org, dt_b_org = dt_a.copy(), dt_b.copy()
        self.deferred.drift_simple(
            dt=dt_a, dE=dE_a, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        self.deferred.drift_simple(
            dt=dt_b, dE=dE_b, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        self.assertEqual(self.deferred.n_pending(), 1)
        np.testing.assert_allclose(dt_a, dt_a_org + dE_a)
        self.deferred.flush()
        np.testing.assert_allclose(dt_b, dt_b_org + dE_b)

    def test_histogram_of_other_array_runs_eagerly(self):
        dt, dE = _make_beam(1000)
        self.deferred.drift_simple(
            dt=dt, dE=dE, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        other = np.linspace(CUT_LEFT, CUT_RIGHT, 1000)
        profile = np.zeros(N_BINS)
        self.deferred.histogram(
            array_read=other,
            array_write=profile,
            start=CUT_LEFT,
            stop=CUT_RIGHT,
        )
        self.assertEqual(profile.sum(), 1000)
        # the pending drift was flushed first, as for any unqueued kernel
        self.assertEqual(self.deferred.n_pending(), 0)

    def test_histogram_of_sorted_beam_matches_numpy(self):
        # sorted dt (as MuSiC sorts it) puts runs of particles into one
        # bin, which the counting must survive exactly
        for n_macroparticles in (1, 5, 1000, 100_003):
            dt, dE = _make_beam(n_macroparticles)
            dt.sort()
            expected, _ = np.histogram(dt, N_BINS, (CUT_LEFT, CUT_RIGHT))
            for specials in (self.eager, self.deferred):
                with self.subTest(n=n_macroparticles, specials=specials):
                    profile = np.zeros(N_BINS)
                    specials.drift_simple(
                        dt=dt, dE=dE, T=0.0, eta_0=1.0, beta=1.0, energy=1.0
                    )
                    specials.histogram(dt, profile, CUT_LEFT, CUT_RIGHT)
                    np.testing.assert_array_equal(profile, expected)

    def test_histogram_of_dE_is_queued(self):
        dt, dE = _make_beam(1000)
        dE_scaled = dE * 1e-15  # fits the cut
        profile_eager = np.zeros(N_BINS)
        profile_deferred = np.zeros(N_BINS)
        self.eager.histogram(dE_scaled, profile_eager, CUT_LEFT, CUT_RIGHT)
        self.deferred.drift_simple(
            dt=dt, dE=dE_scaled, T=0.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        self.deferred.histogram(
            dE_scaled, profile_deferred, CUT_LEFT, CUT_RIGHT
        )
        np.testing.assert_array_equal(profile_deferred, profile_eager)

    def test_queue_keeps_arrays_alive(self):
        dt, dE = _make_beam(1000)
        dE_expected = dE.copy()
        voltage, bin_centers = _make_voltage()
        self.eager.kick_interpolated(
            dt=dt.copy(),
            dE=dE_expected,
            voltage=voltage.copy(),
            bin_centers=bin_centers.copy(),
            charge=1.0,
            acceleration_kick=0.0,
        )
        self.deferred.kick_interpolated(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bin_centers,
            charge=1.0,
            acceleration_kick=0.0,
        )
        del voltage, bin_centers
        gc.collect()
        _ = [np.full(N_BINS, np.nan) for _ in range(100)]  # reuse memory
        self.deferred.flush()
        np.testing.assert_allclose(dE, dE_expected, rtol=1e-13)

    def test_scalars_are_captured_at_enqueue(self):
        dt, dE = _make_beam(10)
        dE_org = dE.copy()
        n_rf_voltage = np.array([0.0])
        self.deferred.kick_multi_harmonic(
            dt=dt,
            dE=dE,
            voltage=n_rf_voltage,
            omega_rf=np.array([1.0]),
            phi_rf=np.array([0.0]),
            charge=1.0,
            n_rf=1,
            acceleration_kick=3.0,
        )
        self.deferred.flush()
        np.testing.assert_allclose(dE, dE_org + 3.0)

    def test_every_compiled_op_is_queued_by_python(self):
        # an op the library knows but Python does not queue is dead code
        for name in self.deferred.compiled_ops():
            with self.subTest(op=name):
                self.assertIn(name, vars(self.deferred))
                self.assertIsNot(
                    getattr(self.deferred, name), getattr(self.eager, name)
                )

    def test_compiled_parameter_names_are_specials_parameters(self):
        # Python packs by these names, so each must name an argument of
        # the `Specials` method or a value the wrapper derives from them.
        derived = {"n_bins", "reads_dE"}
        for name, parameters in self.deferred.compiled_ops().items():
            signature = inspect.signature(getattr(self.eager, name))
            for parameter in parameters:
                with self.subTest(op=name, parameter=parameter):
                    self.assertTrue(
                        parameter in signature.parameters
                        or parameter in derived
                    )

    def test_threads_have_their_own_queue(self):
        dt, dE = _make_beam(100)
        self.deferred.drift_simple(
            dt=dt, dE=dE, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        pending_elsewhere = []
        thread = threading.Thread(
            target=lambda: pending_elsewhere.append(self.deferred.n_pending())
        )
        thread.start()
        thread.join()
        self.assertEqual(pending_elsewhere, [0])
        self.assertEqual(self.deferred.n_pending(), 1)

    def test_concurrent_simulations_match_eager(self):
        n_threads = 4
        barrier = threading.Barrier(n_threads)
        voltage, bin_centers = _make_voltage()
        results = [None] * n_threads

        def simulate(thread_i):
            dt, dE = _make_beam(20_000, seed=thread_i)
            profile = np.zeros(N_BINS)
            barrier.wait()
            for _ in range(20):
                _track_one_turn(
                    self.deferred, dt, dE, voltage, bin_centers, profile
                )
            results[thread_i] = (dt, dE, profile)

        threads = [
            threading.Thread(target=simulate, args=(thread_i,))
            for thread_i in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for thread_i, (dt, dE, profile) in enumerate(results):
            dt_eager, dE_eager = _make_beam(20_000, seed=thread_i)
            profile_eager = np.zeros(N_BINS)
            for _ in range(20):
                _track_one_turn(
                    self.eager,
                    dt_eager,
                    dE_eager,
                    voltage,
                    bin_centers,
                    profile_eager,
                )
            np.testing.assert_array_equal(profile, profile_eager)
            np.testing.assert_allclose(dt, dt_eager, rtol=1e-13)
            np.testing.assert_allclose(dE, dE_eager, rtol=1e-13)

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.deferred.set_chunk_size(0)


class TestSetSpecialsCppDeferred(BLonDTestCase):
    def setUp(self):
        self.backend = Numpy64Bit()

    def tearDown(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

    @pytest.mark.backend_mutation
    def test_set_specials_cpp_deferred(self):
        self.backend.set_specials("cpp_deferred")
        self.assertEqual(self.backend.specials_mode, "cpp_deferred")
        self.assertTrue(hasattr(self.backend.specials, "n_pending"))

    @pytest.mark.backend_mutation
    def test_switching_specials_flushes_queue(self):
        self.backend.set_specials("cpp_deferred")
        dt, dE = _make_beam(100)
        dt_org = dt.copy()
        self.backend.specials.drift_simple(
            dt=dt, dE=dE, T=1.0, eta_0=1.0, beta=1.0, energy=1.0
        )
        self.backend.set_specials("cpp")
        np.testing.assert_allclose(dt, dt_org + dE)

    def test_flush_exists_on_every_eager_specials(self):
        # `Specials.flush` is a no-op outside the deferred backend, so
        # callers can flush unconditionally.
        reload_cpp_backend(np.float64).flush()
