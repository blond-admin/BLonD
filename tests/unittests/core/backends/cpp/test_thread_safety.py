# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Concurrent calls of the C++ kernels that keep scratch buffers between
calls. ``ctypes`` releases the GIL during a call, so Python threads running
separate simulations in one process can execute the same kernel at the same
time; each call must then still see only its own data."""

import threading
import unittest

import numpy as np

from blond.core.backends.cpp.callables import reload_cpp_backend
from blond.testing.backend_testing import BLonDTestCase

N_THREADS = 4
N_CALLS_PER_THREAD = 30
N_MACROPARTICLES = 200_000
N_SLICES = 64

# Sparse layout shared by the sparse kernels: buckets 0 and 2 filled.
FILLING_PATTERN = np.array([True, False, True])
FIRST_LEFT_CUT = 0.0
LEFT_CUT_DISTANCE = 1.0
CUT_WIDTH = 0.8
BUCKET_INDEX_TO_MEMORY_INDEX = np.array([0, 0, N_SLICES], dtype=np.int32)
N_ACTIVE_PROFILES = 2


def _run_concurrently(call, inputs):
    """Run ``call(input)`` repeatedly for every input, one thread per input.

    Parameters
    ----------
    call
        Kernel invocation returning the result for one input.
    inputs
        One input per thread.

    Returns
    -------
    results
        ``results[thread][repetition]``, the result of each call.
    """
    barrier = threading.Barrier(len(inputs))
    results = [[] for _ in inputs]

    def worker(thread_i):
        barrier.wait()
        for _ in range(N_CALLS_PER_THREAD):
            results[thread_i].append(call(inputs[thread_i]))

    threads = [
        threading.Thread(target=worker, args=(thread_i,))
        for thread_i in range(len(inputs))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _sparse_bin_centers():
    bin_width = CUT_WIDTH / N_SLICES
    centers = []
    for bucket_i, filled in enumerate(FILLING_PATTERN):
        if filled:
            cut_left = FIRST_LEFT_CUT + bucket_i * LEFT_CUT_DISTANCE
            centers.append(cut_left + (np.arange(N_SLICES) + 0.5) * bin_width)
    return np.concatenate(centers)


class TestCppKernelsThreadSafety(BLonDTestCase):
    """Each thread gets its own input; a kernel sharing its scratch buffer
    between threads mixes them up and returns another thread's data."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.specials = reload_cpp_backend(
                floattype=np.float64, parallel=True
            )
        except OSError as exc:
            raise unittest.SkipTest(f"C++ backend unavailable: {exc}")

    def setUp(self):
        self.rng = np.random.default_rng(2024)

    def assert_thread_safe(self, call, inputs):
        references = [call(input_) for input_ in inputs]
        results = _run_concurrently(call, inputs)
        for thread_i, thread_results in enumerate(results):
            for repetition, result in enumerate(thread_results):
                np.testing.assert_allclose(
                    result,
                    references[thread_i],
                    rtol=1e-12,
                    err_msg=f"{thread_i=} {repetition=}",
                )

    def test_histogram(self):
        # Different offsets give every thread a clearly different histogram.
        inputs = [
            self.rng.uniform(0.0, 1.0 + thread_i, N_MACROPARTICLES)
            for thread_i in range(N_THREADS)
        ]

        def call(dt):
            histogram = np.zeros(N_SLICES)
            self.specials.histogram(dt, histogram, 0.0, 4.0)
            return histogram

        self.assert_thread_safe(call, inputs)

    def test_histogram_sparse(self):
        inputs = [
            self.rng.uniform(0.0, 3.0, N_MACROPARTICLES) ** (1 + thread_i)
            / 3.0**thread_i
            for thread_i in range(N_THREADS)
        ]

        def call(dt):
            histogram = np.zeros(N_ACTIVE_PROFILES * N_SLICES)
            self.specials.histogram_sparse(
                dt,
                histogram,
                FIRST_LEFT_CUT,
                LEFT_CUT_DISTANCE,
                CUT_WIDTH,
                N_SLICES,
                N_ACTIVE_PROFILES,
                FILLING_PATTERN,
                BUCKET_INDEX_TO_MEMORY_INDEX,
            )
            return histogram

        self.assert_thread_safe(call, inputs)

    def test_kick_interpolated(self):
        bin_centers = np.linspace(0.0, 1.0, N_SLICES)
        dt = self.rng.uniform(0.0, 1.0, N_MACROPARTICLES)
        # Same particles, a different voltage per thread.
        inputs = [
            np.sin((1 + thread_i) * 2 * np.pi * bin_centers)
            for thread_i in range(N_THREADS)
        ]

        def call(voltage):
            dE = np.zeros(N_MACROPARTICLES)
            self.specials.kick_interpolated(
                dt, dE, voltage, bin_centers, 1.0, 0.0
            )
            return dE

        self.assert_thread_safe(call, inputs)

    def test_kick_interpolated_sparse(self):
        bin_centers = _sparse_bin_centers()
        dt = self.rng.uniform(0.0, 3.0, N_MACROPARTICLES)
        inputs = [
            np.cos((1 + thread_i) * 2 * np.pi * bin_centers)
            for thread_i in range(N_THREADS)
        ]

        def call(voltage):
            dE = np.zeros(N_MACROPARTICLES)
            self.specials.kick_interpolated(
                dt,
                dE,
                voltage,
                bin_centers,
                1.0,
                0.0,
                first_left_cut=FIRST_LEFT_CUT,
                left_cut_distance=LEFT_CUT_DISTANCE,
                cut_width=CUT_WIDTH,
                bins_per_profile=N_SLICES,
                filling_pattern=FILLING_PATTERN,
                bucket_index_to_memory_index=BUCKET_INDEX_TO_MEMORY_INDEX,
            )
            return dE

        self.assert_thread_safe(call, inputs)


if __name__ == "__main__":
    unittest.main()
