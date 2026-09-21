# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The C++ `histogram` is told per call how many threads to use.

Every thread of `histogram` pays for waking up and for zeroing and
reducing a private histogram of `n_slices` bins. With few particles or
many bins that overhead exceeds the counting it parallelises, so the
Python wrapper asks `histogram_n_threads` how many threads are worth
using and hands that number to the kernel.
"""

from __future__ import annotations

import ctypes as ct
import os
from unittest import mock

import numpy as np

from blond.core.backends.cpp import callables, histogram_n_threads
from blond.testing.backend_testing import BLonDTestCase

MAX_THREADS = 16
N_BINS_IN_CACHE = 2**10
N_BINS_IN_MAIN_MEMORY = 2**21


class TestHistogramNThreads(BLonDTestCase):
    """`histogram_n_threads(n_macroparticles, n_slices, max_threads)`."""

    def test_few_particles_run_on_one_thread(self) -> None:
        """Waking a thread costs more than counting 1000 particles."""
        self.assertEqual(
            histogram_n_threads.histogram_n_threads(
                1_000, N_BINS_IN_CACHE, MAX_THREADS
            ),
            1,
        )

    def test_many_particles_run_on_all_threads(self) -> None:
        """A cached histogram costs nothing, so all threads count."""
        self.assertEqual(
            histogram_n_threads.histogram_n_threads(
                10**8, N_BINS_IN_CACHE, MAX_THREADS
            ),
            MAX_THREADS,
        )

    def test_thread_count_grows_with_the_particles(self) -> None:
        """More particles never justify fewer threads."""
        for n_bins in (N_BINS_IN_CACHE, N_BINS_IN_MAIN_MEMORY):
            counts = [
                histogram_n_threads.histogram_n_threads(
                    10**exponent, n_bins, MAX_THREADS
                )
                for exponent in range(9)
            ]
            self.assertEqual(counts, sorted(counts), msg=f"{n_bins=}")

    def test_large_histogram_runs_on_fewer_threads(self) -> None:
        """Private histograms in main memory are zeroed once per thread."""
        n_macroparticles = 10**6
        self.assertLess(
            histogram_n_threads.histogram_n_threads(
                n_macroparticles, N_BINS_IN_MAIN_MEMORY, MAX_THREADS
            ),
            histogram_n_threads.histogram_n_threads(
                n_macroparticles, N_BINS_IN_CACHE, MAX_THREADS
            ),
        )

    def test_stays_within_one_and_max_threads(self) -> None:
        """The result is a valid OpenMP team size for any input."""
        for max_threads in (1, 2, MAX_THREADS):
            for n_macroparticles in (0, 1, 10**5, 10**10):
                for n_bins in (1, N_BINS_IN_CACHE, N_BINS_IN_MAIN_MEMORY):
                    n_threads = histogram_n_threads.histogram_n_threads(
                        n_macroparticles, n_bins, max_threads
                    )
                    self.assertIsInstance(n_threads, int)
                    self.assertGreaterEqual(n_threads, 1)
                    self.assertLessEqual(n_threads, max_threads)

    def test_constants_can_be_tuned_at_runtime(self) -> None:
        """The constants describe the machine, so a user may set them."""
        n_macroparticles = 10**6
        with mock.patch.object(
            histogram_n_threads, "PARTICLES_PER_THREAD", n_macroparticles // 2
        ):
            self.assertEqual(
                histogram_n_threads.histogram_n_threads(
                    n_macroparticles, N_BINS_IN_CACHE, MAX_THREADS
                ),
                2,
            )


class TestHistogramKernelNThreads(BLonDTestCase):
    """The C++ `histogram` takes its thread count as last argument."""

    def setUp(self) -> None:
        folder = os.path.dirname(os.path.abspath(callables.__file__))
        __, libblond_path = callables._make_libblond_path(folder, "double")
        if not os.path.isfile(libblond_path):
            self.skipTest("needs the compiled C++ backend")
        self.library = callables._get_libblond(libblond_path)

    def _histogram(self, coordinates: np.ndarray, n_threads: int):
        profile = np.full(64, np.nan)
        self.library.histogram(
            coordinates.ctypes.data_as(ct.c_void_p),
            profile.ctypes.data_as(ct.c_void_p),
            ct.c_double(-12.0),
            ct.c_double(8.0),
            ct.c_int(len(profile)),
            callables.c_index_t(len(coordinates)),
            ct.c_int(n_threads),
        )
        return profile

    def test_result_does_not_depend_on_the_thread_count(self) -> None:
        """The counters are integers: any team gives the same profile."""
        coordinates = (np.random.default_rng(42).random(100_000) - 0.5) * 20
        expected, __ = np.histogram(coordinates, bins=64, range=(-12.0, 8.0))
        for n_threads in (1, 2, 5):
            np.testing.assert_array_equal(
                self._histogram(coordinates, n_threads),
                expected,
                err_msg=f"{n_threads=}",
            )
