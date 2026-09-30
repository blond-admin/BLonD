import ctypes as ct

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp.callables import c_index_t
from blond.testing.backend_testing import BLonDTestCase

# One 64-byte cache line of 64-bit particle coordinates.
CACHE_LINE_PARTICLES = 8


@pytest.mark.backend_mutation
class TestThreadRange(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        self.library = backend.specials._library

    def tearDown(self) -> None:
        backend.set_specials("python")

    def _ranges(self, n: int, n_threads: int) -> np.ndarray:
        begin, end = c_index_t(), c_index_t()
        ranges = []
        for thread_id in range(n_threads):
            self.library.blond_thread_range(
                c_index_t(n),
                ct.c_int(thread_id),
                ct.c_int(n_threads),
                ct.byref(begin),
                ct.byref(end),
            )
            ranges.append((begin.value, end.value))
        return np.array(ranges)

    def test_ranges_tile_zero_to_n(self) -> None:
        for n in (0, 1, 7, 8, 9, 1000, 10**4, 10**5 + 3):
            for n_threads in (1, 2, 5, 12, 64):
                with self.subTest(n=n, n_threads=n_threads):
                    ranges = self._ranges(n, n_threads)
                    self.assertEqual(ranges[0, 0], 0)
                    self.assertEqual(ranges[-1, 1], n)
                    np.testing.assert_array_equal(
                        ranges[1:, 0], ranges[:-1, 1]
                    )
                    self.assertTrue(np.all(ranges[:, 1] >= ranges[:, 0]))

    def test_inner_boundaries_are_whole_cache_lines(self) -> None:
        # Two threads writing one cache line would false-share it.
        for n in (1000, 10**4 + 5, 10**5):
            ranges = self._ranges(n, 12)
            inner = ranges[:-1, 1]
            inner = inner[inner < n]
            np.testing.assert_array_equal(inner % CACHE_LINE_PARTICLES, 0)

    def test_no_thread_exceeds_even_share_by_a_cache_line(self) -> None:
        # The slowest thread sets the kernel's time, so no thread may do
        # more than one cache line beyond an even split.
        for n in (10**4, 10**5, 10**6):
            for n_threads in (6, 12):
                with self.subTest(n=n, n_threads=n_threads):
                    ranges = self._ranges(n, n_threads)
                    largest = int(np.max(ranges[:, 1] - ranges[:, 0]))
                    even_share = -(-n // n_threads)
                    self.assertLess(largest - even_share, CACHE_LINE_PARTICLES)
