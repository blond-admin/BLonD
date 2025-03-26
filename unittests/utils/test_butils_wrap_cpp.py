import time
import unittest
from time import time

import numpy as np
from matplotlib import pyplot as plt
from parameterized import parameterized

from blond.utils import bmath as bm
from blond.utils.butils_wrap_cpp import slice_beam_old, slice_beam


class TestSlices(unittest.TestCase):
    @parameterized.expand([("use_weights",), (False,)])
    def test_profile_slices_cpp(self, use_weights: bool):
        bm.use_py()
        n_particles = 5_000
        n_slices = 64

        np.testing.assert_equal(bm.device, "CPU_PY")

        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        if use_weights is not False:
            weights = (1e3 * np.random.rand(n_particles)).astype(np.int32)
        else:
            weights = None
        dt_py = dt.copy()

        max_dt = dt.max() - 0.1 * 1e-6
        min_dt = dt.min() + 0.1 * 1e-6
        cut_left = min_dt
        cut_right = max_dt
        profile_py = np.empty(n_slices, dtype=float)

        bm.slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)

        bm.use_cpp()
        np.testing.assert_equal(bm.device, "CPU_CPP")
        profile_cpp = bm.empty(n_slices, dtype=float)
        bm.slice_beam(dt, profile_cpp, cut_left, cut_right, weights=weights)
        np.testing.assert_allclose(profile_py, profile_cpp, atol=1)
        bm.use_py()

    @unittest.skip("For devs only")
    def test_profile_slices_old_runtime(self):
        n_particles = int(1e7)
        n_slices = 64

        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        weights = None
        dt_py = dt.copy()

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = min_dt - 0.1 * 1e-6
        cut_right = max_dt - 0.1 * 1e-6
        profile_py = np.empty(n_slices, dtype=float)
        profile_cpp = bm.empty(n_slices, dtype=float)

        slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)
        slice_beam_runtime = 0.0
        slice_beam_old(dt, profile_cpp, cut_left, cut_right, weights=weights)
        slice_beam_old_runtime = 0.0
        for i in range(100):
            t0 = time()
            slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)
            t1 = time()
            slice_beam_runtime += t1 - t0

            t0 = time()
            slice_beam_old(
                dt, profile_cpp, cut_left, cut_right, weights=weights
            )
            t1 = time()
            slice_beam_old_runtime += t1 - t0
        slice_beam_runtime /= i + 1
        slice_beam_old_runtime /= i + 1

        print(slice_beam_runtime, "slice_beam")

        print(slice_beam_old_runtime, "slice_beam_old")
        assert slice_beam_old_runtime > slice_beam_runtime


if __name__ == "__main__":
    unittest.main()
