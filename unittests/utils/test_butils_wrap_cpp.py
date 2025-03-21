import time
import unittest

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
            weights = np.random.randn(n_particles).astype(float)
        else:
            weights = None
        dt_py = dt.copy()

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = min_dt
        cut_right = max_dt
        profile_py = np.empty(n_slices, dtype=float)

        bm.slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)

        bm.use_cpp()
        np.testing.assert_equal(bm.device, "CPU_CPP")
        profile_cpp = bm.empty(n_slices, dtype=float)
        bm.slice_beam(dt, profile_cpp, cut_left, cut_right, weights=weights)
        plt.plot(profile_cpp, label="profile_cpp", c="C0")
        plt.twinx()
        plt.plot(profile_py, label="profile_py", c="C1")
        plt.legend()
        plt.show()
        np.testing.assert_allclose(profile_py, profile_cpp, atol=1)
        bm.use_py()

    def test_profile_slices_old_runtime(self):
        n_particles = 5_000
        n_slices = 64

        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        weights = None
        dt_py = dt.copy()

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = min_dt
        cut_right = max_dt
        profile_py = np.empty(n_slices, dtype=float)
        profile_cpp = bm.empty(n_slices, dtype=float)

        t0 = time.time()
        slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)
        for i in range(100):
            slice_beam(dt_py, profile_py, cut_left, cut_right, weights=weights)
        t1 = time.time()
        slice_beam_tuntime = t1 - t0
        print(t1 - t0, "slice_beam")

        t0 = time.time()
        slice_beam_old(dt, profile_cpp, cut_left, cut_right, weights=weights)
        for i in range(100):
            slice_beam_old(
                dt, profile_cpp, cut_left, cut_right, weights=weights
            )
        t1 = time.time()
        slice_beam_old_tuntime = t1 - t0
        print(t1 - t0, "slice_beam_old")
        assert slice_beam_old_tuntime > slice_beam_tuntime


if __name__ == "__main__":
    unittest.main()
