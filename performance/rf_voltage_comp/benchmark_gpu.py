import time

import cupy as cp
import numpy as np
from cupy import cuda

from blond.utils import bmath as bm
from blond.gpu import butils_wrap_cupy


def main():
    for n_rf in (1, 2, 10):
        n_slices = 100000
        voltages = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        bin_centers = np.linspace(1e-5, 1e-6, n_slices)

        bm.use_gpu()
        voltages = cp.array(voltages)
        omega_rf = cp.array(omega_rf)
        phi_rf = cp.array(phi_rf)
        bin_centers = cp.array(bin_centers)
        t0 = time.time()
        butils_wrap_cupy.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)
        cuda.Device().synchronize()
        t1 = time.time()
        print(t1 - t0, f"s ({n_rf=})" )  # will be used in `compare.py`


if __name__ == "__main__":
    main()
