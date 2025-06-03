import time

import numpy as np

from blond.utils import butils_wrap_cpp


def main():
    for n_rf in (1, 2, 10):
        n_slices = 100000000
        voltages = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        bin_centers = np.linspace(1e-5, 1e-6, n_slices)

        butils_wrap_cpp.rf_volt_comp(voltages, omega_rf, phi_rf,
                                     bin_centers) # warmup

        t0 = time.time()
        butils_wrap_cpp.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)
        t1 = time.time()
        print(t1 - t0, f"s ({n_rf=})")  # will be used in `compare.py`


if __name__ == "__main__":
    main()
