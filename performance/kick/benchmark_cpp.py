import time

import numpy as np

from blond.utils import butils_wrap_cpp

from scipy.constants import elementary_charge


def main():
    for n_rf in (1, 2, 10):
        n_particles = int(1e6)
        voltages = np.random.randn(n_rf)
        charge = elementary_charge
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        dt = np.random.randn(n_particles)
        dE = np.random.randn(n_particles)
        acceleration_kick = 1.0
        butils_wrap_cpp.kick(
            dt, dE, voltages, omega_rf, phi_rf, charge, n_rf, acceleration_kick
        )  # warmup

        t0 = time.time()
        butils_wrap_cpp.kick(
            dt, dE, voltages, omega_rf, phi_rf, charge, n_rf, acceleration_kick
        )
        t1 = time.time()
        print(t1 - t0, f"s ({n_rf=})")  # will be used in `compare.py`


if __name__ == "__main__":
    main()
