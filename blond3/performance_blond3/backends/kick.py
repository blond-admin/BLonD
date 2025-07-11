import time

import numpy as np

dt = np.linspace(-5, 5, int(1e6))
dE = np.zeros_like(dt)
n_rf = 2
voltage = np.linspace(1, 5, n_rf)
omega_rf = np.linspace(1, 5, n_rf)
phi_rf = np.linspace(1, 5, n_rf)

charge = 2
acceleration_kick = 0
from blond3._core.backends.backend import backend, Numpy64Bit

backend.change_backend(Numpy64Bit)
from blond3._core.backends.numba.callables import NumbaSpecials
from blond3._core.backends.cpp.callables import CppSpecials
from blond3._core.backends.fortran.callables import FortranSpecials

functions = (
    NumbaSpecials().kick_multi_harmonic,
    CppSpecials().kick_multi_harmonic,
    FortranSpecials().kick_multi_harmonic,
)
runtimes = {}
for kick_multi_harmonic in functions:
    runtimes[str(kick_multi_harmonic)] = 0.0
for iter in range(1000):
    for kick_multi_harmonic in functions:
        t0 = time.perf_counter()
        kick_multi_harmonic(
            dt=dt,
            dE=dE,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            n_rf=n_rf,
            acceleration_kick=acceleration_kick,
        )
        t1 = time.perf_counter()
        runtimes[str(kick_multi_harmonic)] += t1 - t0
for key in sorted(runtimes.keys()):
    print(runtimes[key], key)

print()
for kick_multi_harmonic in functions:
    runtimes[str(kick_multi_harmonic)] = 0.0
for kick_multi_harmonic in functions:
    for iter in range(1000):
        t0 = time.perf_counter()
        kick_multi_harmonic(
            dt=dt,
            dE=dE,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            n_rf=n_rf,
            acceleration_kick=acceleration_kick,
        )
        t1 = time.perf_counter()
        runtimes[str(kick_multi_harmonic)] += t1 - t0
for key in sorted(runtimes.keys()):
    print(runtimes[key], key)
