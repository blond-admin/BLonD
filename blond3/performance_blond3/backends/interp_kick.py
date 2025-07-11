import time

import numpy as np

dt = np.linspace(-5, 5, int(1e6))
dE = np.zeros_like(dt)
bin_centers = np.linspace(-4, 4, 20)
voltage = bin_centers**2
charge = 10
acceleration_kick = 0
from blond3._core.backends.backend import backend, Numpy64Bit

backend.change_backend(Numpy64Bit)
from blond3._core.backends.numba.callables import NumbaSpecials
from blond3._core.backends.cpp.callables import CppSpecials
from blond3._core.backends.fortran.callables import FortranSpecials

functions = (
    NumbaSpecials().kick_induced_voltage,
    CppSpecials().kick_induced_voltage,
    FortranSpecials().kick_induced_voltage,
)
runtimes = {}
for kick_induced_voltage in functions:
    runtimes[str(kick_induced_voltage)] = 0.0
for iter in range(10000):
    for kick_induced_voltage in functions:
        t0 = time.perf_counter()
        kick_induced_voltage(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )
        t1 = time.perf_counter()
        runtimes[str(kick_induced_voltage)] += t1 - t0
for key in sorted(runtimes.keys()):
    print(runtimes[key], key)


for kick_induced_voltage in functions:
    runtimes[str(kick_induced_voltage)] = 0.0
for kick_induced_voltage in functions:
    for iter in range(10000):
        t0 = time.perf_counter()
        kick_induced_voltage(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )
        t1 = time.perf_counter()
        runtimes[str(kick_induced_voltage)] += t1 - t0
for key in sorted(runtimes.keys()):
    print(runtimes[key], key)
