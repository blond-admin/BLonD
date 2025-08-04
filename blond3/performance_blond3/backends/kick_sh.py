import time

import numpy as np

from blond3._core.backends.backend import backend, Numpy64Bit, Numpy32Bit

import cupy as cp


def main():  # pragma: no cover
    backend.change_backend(Numpy64Bit)
    backend.change_backend(Numpy32Bit)

    dt = backend.linspace(-5, 5, int(1e6), dtype=backend.float)
    dE = backend.zeros(len(dt), dtype=backend.float)
    n_rf = 2
    voltage = backend.float(5)
    omega_rf = backend.float(5)
    phi_rf = backend.float(5)

    dt_cp = cp.array(dt)
    dE_cp = cp.array(dE)


    charge = backend.float(2.0)
    acceleration_kick = backend.float(0.0)

    from blond3._core.backends.numba.callables import NumbaSpecials
    from blond3._core.backends.cpp.callables import CppSpecials
    #from blond3._core.backends.fortran.callables import FortranSpecials
    from blond3._core.backends.cuda.callables import CudaSpecials

    functions = (
        NumbaSpecials().kick_single_harmonic,
        CppSpecials().kick_single_harmonic,
        CppSpecials().kick_single_harmonic,
        #FortranSpecials().kick_single_harmonic,
        CudaSpecials().kick_single_harmonic,
    )
    runtimes = {}
    for kick_single_harmonic in functions:
        runtimes[str(kick_single_harmonic)] = 0.0
    for iter in range(1000):
        for i, kick_single_harmonic in enumerate(functions):
            CUDA = i == 3
            t0 = time.perf_counter()
            kick_single_harmonic(
                dt=dt_cp if CUDA else dt,
                dE=dE_cp if CUDA else dE,
                voltage=voltage,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                charge=charge,
                acceleration_kick=acceleration_kick,
            )
            if CUDA:
                cp.cuda.runtime.deviceSynchronize()
            t1 = time.perf_counter()
            runtimes[str(kick_single_harmonic)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)

    print()
    for i, kick_single_harmonic in enumerate(functions):
        runtimes[str(kick_single_harmonic)] = 0.0
        CUDA = i == 3
        t0 = time.perf_counter()
        for iter in range(1000):
            kick_single_harmonic(
                dt=dt_cp if CUDA else dt,
                dE=dE_cp if CUDA else dE,
                voltage=voltage,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                charge=charge,
                acceleration_kick=acceleration_kick,
            )
        if CUDA:
            cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()
        runtimes[str(kick_single_harmonic)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)


if __name__ == "__main__":  # pragma: no cover
    main()
