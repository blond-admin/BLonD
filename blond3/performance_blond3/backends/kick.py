import time

import cupy as cp

from blond3._core.backends.backend import backend, Numpy64Bit


def main():  # pragma: no cover
    backend.change_backend(Numpy64Bit)

    dt = backend.linspace(
        -5,
        5,
        int(1e6),
        dtype=backend.float,
    )
    dE = backend.zeros(
        len(dt),
        dtype=backend.float,
    )
    n_rf = 2
    voltage = backend.linspace(
        1,
        5,
        n_rf,
        dtype=backend.float,
    )
    omega_rf = backend.linspace(
        1,
        5,
        n_rf,
        dtype=backend.float,
    )
    phi_rf = backend.linspace(
        1,
        5,
        n_rf,
        dtype=backend.float,
    )

    dt_cp = cp.array(dt)
    dE_cp = cp.array(dE)
    voltage_cp = cp.array(voltage)
    omega_rf_cp = cp.array(omega_rf)
    phi_rf_cp = cp.array(phi_rf)

    charge = backend.float(2.0)
    acceleration_kick = backend.float(0.0)

    from blond3._core.backends.numba.callables import NumbaSpecials
    from blond3._core.backends.cpp.callables import CppSpecials
    from blond3._core.backends.fortran.callables import FortranSpecials
    from blond3._core.backends.cuda.callables import CudaSpecials

    functions = (
        NumbaSpecials().kick_multi_harmonic,
        CppSpecials().kick_multi_harmonic,
        FortranSpecials().kick_multi_harmonic,
        CudaSpecials().kick_multi_harmonic,
    )
    runtimes = {}
    for kick_multi_harmonic in functions:
        runtimes[str(kick_multi_harmonic)] = 0.0
    for iter in range(1000):
        for i, kick_multi_harmonic in enumerate(functions):
            CUDA = kick_multi_harmonic == CudaSpecials().kick_multi_harmonic
            t0 = time.perf_counter()
            kick_multi_harmonic(
                dt=dt_cp if CUDA else dt,
                dE=dE_cp if CUDA else dE,
                voltage=voltage_cp if CUDA else voltage,
                omega_rf=omega_rf_cp if CUDA else omega_rf,
                phi_rf=phi_rf_cp if CUDA else phi_rf,
                charge=charge,
                n_rf=n_rf,
                acceleration_kick=acceleration_kick,
            )
            if CUDA:
                cp.cuda.runtime.deviceSynchronize()
            t1 = time.perf_counter()
            runtimes[str(kick_multi_harmonic)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)

    print()
    for i, kick_multi_harmonic in enumerate(functions):
        runtimes[str(kick_multi_harmonic)] = 0.0
        CUDA = kick_multi_harmonic == CudaSpecials().kick_multi_harmonic
        t0 = time.perf_counter()
        for iter in range(1000):
            kick_multi_harmonic(
                dt=dt_cp if CUDA else dt,
                dE=dE_cp if CUDA else dE,
                voltage=voltage_cp if CUDA else voltage,
                omega_rf=omega_rf_cp if CUDA else omega_rf,
                phi_rf=phi_rf_cp if CUDA else phi_rf,
                charge=charge,
                n_rf=n_rf,
                acceleration_kick=acceleration_kick,
            )
        if CUDA:
            cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()
        runtimes[str(kick_multi_harmonic)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)


if __name__ == "__main__":  # pragma: no cover
    main()
