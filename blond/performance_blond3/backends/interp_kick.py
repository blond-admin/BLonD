"""Testing the performance of `kick_induced_voltage`.

Authors
-------
Simon Lauber
"""

import time

import numpy as np


def main():  # pragma: no cover
    """Testing the performance of `kick_induced_voltage`."""
    dt = np.linspace(-5, 5, int(1e6))
    dE = np.zeros_like(dt)
    bin_centers = np.linspace(-4, 4, 20)
    voltage = bin_centers**2
    charge = 10
    acceleration_kick = 0
    from blond._core.backends.backend import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    from blond._core.backends.cpp.callables import CppSpecials
    from blond._core.backends.fortran.callables import FortranSpecials
    from blond._core.backends.numba.callables import NumbaSpecials

    functions = (
        NumbaSpecials().kick_induced_voltage,
        CppSpecials().kick_induced_voltage,
        FortranSpecials().kick_induced_voltage,
    )
    runtimes = {}
    for kick_induced_voltage in functions:
        runtimes[str(kick_induced_voltage)] = 0.0
    for _ in range(10000):
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
        for _ in range(10000):
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


if __name__ == "__main__":  # pragma: no cover
    main()
