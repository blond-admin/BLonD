# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `kick_induced_voltage`."""

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
    from blond.core.backends.backend import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    from blond.core.backends.cpp.callables import CppSpecials
    from blond.core.backends.fortran.callables import FortranSpecials
    from blond.core.backends.numba.callables import recompile_numba_backend

    NumbaSpecials = recompile_numba_backend(backend.float)

    functions = (
        NumbaSpecials().change_dE_interpolated,
        CppSpecials().change_dE_interpolated,
        FortranSpecials().change_dE_interpolated,
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
