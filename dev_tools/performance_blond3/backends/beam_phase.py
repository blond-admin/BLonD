# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `beam_phase`."""

import time

import cupy as cp
import numpy as np


def main():  # pragma: no cover
    """Testing the performance of `beam_phase`."""
    hist_x = np.linspace(0, 1, 1024)
    rng = np.random.default_rng()
    hist_y = rng.standard_normal(len(hist_x))
    hist_x_cp = cp.array(hist_x)
    hist_y_cp = cp.array(hist_y)
    alpha = 1.4
    omega_rf = 1.4
    phi_rf = 1.4
    bin_size = 1.4
    from blond.core.backends.backend import Numpy64Bit, backend

    backend.change_backend(Numpy64Bit)
    from blond.core.backends.cpp.callables import CppSpecials
    from blond.core.backends.cuda.callables import CudaSpecials
    from blond.core.backends.numba.callables import recompile_numba_backend
    from blond.core.backends.python.callables import PythonSpecials

    NumbaSpecials = recompile_numba_backend(backend.float)

    print(f"Testing `beam_phase` for {len(hist_x)} bins..")
    functions = (
        PythonSpecials().beam_phase,
        NumbaSpecials().beam_phase,
        CppSpecials().beam_phase,
        CudaSpecials().beam_phase,
    )
    runtimes = {}
    for beam_phase in functions:
        runtimes[str(beam_phase)] = 0.0
    for _ in range(10000):
        for _, beam_phase in enumerate(functions):
            CUDA = beam_phase == CudaSpecials().beam_phase
            t0 = time.perf_counter()
            beam_phase(
                hist_x=hist_x if not CUDA else hist_x_cp,
                hist_y=hist_y if not CUDA else hist_y_cp,
                alpha=alpha,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                bin_size=bin_size,
            )
            t1 = time.perf_counter()
            runtimes[str(beam_phase)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)
    print()
    for beam_phase in functions:
        runtimes[str(beam_phase)] = 0.0
    for _i, beam_phase in enumerate(functions):
        CUDA = beam_phase == CudaSpecials().beam_phase
        for _ in range(10000):
            t0 = time.perf_counter()
            beam_phase(
                hist_x=hist_x if not CUDA else hist_x_cp,
                hist_y=hist_y if not CUDA else hist_y_cp,
                alpha=alpha,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                bin_size=bin_size,
            )
            t1 = time.perf_counter()
            runtimes[str(beam_phase)] += t1 - t0
    for key in sorted(runtimes.keys()):
        print(runtimes[key], key)


if __name__ == "__main__":  # pragma: no cover
    main()
