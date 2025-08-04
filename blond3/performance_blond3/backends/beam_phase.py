import sys
import time

import numpy as np
import cupy as cp

def main():  # pragma: no cover
    dt = np.linspace(-5, 5, int(1e6))
    hist_x = np.linspace(0, 1, 1024)
    hist_y = np.random.randn(len(hist_x))
    hist_x_cp = cp.array(hist_x)
    hist_y_cp = cp.array(hist_y)
    alpha = float(1.4)
    omega_rf = float(1.4)
    phi_rf = float(1.4)
    bin_size = float(1.4)
    from blond3._core.backends.backend import backend, Numpy64Bit

    backend.change_backend(Numpy64Bit)
    from blond3._core.backends.python.callables import PythonSpecials
    from blond3._core.backends.numba.callables import NumbaSpecials
    from blond3._core.backends.cpp.callables import CppSpecials
    from blond3._core.backends.cuda.callables import CudaSpecials
    from blond3._core.backends.fortran.callables import FortranSpecials

    print(f"Testing `beam_phase` for {len(hist_x)} bins..")
    functions = (
        PythonSpecials().beam_phase,
        NumbaSpecials().beam_phase,
        CppSpecials().beam_phase,
        CudaSpecials().beam_phase,
        FortranSpecials().beam_phase,
    )
    runtimes = {}
    for beam_phase in functions:
        runtimes[str(beam_phase)] = 0.0
    for iter in range(10000):
        for i, beam_phase in enumerate(functions):
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
    for i, beam_phase in enumerate(functions):
        CUDA = beam_phase == CudaSpecials().beam_phase
        for iter in range(10000):
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
