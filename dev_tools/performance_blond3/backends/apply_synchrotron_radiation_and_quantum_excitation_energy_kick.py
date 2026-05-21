# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Testing the performance of `apply_synchrotron_radiation_and_quantum_excitation_energy_kick`.

Note: `dE` is initialised with zeros, so the damping term contributes nothing
to the floating-point work — this benchmark primarily stresses the
quantum-excitation noise generator and the memory bandwidth on `beam_dE`.
"""

import time

import numpy as np

from blond.core.backends.backend import backend


def main():  # pragma: no cover
    """Testing the performance of `apply_synchrotron_radiation_and_quantum_excitation_energy_kick`."""
    n_macroparticles = int(1e6)
    dE = np.zeros(n_macroparticles, dtype=backend.float)

    energy_lost = 1.0e3
    longitudinal_damping_time = 100.0
    natural_energy_spread = 1.0e-3
    total_energy = 1.0e9

    from blond.core.backends.cpp.callables import CppSpecials
    from blond.core.backends.numba.callables import recompile_numba_backend

    NumbaSpecials = recompile_numba_backend(backend.float)

    numba_fn = NumbaSpecials().apply_synchrotron_radiation_and_quantum_excitation_energy_kick
    cpp_fn = CppSpecials().apply_synchrotron_radiation_and_quantum_excitation_energy_kick

    cuda_fn = None
    dE_cp = None
    cp = None
    try:
        import cupy as _cp

        from blond.core.backends.cuda.callables import CudaSpecials

        cp = _cp
        dE_cp = cp.array(dE)
        cuda_fn = CudaSpecials().apply_synchrotron_radiation_and_quantum_excitation_energy_kick
    except ImportError:
        print("cupy/CUDA backend unavailable, skipping CUDA timings.")

    functions = [("numba", numba_fn), ("cpp", cpp_fn)]
    if cuda_fn is not None:
        functions.append(("cuda", cuda_fn))

    for disable_quantum_excitation in (False, True):
        print(
            f"\n=== disable_quantum_excitation={disable_quantum_excitation} ==="
        )
        runtimes = {name: 0.0 for name, _ in functions}
        for _iter in range(1000):
            for name, fn in functions:
                is_cuda = name == "cuda"
                t0 = time.perf_counter()
                fn(
                    beam_dE=dE_cp if is_cuda else dE,
                    energy_lost=energy_lost,
                    longitudinal_damping_time=longitudinal_damping_time,
                    natural_energy_spread=natural_energy_spread,
                    total_energy=total_energy,
                    disable_quantum_excitation=disable_quantum_excitation,
                )
                if is_cuda:
                    cp.cuda.runtime.deviceSynchronize()
                t1 = time.perf_counter()
                runtimes[name] += t1 - t0
        for key in sorted(runtimes.keys()):
            print(runtimes[key], key)

        print()
        runtimes = {name: 0.0 for name, _ in functions}
        for name, fn in functions:
            is_cuda = name == "cuda"
            t0 = time.perf_counter()
            for _iter in range(1000):
                fn(
                    beam_dE=dE_cp if is_cuda else dE,
                    energy_lost=energy_lost,
                    longitudinal_damping_time=longitudinal_damping_time,
                    natural_energy_spread=natural_energy_spread,
                    total_energy=total_energy,
                    disable_quantum_excitation=disable_quantum_excitation,
                )
            if is_cuda:
                cp.cuda.runtime.deviceSynchronize()
            t1 = time.perf_counter()
            runtimes[name] += t1 - t0
        for key in sorted(runtimes.keys()):
            print(runtimes[key], key)


if __name__ == "__main__":  # pragma: no cover
    main()
