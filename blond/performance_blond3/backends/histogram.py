"""Measures the performance of the histogram function.

Authors
-------
Leonard Thiele
"""

import time

import numba
import numpy as np

from blond.core.backends.backend import backend


def main():
    """Measures the performance of the histogram function."""
    rng = np.random.default_rng(42)
    arr_sizes = np.array([1e3, 1e5, 1e7, 1e9], dtype=int)
    specials = ["numba", "python", "cpp"]
    times = np.zeros((len(specials), len(arr_sizes)))
    for arr_ind, arr_size in enumerate(arr_sizes):
        input_array = (rng.random(arr_size) - 0.5) * 20
        for spec_ind, special in enumerate(specials):
            for _ in range(10):
                backend.set_specials(special)
                numba.set_num_threads(14)
                array_write = backend.zeros(21, dtype=backend.float)
                t0 = time.perf_counter()
                backend.specials.histogram(
                    array_read=backend.array(
                        input_array, dtype=backend.float
                    ),  # casting to correct data type
                    array_write=array_write,
                    start=-12.0,
                    stop=8.0,
                )
                t1 = time.perf_counter()
                times[spec_ind, arr_ind] += t1 - t0
    print("numba times" + str(times[0].tolist()))
    print("python times" + str(times[1].tolist()))
    print("cpp times" + str(times[2].tolist()))


if __name__ == "__main__":
    main()
