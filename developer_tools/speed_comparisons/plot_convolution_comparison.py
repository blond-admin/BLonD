import time
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from blond.utils import butils_wrap_cpp as _cpp


def runtime_cpu(func: Callable, args, n_iter: int = 10, n_warmup: int = 0):
    for i in range(n_warmup):
        func(*args)
    runtime = 1e22
    for i in range(n_iter):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        runtime_i = t1 - t0
        runtime = min(runtime_i, runtime)
    return t1 - t0


def plot_comparison():
    functions = (
        ("np.convolve", np.convolve),
        ("blond.cpp.convolve", _cpp.convolve),
        ("sp.fftcon", fftconvolve),
    )
    sizes = np.linspace(32, 2048, dtype=int)
    results = np.empty(len(sizes))
    for name, func in functions:
        for i, size in enumerate(sizes):
            profile_size = size
            kernel_size = 2 * profile_size
            args = (
                np.random.rand(kernel_size),
                np.random.rand(profile_size),
                "same",
            )
            runtime_i = runtime_cpu(func, args, n_warmup=1, n_iter=10)
            results[i] = runtime_i
        plt.plot(sizes, results, label=name)
    plt.legend()


if __name__ == "__main__":
    plot_comparison()
    plt.show()
