import math
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from helper_functions import benchmark

from blond.utils.butils_wrap_numba import (
    slice_beam as slice_beam_numba,
    slice_beam_old as slice_beam_old_numba,
)

from blond.utils.butils_wrap_cpp import (
    slice_beam as slice_beam_cpp,
    slice_beam_old as slice_beam_old_cpp,
)


def compare_runtime(func_old: Callable, func_new: Callable, len_dt:int, len_prof:int):
    global dt, profile, cut_left, cut_right
    dt = np.random.randn(len_dt)  # TODO
    profile = np.empty(len_prof)  # TODO
    cut_left = -1.0  # TODO
    cut_right = 1.0  # TODO
    args = (
        dt,
        profile,
        cut_left,
        cut_right,
    )
    n_iter = int((1e7 - len_dt) * 1e-4)
    print(f"{n_iter=}")
    if n_iter < 1:
        n_iter = 1
    name_old, time_old = benchmark(
        func_old,
        args,
        n_iter=n_iter,
        n_warmup=1,
        verbose=True,
        deepcopy_args=False,
    )
    name_new, time_new = benchmark(
        func_new,
        args,
        n_iter=n_iter,
        n_warmup=1,
        verbose=True,
        deepcopy_args=False,
    )
    return (time_old / time_new - 1) * 100


for func_old, func_new in (
    (slice_beam_old_cpp, slice_beam_cpp),
    #(slice_beam_old_numba, slice_beam_numba),
):
    xs = np.linspace(10, int(1e7), 20, dtype=int)
    ys = np.zeros_like(xs)
    zs_ = np.zeros_like(ys)
    len_prof = 64
    while True:
        for i, len_dt in enumerate(xs):
            speedup = compare_runtime(func_old, func_new, len_dt=len_dt, len_prof=len_prof)
            ys[i] += speedup
            zs_[i] += 1
            print(f"{speedup=}")
        plt.cla()
        plt.plot(xs[1:], ys[1:] / zs_[1:], "o-", label="slice_beam_old_numba")
        plt.axhline(0, color="k")
        plt.draw()
        plt.pause(0.1)
