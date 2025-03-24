from copy import deepcopy
from time import time, perf_counter
from typing import Tuple, Callable, Optional


def benchmark(
    func: Callable,
    args: Tuple,
    n_iter: Optional[int] = 100,
    n_warmup: Optional[int] = 1,
    verbose: bool = True,
        deepcopy_args = True,
) -> Tuple[str, float]:

    runtime_warmup = 0.0
    for _ in range(n_warmup):
        if deepcopy_args:
            args_cp = deepcopy(args)
        else:
            args_cp = args

        t0 = perf_counter()
        func(*args_cp)
        t1 = perf_counter()
        runtime_warmup += t1-t0
    runtime_warmup /= n_warmup

    runtime = 0.0
    for _ in range(n_iter):
        if deepcopy_args:
            args_cp = deepcopy(args)
        else:
            args_cp = args

        t0 = perf_counter()
        func(*args_cp)
        t1 = perf_counter()
        runtime += t1 - t0
    name = getattr(func, "__name__", repr(callable))
    runtime /= n_iter
    if verbose:
        print(name, runtime)
    return name, runtime
