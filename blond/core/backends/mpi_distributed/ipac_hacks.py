import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True)
def parallel_sum(x):
    acc = 0.0
    for i in prange(x.shape[0]):
        acc += x[i]
    return acc


@njit(parallel=True, cache=True, fastmath=True)
def parallel_dot(x, y):
    acc = 0.0
    for i in prange(x.shape[0]):
        acc += x[i] * y[i]
    return acc


if __name__ == "__main__":
    x = np.random.rand(10_000_000).astype(np.float64)
    y = np.random.rand(10_000_000).astype(np.float64)

    s = parallel_sum(x)
    d = parallel_dot(x, y)

    print(s, np.sum(x))
    print(d, np.dot(x, y))
