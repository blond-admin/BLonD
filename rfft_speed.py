import time

import mkl_fft
import numpy as np
import pyfftw

N = 4_500_000
lambda_arr = np.random.randn(N)
# Pre-plan once (this is slow, but you cache it)
a = pyfftw.empty_aligned(N, dtype="float64")
fft_object = pyfftw.builders.rfft(a, threads=8)
t = 0
print("starting")
a[:] = lambda_arr
for i in range(100):
    # In your loop — reuse the plan
    t0 = time.time()
    Λ_f = fft_object()
    t1 = time.time()
    t += t1 - t0
    if t > 2:
        t /= i
        break

print("pyfftw", t, "s")
t = 0
print("starting")
out = mkl_fft.rfft(lambda_arr)
for i in range(100):
    # In your loop — reuse the plan
    t0 = time.time()
    mkl_fft.rfft(lambda_arr, out=out)
    t1 = time.time()
    t += t1 - t0
    if t > 2:
        t /= i
        break
print("mkl_fft", t, "s")
