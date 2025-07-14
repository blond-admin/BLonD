import numpy as np

a = np.fft.rfft(np.random.rand(1024))  # frequency-domain data
out = np.empty(512)  # too small!

np.fft.irfft(a, n=1024, out=out)
