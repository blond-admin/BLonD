import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import scipy.io


from blond.llrf.signal_processing import moving_average, H_cav, moving_average_improved


# Comparison between moving average and a convolution
n_mov_avg = 38
n = 4620

t = np.linspace(0, 2 * np.pi, n)
sig = np.sin(t) + 0.05 * np.random.randn(n)

# Using the moving average function
out1 = moving_average(sig, n_mov_avg)

# Using a convolution
out2 = sps.fftconvolve(sig, (1/n_mov_avg) * np.ones(n_mov_avg))[-n:]
out2 = out2[:n - n_mov_avg + 1]


mat = scipy.io.loadmat('dataTimeDomainSim.mat')
h_3sec = mat['hCav3Section'][0]
h_4sec = mat['hCav4Section'][0]

out3 = sps.fftconvolve(sig, h_3sec)[-n:]
out3 = out3[:n - n_mov_avg + 1]
out4 = sps.fftconvolve(sig, h_4sec)[-n:]
out4 = out4[:n - n_mov_avg + 1]


print(len(out1), len(out2), len(out3), len(out4))

plt.plot(out1)
plt.plot(out2)
plt.plot(out3)
plt.plot(out4)
plt.show()
