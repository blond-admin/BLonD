import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps


h = 4620
bunch_spacing = 5
N_tau = 40

sig1 = np.zeros(h)
sig2 = np.zeros(2 * h)
sig1[::bunch_spacing] = 1
sig2[::bunch_spacing] = 1

x = np.linspace(0, h, h)
conv_sig = np.zeros(h)
conv_sig[:N_tau] = -1/N_tau * x[:N_tau] + 1

co1 = sps.fftconvolve(sig1, conv_sig, mode='full')[:sig1.shape[0]]
co2 = sps.fftconvolve(sig2, conv_sig, mode='full')[:sig2.shape[0]]

co1 = co1[-h:]
co2 = co2[-h:]



plt.plot(co1)
plt.plot(co2)
plt.show()