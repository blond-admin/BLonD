import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spf

T = 100 * 2 * np.pi
N = 10000

x = np.linspace(0, T, N)
sig = np.sin(x) + np.cos(2 * x)

freq1 = 1 / 2 / np.pi
freq2 = 2 / 2 / np.pi

fsig = spf.fft(sig)
fsig = spf.fftshift(fsig)
Amp = np.abs(fsig)
Pha = np.angle(fsig)
freq = spf.fftshift(spf.fftfreq(N, T/N))
AngFreq = spf.fftshift(spf.fftfreq(N, T/N/(2 * np.pi)))

plt.plot(AngFreq, fsig.real)
plt.plot(AngFreq, fsig.imag)
#plt.vlines(freq1, np.min(fsig.real), np.max(fsig.real))
#plt.vlines(freq2, np.min(fsig.real), np.max(fsig.real))
plt.show()
