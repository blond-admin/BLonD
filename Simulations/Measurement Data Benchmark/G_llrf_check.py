import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import scipy.constants as spc
import scipy.signal as spsig

from blond.impedances.impedance_sources import TravelingWaveCavity


plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble':r'\usepackage{fourier}',
    'font.family':'serif'
})

# Functions -------------------------------------------------------------------
def Gaussian_Batch(t, N_b, a, N_p, sigma, t_start):
    func = np.zeros(t.shape[0])
    for i in range(N_b):
        func += (N_p / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(1/2) * ((t - i * a - t_start)**2)/(sigma**2))
    return func

def Fourier_Gaussian_Batch(freq, N_b, a, N_p, sigma, t_start):
    func = np.zeros(freq.shape[0], dtype=complex)
    for i in range(N_b):
        func += np.exp(1j * 2 * np.pi * freq * (i * a + t_start))
    return func * (N_p / np.sqrt(2 * np.pi)) * np.exp(-(1/2) * (2 * np.pi * freq * sigma)**2)


# Travelling Wave Cavity ------------------------------------------------------
twc = TravelingWaveCavity(485202, 200.1e6, 462e-9)
twc_2 = TravelingWaveCavity(876112, 2001e6, 621e-9)

# Impedance Calculation
N_freq = 1000000
freq = np.linspace(0, 10 * 200.1e6, N_freq)
twc.imped_calc(freq)
imp = twc.impedance

# Wake Function
h = 4620
N_time = 200 * h
turn_time = 2.30545e-5
time_array = np.linspace(0, turn_time, N_time)
twc.wake_calc(time_array)
twc_2.wake_calc(time_array)
wake = twc.wake


# Beam ------------------------------------------------------------------------
N_bunches = 72
a = 25e-9
N_p = 1.17e11
sigma = 1.87e-9 / 4
t_start = 2.5e-9

batch = Gaussian_Batch(time_array, N_bunches, a, N_p, sigma, t_start)
numerical_fourier_batch = spfft.fftshift(spfft.fft(batch)) * (time_array[1]-time_array[0])

freq2 = spfft.fftshift(spfft.fftfreq(N_time, time_array[1]-time_array[0]))

fourier_batch = Fourier_Gaussian_Batch(freq2, N_bunches, a, N_p, sigma, t_start)

normalized_numerical_fourier_batch = np.copy(numerical_fourier_batch) / np.sum(np.abs(numerical_fourier_batch))
normalized_fourier_batch = np.copy(fourier_batch) / np.sum(np.abs(fourier_batch))


N_half = normalized_fourier_batch.shape[0]//2
numerical_fourier_batch = numerical_fourier_batch / np.sqrt(2 * np.pi)

twc.imped_calc(freq2)
twc_2.imped_calc(freq2)
#plt.plot(freq2[-N_half:], np.abs(twc.impedance[-N_half:]))
#plt.plot(freq2[-N_half:], np.abs(fourier_batch[-N_half:]) * spc.e, label='Analytic')
#plt.show()

#plt.plot(freq2[-N_half:], np.abs(numerical_fourier_batch[-N_half:]), label='Numerical')
#plt.plot(freq2[-N_half:], np.abs(fourier_batch[-N_half:]), label='Analytic')
#plt.legend()
#plt.show()


V_beam_from_fourier = spfft.ifftshift(spfft.ifft(fourier_batch[-N_half:] * (4 * twc.impedance[-N_half:] +
                                                                            2 * twc_2.impedance[-N_half:])))
V_beam_from_conv = -spsig.convolve(batch, 4 * wake + 2 * twc_2.wake, mode='full')

plt.plot(np.abs(V_beam_from_fourier) * spc.e)
plt.show()

plt.plot(time_array, V_beam_from_conv[:len(batch)] * spc.e * (time_array[1] - time_array[0]))
plt.show()

print(np.max(np.abs(V_beam_from_fourier)) * spc.e)

plt.plot(freq2[-N_half:], twc.impedance[-N_half:].real)
plt.plot(freq2[-N_half:], twc.impedance[-N_half:].imag)
plt.show()

# 246.17842912660558