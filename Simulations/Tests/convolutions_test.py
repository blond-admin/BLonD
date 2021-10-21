
# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utils_test as ut
import sys
import scipy.signal

from blond.utils import bmath as bm


# Parameters ------------------------------------------------------------------
Nx = 200 #4620                                   # Number of points along the x-axis
N_step_len = 100                             # Length of the square function
step_pos = 0                                # Position of the square function
Amp = 1                                     # Amplitude of the constant signal


# Arrays ----------------------------------------------------------------------
x = np.arange(Nx)

square = ut.make_step(Nx, N_step_len, 1, step_pos)
#square = Amp * np.ones(200)
constant_signal = Amp * np.ones(50)
square2 = ut.make_step(100, 50, 1, 0)

plt.plot(square)
plt.plot(constant_signal)
plt.show()

# Convolutions ----------------------------------------------------------------

# SicPy FFT convolutions
scipy_fft_full = scipy.signal.fftconvolve(square, constant_signal, mode='full')
scipy_fft_valid = scipy.signal.fftconvolve(square, constant_signal, mode='valid')
scipy_fft_same = scipy.signal.fftconvolve(square, constant_signal, mode='same')
scipy_fft_same2 = scipy.signal.fftconvolve(square, square2, mode='full')

print(len(scipy_fft_full), len(scipy_fft_valid), len(scipy_fft_same), len(scipy_fft_same2))

# SciPy convolutions
scipy_full_direct = scipy.signal.convolve(square, constant_signal, mode='full', method='direct')
scipy_full_fft = scipy.signal.convolve(square, constant_signal, mode='full', method='fft')

# NumPy convolution
numpy_full = np.convolve(square, constant_signal, mode='full')

# BLonD C++ convolution
signal = np.ascontiguousarray(square)
kernel = np.ascontiguousarray(constant_signal)

result = np.zeros(len(kernel) + len(signal) - 1)
bm.convolve(signal, kernel, result=result, mode='full')


# Plots -----------------------------------------------------------------------

# Comparison with different full convolutions
plt.plot(scipy_fft_full, label='scipy_fft_full', linestyle='-', marker='x')
plt.plot(scipy_full_fft, label='scipy_full_fft', linestyle='--')
plt.plot(scipy_full_direct, label='scipy_full_direct', linestyle='-.')
plt.plot(numpy_full, label='numpy_full', linestyle='dotted')
plt.legend()
plt.show()

# Comparison between different modes
#plt.plot(scipy_fft_full, label='scipy_fft_full', linestyle='-')
#plt.plot(scipy_fft_valid, label='scipy_fft_valid', linestyle='--')
#plt.plot(scipy_fft_same, label='scipy_fft_same', linestyle='dotted')
#plt.legend()
#plt.show()


# Input with output
plt.plot(50 * square, label='square', linestyle='-')
plt.plot(scipy_fft_full, label='scipy_fft_full', linestyle='--')
plt.plot(scipy_fft_same2, label='square square', linestyle='dotted')
plt.plot(result)
plt.legend()
plt.show()