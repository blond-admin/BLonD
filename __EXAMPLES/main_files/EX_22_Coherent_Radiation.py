#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:11:37 2020

@author: MarkusArbeit
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as cLight
from scipy.constants import e as elCharge

from blond.impedances.impedance_sources import CoherentSynchrotronRadiation

def w(mu):
    ret = np.zeros_like(mu)
    
    negIndices = mu<0
    zeroIndex = mu==0
    posIndices = mu>0
    # print(negIndices, zeroIndex, posIndices, type(ret))
    ret[negIndices] = 0
    ret[zeroIndex] = 0.5
    mu = mu[posIndices]
    ret[posIndices] =\
        9 * (1-((1+2*mu**2)*np.cosh(5*np.arcsinh(mu)/3)/np.sqrt(1+mu**2)
                - 5*mu*np.sinh(5*np.arcsinh(mu)/3)/3) / (1+mu**2)
              )/ (8*mu**2)
    
    return ret


h, rBend = 32e-3, 1.273  # chamber height [m], bending radius[m]
fRev = cLight / (2*np.pi*rBend)
gamma = 40e6/511e3

fcut = np.sqrt(2/3) * (np.pi * rBend/h)**(3/2) * fRev  # cut off frequency [Hz]
fc = 3*gamma**3 * cLight / (4*np.pi*rBend)  # critical frequency [Hz]

freqs = 10**np.linspace(8,15, num=200)

Z_fs_appr = CoherentSynchrotronRadiation(rBend)
Z_fs_appr.imped_calc(freqs)

Z_fs = CoherentSynchrotronRadiation(rBend, gamma=gamma)
Z_fs.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)

Z_pp_appr = CoherentSynchrotronRadiation(rBend, chamber_height=h)
Z_pp_appr.imped_calc(freqs, high_frequency_transition=10)

Z_pp = CoherentSynchrotronRadiation(rBend, chamber_height=h, gamma=gamma)
Z_pp.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)

plt.figure('re Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-0.1, 12.1)
plt.plot(freqs, Z_fs.impedance.real / 1e3, '.-', label='free-space')
plt.plot(freqs, Z_pp.impedance.real / 1e3, label='parallel-plates')
plt.plot(freqs, Z_fs_appr.impedance.real / 1e3, label='free-space, approx.')
plt.plot(freqs, Z_pp_appr.impedance.real / 1e3, '--', label='parallel-plates, approx.')
plt.axvline(fc, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(fcut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
# plt.savefig('./csr_impedance_real.png')

plt.figure('im Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-10.1, 8.1)
plt.plot(freqs, Z_fs.impedance.imag / 1e3, '.-', label='free-space')
plt.plot(freqs, Z_pp.impedance.imag / 1e3, label='parallel-plates')
plt.plot(freqs, Z_fs_appr.impedance.imag / 1e3, label='free-space, approx.')
plt.plot(freqs, Z_pp_appr.impedance.imag / 1e3, '--', label='parallel-plates, approx.')
plt.axvline(fc, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(fcut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
# plt.savefig('./csr_impedance_imag.png')

# freqs = np.linspace(0,1e14, num=1000)
# freqs = np.linspace(0,1e16, num=10000)
freqs = np.linspace(0,1e16, num=10000)
print((freqs[1]-freqs[0])/fc)
nFreq = len(freqs)
nTime = 2*(nFreq-1)
extFreqs = np.zeros(nTime)
extFreqs[:nFreq-1] = freqs[:-1]
extFreqs[nFreq-1:] = -np.flip(freqs[1:])

FFTtime = np.fft.fftfreq(extFreqs.size, extFreqs[1]-extFreqs[0])
FFTtime = np.fft.fftshift(FFTtime)  # shift to normal time order


Z_fs_appr.imped_calc(freqs)
Z_fs.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)
# Z_pp_appr.imped_calc(freqs, high_frequency_transition=10)
# Z_pp.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)

W_fs_exact_FFT = np.fft.ifftshift(np.fft.irfft(Z_fs.impedance)) * (freqs[1]-freqs[0]) * FFTtime.size
W_fs_exact_anal = Z_fs.Z0 * gamma * fc/(2*rBend) * 8/9 * w(2*np.pi*fc * FFTtime)
W_fs_exact_anal *= 2 *np.pi * np.sqrt(2)

plt.figure('W from FFT', clear=True)
plt.grid()
plt.xlim(-5e-14,5e-14)
plt.ylim(-5e17,2.5e18)
plt.plot(FFTtime, W_fs_exact_FFT, '.-', label='free space, exact, iFFT')
plt.plot(FFTtime, W_fs_exact_anal, '--', label='free space, exact, anal')


plt.figure('Z for FFT', clear=True)
plt.grid()
plt.ylim(-10,4)
plt.plot(freqs, Z_fs.impedance.imag / 1e3, '.-', label='free space, exact')
# plt.plot(freqs, Z_pp.impedance.imag / 1e3, label='parallel plates, exact')
plt.plot(freqs, Z_fs_appr.impedance.imag / 1e3, label='free space, aprrox.')
# plt.plot(freqs, Z_pp_appr.impedance.imag / 1e3, '--', label='parallel plates, approx.')
plt.axvline(fc, color='red', linestyle='--')

# freqs = np.linspace(0,1e15, num=100)

# sigma = 1e-15
# tmp_FFT = np.exp(-0.5*(2*np.pi*freqs*sigma)**2)

# nFreq = len(freqs)
# nTime = 2*(nFreq-1)
# extFreqs = np.zeros(nTime)
# extFreqs[:nFreq-1] = freqs[:-1]
# extFreqs[nFreq-1:] = -np.flip(freqs[1:])

# FFTtime = np.fft.fftfreq(extFreqs.size, extFreqs[1]-extFreqs[0])
# FFTtime = np.fft.fftshift(FFTtime)  # shift to normal time order

# plt.figure('tmp, freq', clear=True)
# plt.grid()
# plt.plot(freqs, tmp_FFT)

# tmp_time = np.fft.ifftshift(np.fft.irfft(tmp_FFT)) * (freqs[1]-freqs[0]) * FFTtime.size
# plt.figure('tmp, time', clear=True)
# plt.grid()
# plt.plot(FFTtime, tmp_time)

