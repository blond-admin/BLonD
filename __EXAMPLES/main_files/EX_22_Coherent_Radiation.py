#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:11:37 2020

@author: MarkusArbeit
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as cLight

from blond.impedances.impedance_sources import CoherentSynchrotronRadiation

h, rBend = 32e-3/2, 1.273  # [m], [m]
fRev = cLight / (2*np.pi*rBend)
# gamma = 42e6/511e3
gamma = 40e6/511e3
# fcut = cLight / h  # [Hz]
# fcut = np.pi * rBend /(h*1) * fRev
fcut = np.sqrt(2/3) * (np.pi * rBend/h)**(3/2) * fRev
fc = 3*gamma**3 * cLight / (4*np.pi*rBend)  # [Hz]

freqs = np.linspace(8,15, num=100)
# freqs = np.linspace(10,14, num=40)
freqs = 10**freqs

Z_fs_low = CoherentSynchrotronRadiation(rBend)
Z_fs_low.imped_calc(freqs)

Z_fs = CoherentSynchrotronRadiation(rBend, gamma=gamma)
Z_fs.imped_calc(freqs, low_frequency_transition=1e-4)

Z_fs2 = CoherentSynchrotronRadiation(rBend, gamma=gamma)
Z_fs2.imped_calc = Z_fs2._fs_exact_spectrum2
Z_fs2.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=10)


Z_pp_low = CoherentSynchrotronRadiation(rBend, chamber_height=h)
Z_pp_low.imped_calc(freqs, high_frequency_transition=10)

Z_pp = CoherentSynchrotronRadiation(rBend, chamber_height=h, gamma=gamma)
Z_pp.imped_calc(freqs, low_frequency_transition=1e-4)

Z_pp2 = CoherentSynchrotronRadiation(rBend, chamber_height=h, gamma=gamma)
Z_pp2.imped_calc = Z_pp2._pp_exact_spectrum2
Z_pp2.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=15)


plt.figure('re Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-0.1, 12.1)
plt.plot(freqs, Z_fs_low.impedance.real / 1e3, label='free space, low-freq')
plt.plot(freqs, Z_fs.impedance.real / 1e3, label='free space, exact')
plt.plot(freqs, Z_fs2.impedance.real / 1e3, '--', label='free space2, exact')
# plt.plot(freqs, Z_pp_low.impedance.real / 1e3, label='parallel plates, low-freq')
plt.plot(freqs, Z_pp.impedance.real / 1e3, label='parallel plates, exact')
plt.plot(freqs, Z_pp2.impedance.real / 1e3, '--', label='parallel plates, exact 2')
plt.axvline(fc, color='red', linestyle='--')
plt.axvline(fcut, linestyle='--')
plt.legend()
plt.tight_layout()

plt.figure('im Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-10.1, 8.1)
plt.plot(freqs, Z_fs_low.impedance.imag / 1e3, label='free space, low-freq')
plt.plot(freqs, Z_fs.impedance.imag / 1e3, label='free space, exact')
plt.plot(freqs, Z_fs2.impedance.imag / 1e3, '--', label='free space2, exact')
# plt.plot(freqs, Z_pp_low.impedance.imag / 1e3, label='parallel plates, low-freq')
plt.plot(freqs, Z_pp.impedance.imag / 1e3, label='parallel plates, exact')
plt.plot(freqs, Z_pp2.impedance.imag / 1e3, '--', label='parallel plates, exact 2')
plt.axvline(fc, color='red', linestyle='--')
plt.axvline(fcut, linestyle='--')
plt.legend()
plt.tight_layout()

# plt.figure('tmp', clear=True)
# plt.grid()
# plt.plot(freqs, Z_fs.impedance.imag / Z_fs2.impedance.imag ,)

# #%%
# zeta = np.linspace(0,15)

# h_anal = Z_pp._hFun(np.exp(1j*np.pi/3) * zeta)

# h_num = np.zeros_like(h_anal)
# for it, zet in enumerate(zeta):
#     h_num[it] = Z_pp._hFun(np.exp(1j*np.pi/3) * zet)

# plt.figure('h', clear=True)
# plt.grid()
# plt.plot(zeta, h_anal.real)
# plt.plot(zeta, h_anal.imag)
# plt.plot(zeta, h_num.real, '--')
# plt.plot(zeta, h_num.imag, '--')
# plt.tight_layout()