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

freqs = np.linspace(8,15, num=400)
# freqs = np.linspace(11.32,11.8, num=30)
freqs = 10**freqs

Z_fs_low = CoherentSynchrotronRadiation(rBend)
Z_fs_low.imped_calc(freqs)

Z_fs = CoherentSynchrotronRadiation(rBend, gamma=gamma)
# Z_fs.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=1e2)
Z_fs.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=np.inf)

# Z_pp_low = CoherentSynchrotronRadiation(rBend, chamber_height=h)
# Z_pp_low.imped_calc(freqs, pMax=634)

# Z_pp_low2 = CoherentSynchrotronRadiation(rBend, chamber_height=h)
# Z_pp_low2.imped_calc = Z_pp_low2._pp_low_frequency2
# Z_pp_low2.imped_calc(freqs)

# Z_pp_low3 = CoherentSynchrotronRadiation(rBend, chamber_height=h)
# Z_pp_low3.imped_calc = Z_pp_low3._pp_low_frequency3
# Z_pp_low3.imped_calc(freqs, u_max=5)

Z_pp_low = CoherentSynchrotronRadiation(rBend, chamber_height=h)
Z_pp_low.imped_calc(freqs, high_frequency_transition=10)


plt.figure('re Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-0.1, 12.1)
plt.plot(freqs, Z_fs_low.impedance.real / 1e3, label='free space, low-freq')
plt.plot(freqs, Z_fs.impedance.real / 1e3, label='free space, full')
plt.plot(freqs, Z_pp_low.impedance.real / 1e3, label='parallel plates, low-freq')
# plt.plot(freqs, Z_pp_low2.impedance.real / 1e3, label='parallel plates 2, low-freq')
# plt.plot(freqs, Z_pp_low3.impedance.real / 1e3, '.', label='parallel plates 3, low-freq')
plt.axvline(fc, color='red', linestyle='--')
plt.axvline(fcut, linestyle='--')
plt.legend()
plt.tight_layout()

plt.figure('im Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-10.1, 6.1)
plt.plot(freqs, Z_fs_low.impedance.imag / 1e3, label='free space, low-freq')
plt.plot(freqs, Z_fs.impedance.imag / 1e3, label='free space, full')
plt.plot(freqs, Z_pp_low.impedance.imag / 1e3, label='parallel plates, low-freq')
# plt.plot(freqs, Z_pp_low2.impedance.imag / 1e3, label='parallel plates 2, low-freq')
# plt.plot(freqs, Z_pp_low3.impedance.imag / 1e3, '.', label='parallel plates 3, low-freq')
# plt.axvline(fc, color='red', linestyle='--')
# plt.axvline(fcut, linestyle='--')
plt.legend()
plt.tight_layout()