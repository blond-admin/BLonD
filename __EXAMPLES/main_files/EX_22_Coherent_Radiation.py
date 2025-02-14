# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example script to plot the different CSR impedances

:Authors: **Markus Schwarz*

"""

import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import e as elCharge

from blond.beam.beam import Electron
from blond.impedances.impedance_sources import CoherentSynchrotronRadiation
# to compute induced voltage from BLonD routines
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.distributions import bigaussian
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage


mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_22_fig/', exist_ok=True)


h, r_bend = 32e-3, 1.273  # chamber height [m], bending radius [m]
energy, intensity = 40e6, 7e6  # beam energy [eV] and intensity
sigma_dt = 3e-12  # bunch duration [s]
gamma = energy / Electron().mass  # Lorentz factor
n_macroparticles = 1e6  # macro particles for beam object

# frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
freqs = 10**np.linspace(8, 15, num=200)

# approximate free-space CSR impedance increases as f^2/3
Z_fs_appr = CoherentSynchrotronRadiation(r_bend)
Z_fs_appr.imped_calc(freqs)

# The exact free-space CSR impedance increases as f^2/3 up to the critical frequency...
# ... and is exponentially suppressed above.
Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
Z_fs.imped_calc(freqs, high_frequency_transition=10)

f_crit = Z_fs.f_crit  # critical frequency [Hz]

# The approximate parallel-plates impedance is suppressed below the cut-off frequency...
# ... and approaches the approximate free-space CSR impedance for large frequencies.
Z_pp_appr = CoherentSynchrotronRadiation(r_bend, chamber_height=h)
Z_pp_appr.imped_calc(freqs, high_frequency_transition=10)

f_cut = Z_pp_appr.f_cut  # cut-off frequency [Hz]

# The parllel-plates impedance is supressed below the cut-off frequency, and approaches...
# ... the exact free-space impedance for larger frequencies
Z_pp = CoherentSynchrotronRadiation(r_bend, chamber_height=h, gamma=gamma)
Z_pp.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=10)

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
plt.axvline(f_crit, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(f_cut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
plt.savefig (this_directory + '../output_files/EX_22_fig/csr_impedance_real.png')

plt.figure('im Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Im$Z$ / k$\Omega$')
plt.ylim(-5.1, 10.1)
plt.plot(freqs, Z_fs.impedance.imag / 1e3, '.-', label='free-space')
plt.plot(freqs, Z_pp.impedance.imag / 1e3, label='parallel-plates')
plt.plot(freqs, Z_fs_appr.impedance.imag / 1e3, label='free-space, approx.')
plt.plot(freqs, Z_pp_appr.impedance.imag / 1e3, '--', label='parallel-plates, approx.')
plt.axvline(f_crit, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(f_cut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
plt.savefig(this_directory + '../output_files/EX_22_fig/csr_impedance_imag.png')

# compute the energy loss per turn as a cross-check
energy_loss = np.trapz(Z_fs.impedance.real, freqs) * elCharge**2  # [J]
energy_loss *= 2  # take negative frequencies into account
print(f'energy loss per turn (integrated spectrum): {energy_loss / elCharge:1.3f} eV')

# compare to textbook result
print(f"energy loss per turn (Sand's constant): {Electron().c_gamma * (40e6)**4 / r_bend:1.3f} eV")

# create beam with above parameters
ring = Ring(43.2, 14.8e-3, energy, Electron(),
            synchronous_data_type='total energy', n_turns=1)
rf_station = RFStation(ring, 72, 500e3, 0, n_rf=1)
beam = Beam(ring, n_macroparticles, intensity)
bigaussian(ring, rf_station, beam, sigma_dt)
profile = Profile(beam, CutOptions = CutOptions(cut_left=-6*sigma_dt, cut_right=6*sigma_dt,
                                                n_slices=128))

# create the CSR impedance sources
cSR_source_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
cSR_source_pp = CoherentSynchrotronRadiation(r_bend, chamber_height=h, gamma=gamma)
cSR_source_pp_appr = CoherentSynchrotronRadiation(r_bend, chamber_height=h)
cSR_source_fs_appr = CoherentSynchrotronRadiation(r_bend)

# compute the CSR induced voltage via the frequency domain
freq_res = 0.01e12  # [Hz]
cSR_Z_fs = InducedVoltageFreq(beam, profile, [cSR_source_fs], frequency_resolution=freq_res)
cSR_Z_pp = InducedVoltageFreq(beam, profile, [cSR_source_pp], frequency_resolution=freq_res)
cSR_Z_fs_appr = InducedVoltageFreq(beam, profile, [cSR_source_fs_appr], 
                                   frequency_resolution=freq_res)
cSR_Z_pp_appr = InducedVoltageFreq(beam, profile, [cSR_source_pp_appr],
                                   frequency_resolution=freq_res)
W_fs = TotalInducedVoltage(beam, profile, [cSR_Z_fs])
W_fs_appr = TotalInducedVoltage(beam, profile, [cSR_Z_fs_appr])
W_pp = TotalInducedVoltage(beam, profile, [cSR_Z_pp])
W_pp_appr = TotalInducedVoltage(beam, profile, [cSR_Z_pp_appr])

profile.track()
# compute the induced voltage
for tmp in [W_fs, W_pp, W_fs_appr, W_pp_appr]:
    tmp.induced_voltage_sum()
    
plt.figure('induced voltage',clear=True)
plt.grid()
plt.xlabel('time / ps')
plt.ylabel('induced voltage / a.u.')
plt.plot(W_fs.time_array*1e12, W_fs.induced_voltage, '.-', label='free-space')
plt.plot(W_pp.time_array*1e12, W_pp.induced_voltage, label='parallel-plates')
plt.plot(W_fs_appr.time_array*1e12, W_fs_appr.induced_voltage, 
         label='free-space, approx.')
plt.plot(W_pp_appr.time_array*1e12, W_pp_appr.induced_voltage, '--', 
         label='parallel-plates, approx.')
plt.legend()
plt.twinx()
plt.ylabel('bunch profile / a.u.', color='grey')
plt.tick_params('y', labelcolor='grey')
plt.plot(profile.bin_centers*1e12, profile.n_macroparticles/beam.n_macroparticles, 'grey')
plt.tight_layout()
plt.savefig(this_directory + '../output_files/EX_22_fig/inducedvoltages.png')


# times at which to compute the wake potentials
times = np.linspace(profile.edges[0], profile.edges[-1], num=201)

# analytic bunch spectrum
Lambda = np.exp(-0.5*(2*np.pi*freqs*sigma_dt)**2)

w_fs = np.zeros_like(times)
w_pp = np.zeros_like(w_fs)
w_fs_appr = np.zeros_like(w_fs)
w_pp_appr = np.zeros_like(w_fs)
# compute the wake potentials
for it, t in enumerate(times):
    w_fs[it] = np.trapz(Z_fs.impedance*Lambda*np.exp(2j*np.pi*freqs*t), freqs).real
    w_pp[it] = np.trapz(Z_pp.impedance*Lambda*np.exp(2j*np.pi*freqs*t), freqs).real
    w_fs_appr[it] = np.trapz(Z_fs_appr.impedance*Lambda*np.exp(2j*np.pi*freqs*t), freqs).real
    w_pp_appr[it] = np.trapz(Z_pp_appr.impedance*Lambda*np.exp(2j*np.pi*freqs*t), freqs).real
# convert to volt
for tmp in [w_fs, w_pp, w_fs_appr, w_pp_appr]:
    tmp *= 2 * elCharge * intensity

plt.figure('wake', clear=True)
plt.grid()
plt.xlabel('time / ps')
plt.ylabel('wake potential / V')
plt.plot(times*1e12, w_fs, '.-', label='free-space')
plt.plot(times*1e12, w_pp, label='parallel-plates')
plt.plot(times*1e12, w_fs_appr, label='free-space, approx.')
plt.plot(times*1e12, w_pp_appr, '--', label='parallel-plates, approx.')
plt.legend()
plt.twinx()
plt.ylabel('bunch profile / a.u.', color='grey')
plt.tick_params('y', labelcolor='grey')
plt.plot(times*1e12, np.exp(-0.5*(times/sigma_dt)**2), 'grey')
plt.tight_layout()
plt.show()
plt.savefig(this_directory + '../output_files/EX_22_fig/wakepotentials.png')
