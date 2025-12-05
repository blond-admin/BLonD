#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 17:46:40 2025

@author: MarkusArbeit
"""

import numpy as np
np.set_printoptions(legacy="1.25")
from matplotlib import pyplot as plt
from blond.input_parameters.ring import Ring
from blond.beam.beam import Electron, Beam
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance_sources import CoherentSynchrotronRadiation
from blond.impedances.impedance import InducedVoltageFreq, \
    TotalInducedVoltage
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from scipy.constants import e
from blond.beam.distributions import Haissinski

import os

# Directory to save files
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
results_directory = this_directory + "../output_files/EX_28"
os.makedirs(results_directory, exist_ok=True)

seed = 1789 * 1989

V0 = 1.0e6
h = 184

n_turns = int(1*300)  # T_s ~ 300 turns
energy = 1.3e9

n_bins = 128
n_macroparticles = int(1e6)

current = 2 * 0.1e-3  # [A], threshold for V=1MV, alpha0=5e-4 at ~ 0.2 mA

Rbend = 5.559  # [m]
dipole_chamber_height = 32e-3  # [m]

alpha0 = 5e-4

ring = Ring(110.4, alpha0, energy, Electron(), n_turns=n_turns, 
            synchronous_data_type='total energy')
tRev = ring.t_rev[0]


beam = Beam(ring, n_macroparticles, current * tRev / e)

iSR = SynchrotronRadiation(ring, RFStation(ring, h, V0, 0), beam, Rbend, n_kicks=1, 
                           quantum_excitation=True, seed=seed, shift_beam=False)
# equilibrium energy spread [eV]
sigmaE = energy * iSR.sigma_dE  

# shift rf phase to synchronous phase
rf_station = RFStation(ring, h, V0, 
                       np.arcsin(iSR.U0 / ring.particle.charge / V0))
phi_rf = rf_station.phi_rf_d[0,0]
eta0 = ring.eta_0[0,0]
t_rf = rf_station.t_rf[0,0]
omega_rf = rf_station.omega_rf_d[0,0]
fs0 = rf_station.omega_s0[0]/2/np.pi

# zero-current bunch length [s]
sigma0 = np.abs(ring.eta_0[0,0]) / (ring.beta[0,0]*ring.energy[0,0]) \
    * sigmaE / rf_station.omega_s0[0]

profile = Profile(beam, cut_options = CutOptions(cut_left=-5*sigma0, cut_right=5*sigma0,
                                                n_slices=n_bins))
# free-space CSR impedance
cSR_source = CoherentSynchrotronRadiation(Rbend)
cSR_impedance = InducedVoltageFreq(beam, profile, [cSR_source],
                                   frequency_resolution=1/np.sqrt(2)/np.pi/sigma0 / 8)
inducedVoltage = TotalInducedVoltage(beam, profile, [cSR_impedance])

tracker = RingAndRFTracker(rf_station, beam, profile=profile,
                           total_induced_voltage=inducedVoltage,
                           interpolation=True)
ring_tracker = FullRingAndRF([tracker])

haissinski_solution = Haissinski(ring_tracker, iSR, verbose=True, seed=seed)

# create profile for plotting
profile.track()

fig_h, ax_h = plt.subplots(num='Haissinski', clear=True)
ax_h.grid(True)
ax_h.set_xlabel('time / ps')
ax_h.set_ylabel('peak current / A')
ax_h.plot(profile.bin_centers*1e12, haissinski_solution.x * beam.intensity * e, 
         'C1', label='Haissinski')
ax_h.plot(profile.bin_centers*1e12, 
          profile.n_macroparticles / n_macroparticles / profile.bin_size * beam.intensity * e, 
          'C2--', label='initial profile')

init_beam_dt = np.copy(beam.dt)
init_beam_dE = np.copy(beam.dE)


fig_di, ax_di = plt.subplots(num='distribution before tracking', clear=True)
ax_di.set_xlabel('time / ps')
ax_di.set_ylabel(r'$\Delta E$ / MeV')
ax_di.hist2d(init_beam_dt*1e12, init_beam_dE/1e6, bins=n_bins)
fig_di.tight_layout()
fig_di.savefig(results_directory + "/figure_final_distribution.png")

#%%
### Tracker object
frodo_dt = np.zeros(n_turns)
frodo_dE = np.zeros(n_turns)

ind = np.argmin(np.abs(beam.dt - sigma0) * np.abs(beam.dE - 0))

for turn in range(n_turns):
    frodo_dt[turn] = beam.dt[ind]
    frodo_dE[turn] = beam.dE[ind]
    profile.track()

    # compute induced voltage; kick applied later by tracker.track()    
    inducedVoltage.induced_voltage_sum()

    # energy loss due to incoherent SR
    iSR.track()
    # beam.dE -= iSR.U0

    # kick and drift
    tracker.track()

#%%
ax_h.plot(profile.bin_centers*1e12, 
         profile.n_macroparticles / n_macroparticles / profile.bin_size * beam.intensity * e, 
         'C3.', label='final profile')
fig_h.legend()
fig_h.tight_layout()
fig_h.savefig(results_directory + "/figure_Haissinksi_profile.png")


#%%
indexes = (beam.dt > profile.cut_left) * (beam.dt < profile.cut_right)
fig_df, ax_df = plt.subplots(num='distribution after tracking', clear=True)
ax_df.grid(True)
ax_df.set_xlabel('time / ps')
ax_df.set_ylabel(r'$\Delta E$ / MeV')
ax_df.hist2d(beam.dt[indexes]*1e12, beam.dE[indexes]/1e6, bins=n_bins)
ax_df.plot(frodo_dt*1e12, frodo_dE/1e6, 'C3.')
fig_df.tight_layout()
fig_df.savefig(results_directory + "/figure_final_distribution.png")
