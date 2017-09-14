# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for llrf.filters and llrf.cavity_feedback

:Authors: **Helga Timko**
"""

import h5py
import numpy as np
from scipy.constants import e
import matplotlib.pyplot as plt
from matplotlib import gridspec
import logging

from toolbox.logger import Logger
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from trackers.tracker import RingAndRFTracker
from impedances.impedance_sources import TravelingWaveCavity
from impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from plots.plot_beams import plot_long_phase_space


# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                    # 200 MHz system harmonic
V = [4.5e6] #2.2e6                   # 200 MHz RF voltage
phi = [0.]                    # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 25                    # Number of turns to track
N_pretrack = 25             # Number of turns to pre-track
n_bunches = 144             # Number of bunches
bunch_spacing = 5           # In RF buckets
# CERN SPS --------------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')  
plt.rc('legend', fontsize=12)  
# Colors
jet= plt.get_cmap('jet')
colors = jet(np.linspace(0,1,N_t))


# Logger for messages on console & in file
Logger(debug = True)

# Set up machine parameters
ring = Ring(N_t, C, alpha, p_s, Particle=Proton())
print("Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, 1, h, V, phi)
#rf.omega_rf[0,0] = 2*np.pi*200.222e6 # cavity central frequency
logging.debug("RF frequency %.6e Hz", rf.omega_rf[0,0]/(2*np.pi))
logging.debug("Revolution period %.6e s", rf.t_rev[0])
print("RF parameters set!")

# Single bunch
bunch = Beam(ring, N_m, N_b)
bigaussian(ring, rf, bunch, 3.2e-9/4, seed = 1234, reinsertion = True) 

# Create beam
beam = Beam(ring, n_bunches*N_m, n_bunches*N_b)
for i in range(n_bunches):
    beam.dt[int(i*N_m):int((i+1)*N_m)] = bunch.dt + i*rf.t_rf[0]*bunch_spacing
    beam.dE[int(i*N_m):int((i+1)*N_m)] = bunch.dE

profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9, 
    cut_right=rf.t_rev[0], n_slices=46200))#4620*2))
profile.track()
print(profile.Beam.ratio)

plot_long_phase_space(ring, rf, beam, 0, 5e-9, -2e8, 2e8, 
                      dirname='fig', alpha=0.5, color=colors[0])


# # # Compare with induced voltage from impedances
# TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 3.899e-6)
# TWC200_5 = TravelingWaveCavity(1.38e6, 200.222e6, 4.897e-6)
# indVoltageTWC = InducedVoltageTime(beam, profile, [TWC200_4, TWC200_4,
#                                                    TWC200_5, TWC200_5])
# indVoltage = TotalInducedVoltage(beam, profile, [indVoltageTWC])
# indVoltage.induced_voltage_sum()
# plt.figure()
# plt.plot(indVoltage.time_array, indVoltage.induced_voltage, 
#          label='Time domain w FFT')
# plt.show()


OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.5, a_comb=15/16, 
    turns=N_pretrack, Commissioning=CavityFeedbackCommissioning(debug=True))

tracker = RingAndRFTracker(rf, beam, CavityFeedback=OTFB, interpolation=True, 
                           Profile=profile)




map_ = [profile] + [OTFB] + [tracker] 


# Plot 1: cavity voltage
fig1 = plt.figure(4, figsize=(8,10))
gs1 = gridspec.GridSpec(2, 1) 
ax1_1 = plt.subplot(gs1[0])
ax1_2 = plt.subplot(gs1[1], sharex=ax1_1)
plt.setp(ax1_1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1_1.yaxis.get_major_ticks()
yticks[0].set_visible(False)
plt.subplots_adjust(hspace=.0)
ax1_1.set_ylabel(r"$Re(V_{\mathsf{cav}})$ [MV]")
ax1_2.set_xlabel(r"Time [$\mu$s]")
ax1_2.set_ylabel(r"$Im(V_{\mathsf{cav}})$ [MV]")
ax1_1.set_ylim((-1,5))
ax1_2.set_ylim((0,7))
 
 
# Plot 2: cavity voltage
fig2 = plt.figure(5, figsize=(8,10))
gs2 = gridspec.GridSpec(2, 1) 
ax2_1 = plt.subplot(gs2[0])
ax2_2 = plt.subplot(gs2[1], sharex=ax2_1)
plt.setp(ax2_1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax2_1.yaxis.get_major_ticks()
yticks[0].set_visible(False)
plt.subplots_adjust(hspace=.0)
ax2_1.set_ylabel(r"$Re(V_{\mathsf{ind,beam}})$ [MV]")
ax2_2.set_xlabel(r"Time [$\mu$s]")
ax2_2.set_ylabel(r"$Im(V_{\mathsf{ind,beam}})$ [MV]")
ax2_1.set_ylim((-1,5))
ax2_2.set_ylim((-1,0.4))
 
 
OTFB.OTFB_4.V_ind_beam = np.zeros(profile.n_slices)
OTFB.OTFB_5.V_ind_beam = np.zeros(profile.n_slices)



print("Starting to track...")
for i in range(N_t):

        
    ax1_1.plot(1e6*profile.bin_centers, 1e-6*OTFB.V_sum.real, color=colors[i])
    ax1_1.fill_between(1e6*profile.bin_centers, 0, 1e-6*OTFB.V_sum.real, alpha=0.2, color=colors[i])
    ax1_2.plot(1e6*profile.bin_centers, 1e-6*OTFB.V_sum.imag, color=colors[i])
    ax1_2.fill_between(1e6*profile.bin_centers, 0, 1e-6*OTFB.V_sum.imag, alpha=0.2, color=colors[i])
    fig1.savefig("fig/V_ant_" + "%d" %(i+N_pretrack+1) + ".png")
 
     
    ax2_1.plot(1e6*profile.bin_centers, 1e-6*OTFB.OTFB_4.V_ind_beam.real + 1e-6*OTFB.OTFB_5.V_ind_beam.real, color=colors[i])
    ax2_1.fill_between(1e6*profile.bin_centers, 0, 1e-6*OTFB.OTFB_4.V_ind_beam.real + 1e-6*OTFB.OTFB_5.V_ind_beam.real, alpha=0.2, color=colors[i])
    ax2_2.plot(1e6*profile.bin_centers, 1e-6*OTFB.OTFB_4.V_ind_beam.imag + 1e-6*OTFB.OTFB_5.V_ind_beam.imag, color=colors[i])
    ax2_2.fill_between(1e6*profile.bin_centers, 0, 1e-6*OTFB.OTFB_4.V_ind_beam.imag + 1e-6*OTFB.OTFB_5.V_ind_beam.imag, alpha=0.2, color=colors[i])
    fig2.savefig("fig/V_ind_beam_" + "%d" %(i) + ".png")
    
    # Track
    for m in map_:
        m.track()

    plot_long_phase_space(ring, rf, beam, 0, 5e-9, -2e8, 2e8, 
                          dirname='fig', alpha=0.5, color=colors[i])

    
#plt.show()
       

