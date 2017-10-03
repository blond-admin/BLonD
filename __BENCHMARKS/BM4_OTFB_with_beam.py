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
N_t = 30                    # Number of turns to track
n_bunches = 144             # Number of bunches
bunch_spacing = 5           # In RF buckets
# CERN SPS --------------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)

plt.rc('font', family='sans-serif')  
plt.rc('legend', fontsize=12)  


# Logger for messages on console & in file
Logger(debug = True)

# Set up machine parameters
ring = Ring(N_t, C, alpha, p_s, Particle=Proton())
print("Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, 1, h, V, phi)
rf.omega_rf[0,0] = 2*np.pi*200.222e6 # cavity central frequency
logging.debug("RF frequency %.6e Hz", rf.omega_rf[0,0]/(2*np.pi))
logging.debug("Revolution period %.6e s", rf.t_rev[0])
print("RF parameters set!")

# Single bunch
bunch = Beam(ring, N_m, N_b)
bigaussian(ring, rf, bunch, 3.2e-9/4, seed = 1234, reinsertion = True) 


# # Load PS beam to clone
# #PS_folder = 'C:/Users/schwarz/git/BLonD_PS_SPS/models/PS_SPS_model/test/'
# PS_case = '2x40_2x80'
# #with h5py.File(PS_folder + PS_case + '.hd5', 'r') as data_file:
# with h5py.File(PS_case + '.hd5', 'r') as data_file:
#     n_macroparticles_PS = data_file['n_macroparticles'].value
#     
#     #get time and energy coordinates of particles
#     PS_dt = data_file['dt'][:]
#     PS_dE = data_file['dE'][:]
#     
#     data_file.close()
#     
#     PS_dt -= np.mean(PS_dt) #make sure PS bunch is centered at 0
# print('PS beam loaded')

# # how many clones and where to put them
# n_bunches_PS = 1    # one bunch from the PS
# bunch_copies = 72   # how many times to clone the PS bunches
# bunch_spacing = 5   # how many SPS RF buckets between bunches in the SPS
# 
# n_macroparticles_pb = len(PS_dt)  # number of macro-particles per SPS bunch
# 
# n_bunches = n_bunches_PS * bunch_copies # number of bunches in the SPS (72)
# intensity = n_bunches * N_b     # total intensity SPS
# n_macroparticles = n_macroparticles_pb * bunch_copies
# 
# t_rf = 2*np.pi/rf.omega_rf_d[0,0]
# 
# PS_dt += t_rf/2 #shift PS beam to center of SPS bucket
# 
# # Create SPS beam
# beam = Beam(ring, n_macroparticles, intensity)

# # Create additional SPS bunches by cloning the PS beam
# for it in range(bunch_copies):
#     # Place SPS bunch at correct RF bucket
#     beam.dt[int(it*n_macroparticles/bunch_copies) \
#                 :int((it+1)*n_macroparticles/bunch_copies)] \
#     = PS_dt + it * t_rf * n_bunches_PS*bunch_spacing
#     
#     # Set energy of SPS bunch
#     beam.dE[int(it*n_macroparticles/bunch_copies) \
#                 :int((it+1)*n_macroparticles/bunch_copies)] \
#     = PS_dE
# print("Beam set! Number of particles %d" %len(beam.dt))


# Create beam
beam = Beam(ring, n_bunches*N_m, n_bunches*N_b)
for i in range(n_bunches):
    beam.dt[int(i*N_m):int((i+1)*N_m)] = bunch.dt + i*rf.t_rf[0]*bunch_spacing
    beam.dE[int(i*N_m):int((i+1)*N_m)] = bunch.dE

profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9, 
    cut_right=rf.t_rev[0], n_slices=4620))
profile.track()

# # Compare with induced voltage from impedances
# TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 3.899e-6)
# TWC200_5 = TravelingWaveCavity(1.38e6, 200.222e6, 4.897e-6)
# indVoltageTWC = InducedVoltageTime(beam, profile, [TWC200_4, TWC200_4, TWC200_5, TWC200_5])
# indVoltage = TotalInducedVoltage(beam, profile, [indVoltageTWC])
# indVoltage.induced_voltage_sum()
# plt.figure()
# plt.plot(indVoltage.time_array, indVoltage.induced_voltage, 
#          label='Time domain w FFT')
# plt.show()


OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.5, a_comb=15/16, 
                         turns=50, Commissioning=CavityFeedbackCommissioning())

tracker = RingAndRFTracker(rf, beam, CavityFeedback=OTFB, interpolation=True, 
                           Profile=profile)




map_ = [profile] + [OTFB] + [tracker] 


plt.figure(4)
ax4 = plt.axes()

plt.figure(5)
ax5 = plt.axes()

plt.figure(6)
ax6 = plt.axes()

print("Starting to track...")
for i in range(N_t):

    # Track
    for m in map_:
        m.track()
        
    ax4.plot(profile.bin_centers, OTFB.V_sum.real)
    ax4.plot(profile.bin_centers, OTFB.V_sum.imag, ':')
    ax5.plot(profile.bin_centers, OTFB.OTFB_4.V_ind_beam.real + OTFB.OTFB_5.V_ind_beam.real)
    ax5.plot(profile.bin_centers, OTFB.OTFB_4.V_ind_beam.imag + OTFB.OTFB_5.V_ind_beam.imag, ':')
    ax6.plot(profile.bin_centers, OTFB.OTFB_4.I_beam + OTFB.OTFB_5.I_beam.real)
    ax6.plot(profile.bin_centers, OTFB.OTFB_4.I_beam.imag + OTFB.OTFB_5.I_beam.imag, ':')
    
plt.show()
       

