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

import numpy as np
from scipy.constants import e
import matplotlib.pyplot as plt 

from toolbox.logger import Logger
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.cavity_feedback import SPSOneTurnFeedback
from llrf.signal_processing import rf_beam_current #, low_pass_filter
#from llrf.impulse_response import triangle
from impedances.impedance_sources import TravelingWaveCavity
from llrf.impulse_response import SPS4Section200MHzTWC
#from impedances.impedance_sources import InputTable
from impedances.impedance import InducedVoltageTime, TotalInducedVoltage


# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = 4620                    # 200 MHz system harmonic
V = 4.5e6                   # 200 MHz RF voltage
phi = 0.                    # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1000                  # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# OPTIONS TO TEST -------------------------------------------------------------
LOGGING = True              # Logging messages
RF_CURRENT = True           # RF beam current
TWC = True                  # Impulse response of travelling wave cavity
VIND_BEAM = True            # Beam-induced voltage

# OPTIONS TO TEST -------------------------------------------------------------

# Logger for messages on console & in file
if LOGGING == True:
    Logger(debug = True)
else:
    Logger().disable()

# Set up machine parameters
ring = Ring(N_t, C, alpha, p_s)
print("Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, 1, h, V, phi)
print("RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, 
           reinsertion = True) 
print("Beam set! Number of particles %d" %len(beam.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt), 
                                                     np.max(beam.dt)))

profile = Profile(beam, CutOptions = CutOptions(cut_left=-1.e-9, 
    cut_right=6.e-9, n_slices = 100))
profile.track()

if RF_CURRENT == True:
    rf_current = rf_beam_current(profile, 2*np.pi*200.222e6, ring.t_rev[0])
    np.set_printoptions(precision=10)
    #print(repr(rf_current.real))
    plt.figure(1)
    plt.plot(profile.bin_centers, rf_current.real, 'b')
    plt.plot(profile.bin_centers, rf_current.imag, 'r')
    plt.plot(profile.bin_centers, profile.n_macroparticles, 'g')


if TWC == True:
    time = np.linspace(-1e-6, 4.e-6, 10000)
    impResp = SPS4Section200MHzTWC()
    impResp.impulse_response(2*np.pi*195.e6, time)
    #print(impResp.t_beam[1] - impResp.t_beam[0])
    #print(len(impResp.t_beam))
    #print(impResp.tau)
    #print(3.56e-6/2/np.pi)
    TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 2*np.pi*6.207e-7)
    TWC200_4.wake_calc(time)
    plt.figure(2)
    plt.plot(TWC200_4.time_array, TWC200_4.wake, 'b')
    plt.plot(impResp.t_beam, impResp.W_beam, 'r')
    plt.plot(impResp.t_beam, impResp.hs_beam, 'g')
    plt.plot(impResp.t_gen, impResp.hs_gen, 'purple', marker='.')


if VIND_BEAM == True:
    OTFB = SPSOneTurnFeedback(rf, beam, profile)
    OTFB.counter = 0 # First turn
    OTFB.omega_c = rf.omega_rf[0,0] #2*np.pi*200.222e6  
    OTFB.beam_induced_voltage()
    plt.figure(3)
    convtime = np.linspace(-1e-9, -1e-9+len(OTFB.Vind_beam.real)*
                           profile.bin_size, len(OTFB.Vind_beam.real))
#    plt.plot(profile.bin_centers, OTFB.Vind_beam.real, 'b')
#    plt.plot(profile.bin_centers, OTFB.Vind_beam.imag, 'r')
    plt.plot(convtime, OTFB.Vind_beam.real, 'b--')
    plt.plot(convtime[:100], OTFB.Vind_beam.real[:100], 'b')
    plt.plot(convtime, OTFB.Vind_beam.imag, 'r--')
    plt.plot(convtime[:100], OTFB.Vind_beam.imag[:100], 'r')
#    plt.plot(convtime[:100], np.abs(OTFB.Vind_beam)[:100]*np.cos(OTFB.omega_c*convtime[:100]), color='purple')
    plt.plot(convtime[:100], -OTFB.Vind_beam.real[:100]*np.cos(OTFB.omega_c*convtime[:100])-OTFB.Vind_beam.imag[:100]*np.sin(OTFB.omega_c*convtime[:100]), color='purple')
    
    # Comparison with impedances
    TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 3.899e-6)
    TWC200_5 = TravelingWaveCavity(1.4634e6, 200.222e6, 4.897e-6)
    indVoltageTWC = InducedVoltageTime(beam, profile, [TWC200_4, TWC200_4, TWC200_5, TWC200_5])
    indVoltage = TotalInducedVoltage(beam, profile, [indVoltageTWC])
    indVoltage.induced_voltage_sum()
    #plt.plot(indVoltage.time_array, indVoltage.induced_voltage*np.max(np.abs(OTFB.Vind_beam))/np.max(indVoltage.induced_voltage), 'g')
    plt.plot(indVoltage.time_array, indVoltage.induced_voltage*3, 'g')
    print(len(indVoltage.induced_voltage))

plt.show()
print("")
print("Done!")



