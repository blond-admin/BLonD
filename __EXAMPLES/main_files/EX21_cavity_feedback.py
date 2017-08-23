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
RF_CURRENT = False          # RF beam current
TWC = False                 # Impulse response of travelling wave cavity
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
bunch = Beam(ring, N_m, N_b)
bigaussian(ring, rf, bunch, 3.2e-9/4, seed = 1234, 
           reinsertion = True) 
print("Beam set! Number of particles %d" %len(bunch.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(bunch.dt), 
                                                     np.max(bunch.dt)))

profile = Profile(bunch, CutOptions = CutOptions(cut_left=-1.e-9, cut_right=6.e-9, n_slices = 100))
profile.track()

if RF_CURRENT == True:
    rf_current = rf_beam_current(profile, 2*np.pi*200.222e6, ring.t_rev[0])
    np.set_printoptions(precision=10)
    print(repr(rf_current.real))
    plt.plot(profile.bin_centers, rf_current.real, 'b')
    plt.plot(profile.bin_centers, rf_current.imag, 'r')
    plt.plot(profile.bin_centers, profile.n_macroparticles, 'g')
    plt.show()


if TWC == True:
    time = np.linspace(-1e-6, 4.e-6, 10000)
    impResp = SPS4Section200MHzTWC()
    impResp.impulse_response(2*np.pi*200.e6, time)
    print(impResp.tau)
    print(3.56e-6/2/np.pi)
    TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 2*np.pi*6.207e-7)
    TWC200_4.wake_calc(time)
    plt.plot(TWC200_4.time_array, TWC200_4.wake, 'b')
    plt.plot(impResp.time, impResp.W_beam, 'r')
    plt.plot(impResp.time, impResp.hs_beam, 'g')
    plt.plot(impResp.time, impResp.hs_gen, 'purple', marker='.')
    plt.show()


if VIND_BEAM == True:
    OTFB = SPSOneTurnFeedback(rf, bunch, profile)
    OTFB.counter = 0 # First turn
    OTFB.omega_c = rf.omega_rf[0,0]
    OTFB.beam_induced_voltage()
    plt.plot(profile.bin_centers, OTFB.Vind_beam.real, 'b')
    plt.plot(profile.bin_centers, OTFB.Vind_beam.imag, 'r')
    plt.show()

print("")
print("Done!")



