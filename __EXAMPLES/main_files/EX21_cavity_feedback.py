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
from llrf.signal_processing import rf_beam_current, low_pass_filter
from llrf.impulse_response import triangle


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


# Logger for messages on console & in file
#Logger().disable()
Logger(debug = True)

# Set up machine parameters
ring = Ring(N_t, C, alpha, p_s)
print("Machine parameters set!")

# Set up RF parameters
RF = RFStation(ring, 1, h, V, phi)
print("RF parameters set!")

# Define beam and fill it
bunch = Beam(ring, N_m, N_b)
bigaussian(ring, RF, bunch, 3.2e-9/4, seed = 1234, 
           reinsertion = True) 
print("Beam set! Number of particles %d" %len(bunch.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(bunch.dt), 
                                                     np.max(bunch.dt)))

#profile = Profile(RF, bunch, 100, cut_left=-1.e-9, cut_right=6.e-9)
profile = Profile(bunch, CutOptions = CutOptions(cut_left=-1.e-9, cut_right=6.e-9, n_slices = 100))
profile.track()
#plt.plot(Slices.bin_centers, Slices.n_macroparticles)
#plt.show()
#Q_tot = Beam.intensity*Beam.charge*e/Beam.n_macroparticles*np.sum(Slices.n_macroparticles)
#print("Total charges %.4e C" %Q_tot)

#print(RFParams.omega_rf[0][0]) # To be CORRECTED in RFParams!!!
#rf_current = rf_beam_current(Slices, RFParams.omega_rf[0][0], GeneralParams.t_rev[0])
rf_current = rf_beam_current(profile, 2*np.pi*200.222e6, ring.t_rev[0])
# Apply LPF on current
#filtered_1 = low_pass_filter(rf_current.real, 20.e6)
#filtered_2 = low_pass_filter(rf_current.imag, 20.e6)
np.set_printoptions(precision=10)
print(repr(rf_current.imag))
#print(repr(filtered_2))
plt.plot(rf_current.real, 'b')
plt.plot(rf_current.imag, 'r')
#plt.plot(filtered_1, 'turquoise')
#plt.plot(filtered_2, 'orange')
plt.show()


#OTFB = SPSOneTurnFeedback(RFParams, Beam, Slices)

print("")
print("Done!")
